#include "rgdic.h"
#include <iostream>
#include <functional>
#include <atomic>
#include <chrono>

// Constructor with computation mode parameter
RGDIC::RGDIC(int subsetRadius, double convergenceThreshold, int maxIterations,
           double ccThreshold, double deltaDispThreshold, ShapeFunctionOrder order,
           ComputationMode mode, int numThreads)
    : m_subsetRadius(subsetRadius),
      m_convergenceThreshold(convergenceThreshold),
      m_maxIterations(maxIterations),
      m_ccThreshold(ccThreshold),
      m_deltaDispThreshold(deltaDispThreshold),
      m_order(order),
      m_mode(mode)
{
    // Set number of parameters based on shape function order
    m_numParams = (order == FIRST_ORDER) ? 6 : 12;
    
    // Set number of threads for parallel processing
    if (numThreads <= 0) {
        m_numThreads = std::thread::hardware_concurrency();
    } else {
        m_numThreads = numThreads;
    }
    
    // Initialize OpenMP threads
    omp_set_num_threads(m_numThreads);
    
    // Initialize CUDA if using GPU mode
    if (mode == GPU_CUDA || mode == HYBRID) {
        RGDIC_CUDA::initializeGPU();
        RGDIC_CUDA::printGPUInfo();
        
        if (!RGDIC_CUDA::checkGPUCompatibility()) {
            std::cerr << "Warning: GPU compatibility issues detected. Falling back to CPU mode." << std::endl;
            m_mode = CPU_MULTI_THREAD;
        }
    }
    
    std::cout << "RGDIC initialized with " << m_numThreads << " threads and computation mode: ";
    switch (m_mode) {
        case CPU_SINGLE_THREAD: std::cout << "CPU single thread"; break;
        case CPU_MULTI_THREAD: std::cout << "CPU multi-threaded"; break;
        case GPU_CUDA: std::cout << "GPU CUDA"; break;
        case HYBRID: std::cout << "Hybrid CPU/GPU"; break;
    }
    std::cout << std::endl;
}

RGDIC::DisplacementResult RGDIC::compute(const cv::Mat& refImage, const cv::Mat& defImage, const cv::Mat& roi)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Initialize results
    DisplacementResult result;
    result.u = cv::Mat::zeros(roi.size(), CV_64F);
    result.v = cv::Mat::zeros(roi.size(), CV_64F);
    result.cc = cv::Mat::zeros(roi.size(), CV_64F);
    result.validMask = cv::Mat::zeros(roi.size(), CV_8U);
    
    // Calculate signed distance array for ROI
    cv::Mat sda = calculateSDA(roi);
    
    // Find seed point
    cv::Point seedPoint = findSeedPoint(roi, sda);
    
    std::cout << "Starting RGDIC computation with seed point at (" << seedPoint.x << ", " << seedPoint.y << ")" << std::endl;
    
    // Choose appropriate implementation based on computation mode
    switch (m_mode) {
        case CPU_SINGLE_THREAD:
            // Original single-threaded implementation
            {
                // Create the ICGN optimizer
                ICGNOptimizer optimizer(refImage, defImage, m_subsetRadius, m_order, 
                                      m_convergenceThreshold, m_maxIterations, CPU_SINGLE_THREAD);
                
                // Initialize warp parameters matrix
                cv::Mat warpParams = cv::Mat::zeros(m_numParams, 1, CV_64F);
                
                // Compute initial ZNCC
                double initialZNCC = 0.0;
                
                // Process seed point
                if (!optimizer.initialGuess(seedPoint, warpParams, initialZNCC) || 
                    !optimizer.optimize(seedPoint, warpParams, initialZNCC) ||
                    initialZNCC > m_ccThreshold) {
                    std::cerr << "Failed to process seed point." << std::endl;
                    return result;
                }
                
                // Save seed point results
                result.u.at<double>(seedPoint) = warpParams.at<double>(0);
                result.v.at<double>(seedPoint) = warpParams.at<double>(1);
                result.cc.at<double>(seedPoint) = initialZNCC;
                result.validMask.at<uchar>(seedPoint) = 1;
                
                // Define comparator for priority queue
                auto comparator = [](const std::pair<cv::Point, double>& a, const std::pair<cv::Point, double>& b) {
                    return a.second > b.second;
                };
                
                // Create priority queue for reliability-guided search
                PriorityQueue queue(comparator);
                
                // Initialize queue with seed point
                queue.push(std::make_pair(seedPoint, initialZNCC));
                
                // Create analyzed points tracker
                cv::Mat analyzedPoints = cv::Mat::zeros(roi.size(), CV_8U);
                analyzedPoints.at<uchar>(seedPoint) = 1;
                
                // Define 4-connected neighbors
                const cv::Point neighbors[] = {
                    {0, 1}, {1, 0}, {0, -1}, {-1, 0}
                };
                
                // Reliability-guided search
                while (!queue.empty()) {
                    // Get point with highest reliability (lowest ZNCC value)
                    auto current = queue.top();
                    queue.pop();
                    
                    cv::Point currentPoint = current.first;
                    
                    // Check all neighbors
                    for (int i = 0; i < 4; i++) {
                        cv::Point neighborPoint = currentPoint + neighbors[i];
                        
                        // Check if neighbor is within image bounds and ROI
                        if (neighborPoint.x >= 0 && neighborPoint.x < roi.cols &&
                            neighborPoint.y >= 0 && neighborPoint.y < roi.rows &&
                            roi.at<uchar>(neighborPoint) > 0 &&
                            analyzedPoints.at<uchar>(neighborPoint) == 0) {
                            
                            // Mark as analyzed
                            analyzedPoints.at<uchar>(neighborPoint) = 1;
                            
                            // Try to analyze this point
                            analyzePoint(neighborPoint, optimizer, roi, result, queue, analyzedPoints);
                        }
                    }
                }
            }
            break;
            
        case CPU_MULTI_THREAD:
            // Multi-threaded CPU implementation
            reliabilityGuidedSearch(refImage, defImage, roi, seedPoint, result);
            break;
            

    case GPU_CUDA:
    case HYBRID:
        // GPU implementation
        reliabilityGuidedSearchGPU(refImage, defImage, roi, seedPoint, result);
        break;
    }

    // Post-process to remove outliers
    // Create a copy of valid mask
    cv::Mat validMaskCopy = result.validMask.clone();

    // Define 4-connected neighbors
    const cv::Point neighbors[] = {
    {0, 1}, {1, 0}, {0, -1}, {-1, 0}
    };

    // Parallel outlier removal for multi-threaded modes
    if (m_mode != CPU_SINGLE_THREAD) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < result.validMask.rows; y++) {
        for (int x = 0; x < result.validMask.cols; x++) {
            if (result.validMask.at<uchar>(y, x)) {
                // Check displacement jumps with valid neighbors
                bool isOutlier = false;
                
                for (int i = 0; i < 4; i++) {
                    cv::Point neighborPoint(x + neighbors[i].x, y + neighbors[i].y);
                    
                    if (neighborPoint.x >= 0 && neighborPoint.x < roi.cols &&
                        neighborPoint.y >= 0 && neighborPoint.y < roi.rows &&
                        result.validMask.at<uchar>(neighborPoint)) {
                        
                        // Calculate displacement jump
                        double du = result.u.at<double>(y, x) - result.u.at<double>(neighborPoint);
                        double dv = result.v.at<double>(y, x) - result.v.at<double>(neighborPoint);
                        double dispJump = std::sqrt(du*du + dv*dv);
                        
                        // Mark as outlier if displacement jump is too large
                        if (dispJump > m_deltaDispThreshold) {
                            isOutlier = true;
                            break;
                        }
                    }
                }
                
                if (isOutlier) {
                    #pragma omp critical
                    {
                        validMaskCopy.at<uchar>(y, x) = 0;
                    }
                }
            }
        }
    }
    } else {
    // Single-threaded outlier removal
    for (int y = 0; y < result.validMask.rows; y++) {
        for (int x = 0; x < result.validMask.cols; x++) {
            if (result.validMask.at<uchar>(y, x)) {
                // Check displacement jumps with valid neighbors
                bool isOutlier = false;
                
                for (int i = 0; i < 4; i++) {
                    cv::Point neighborPoint(x + neighbors[i].x, y + neighbors[i].y);
                    
                    if (neighborPoint.x >= 0 && neighborPoint.x < roi.cols &&
                        neighborPoint.y >= 0 && neighborPoint.y < roi.rows &&
                        result.validMask.at<uchar>(neighborPoint)) {
                        
                        // Calculate displacement jump
                        double du = result.u.at<double>(y, x) - result.u.at<double>(neighborPoint);
                        double dv = result.v.at<double>(y, x) - result.v.at<double>(neighborPoint);
                        double dispJump = std::sqrt(du*du + dv*dv);
                        
                        // Mark as outlier if displacement jump is too large
                        if (dispJump > m_deltaDispThreshold) {
                            isOutlier = true;
                            break;
                        }
                    }
                }
                
                if (isOutlier) {
                    validMaskCopy.at<uchar>(y, x) = 0;
                }
            }
        }
    }
    }

    // Update result with filtered mask
    result.validMask = validMaskCopy;

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Count valid points
    int validPoints = cv::countNonZero(result.validMask);
    int totalROIPoints = cv::countNonZero(roi);

    std::cout << "RGDIC computation completed in " << duration.count() / 1000.0 << " seconds." << std::endl;
    std::cout << "Valid points: " << validPoints << " out of " << totalROIPoints 
        << " (" << 100.0 * validPoints / totalROIPoints << "%)" << std::endl;

    return result;
    }

    // Multi-threaded implementation of reliability-guided search
    void RGDIC::reliabilityGuidedSearch(const cv::Mat& refImage, const cv::Mat& defImage, 
                            const cv::Mat& roi, cv::Point seedPoint, 
                            DisplacementResult& result)
    {
    // Create the ICGN optimizer
    ICGNOptimizer optimizer(refImage, defImage, m_subsetRadius, m_order, 
                    m_convergenceThreshold, m_maxIterations, CPU_MULTI_THREAD);

    // Initialize warp parameters matrix for seed point
    cv::Mat warpParams = cv::Mat::zeros(m_numParams, 1, CV_64F);
    double initialZNCC = 0.0;

    // Process seed point
    if (!optimizer.initialGuess(seedPoint, warpParams, initialZNCC) || 
    !optimizer.optimize(seedPoint, warpParams, initialZNCC) ||
    initialZNCC > m_ccThreshold) {
    std::cerr << "Failed to process seed point." << std::endl;
    return;
    }

    // Save seed point results
    result.u.at<double>(seedPoint) = warpParams.at<double>(0);
    result.v.at<double>(seedPoint) = warpParams.at<double>(1);
    result.cc.at<double>(seedPoint) = initialZNCC;
    result.validMask.at<uchar>(seedPoint) = 1;

    // Define comparator for priority queue
    auto comparator = [](const std::pair<cv::Point, double>& a, const std::pair<cv::Point, double>& b) {
    return a.second > b.second;
    };

    // Create thread-safe priority queue
    PriorityQueue queue(comparator);
    queue.push(std::make_pair(seedPoint, initialZNCC));

    // Create analyzed points tracker
    cv::Mat analyzedPoints = cv::Mat::zeros(roi.size(), CV_8U);
    analyzedPoints.at<uchar>(seedPoint) = 1;

    // Define 4-connected neighbors
    const cv::Point neighbors[] = {
    {0, 1}, {1, 0}, {0, -1}, {-1, 0}
    };

    // Define a worker function for threads
    auto workerFunction = [&](int threadId) {
    // Create local optimizer for this thread
    ICGNOptimizer localOptimizer(refImage, defImage, m_subsetRadius, m_order, 
                                m_convergenceThreshold, m_maxIterations, CPU_MULTI_THREAD);

    while (true) {
        // Get next point from queue
        cv::Point currentPoint;
        double currentZNCC = 0.0;
        
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            if (queue.empty()) {
                break;  // No more points to process
            }
            
            auto current = queue.top();
            queue.pop();
            currentPoint = current.first;
            currentZNCC = current.second;
        }
        
        // Process neighbors of current point
        for (int i = 0; i < 4; i++) {
            cv::Point neighborPoint = currentPoint + neighbors[i];
            
            // Check if neighbor is within bounds and not yet analyzed
            bool shouldProcess = false;
            {
                std::lock_guard<std::mutex> lock(m_resultMutex);
                if (neighborPoint.x >= 0 && neighborPoint.x < roi.cols &&
                    neighborPoint.y >= 0 && neighborPoint.y < roi.rows &&
                    roi.at<uchar>(neighborPoint) > 0 &&
                    analyzedPoints.at<uchar>(neighborPoint) == 0) {
                    
                    // Mark as analyzed
                    analyzedPoints.at<uchar>(neighborPoint) = 1;
                    shouldProcess = true;
                }
            }
            
            if (shouldProcess) {
                // Local queue for this thread
                PriorityQueue localQueue(comparator);
                
                // Try to analyze this point
                bool success = analyzePoint(neighborPoint, localOptimizer, roi, result, 
                                        localQueue, analyzedPoints);
                
                // If successful, add new points to the global queue
                if (success && !localQueue.empty()) {
                    std::lock_guard<std::mutex> lock(m_queueMutex);
                    while (!localQueue.empty()) {
                        queue.push(localQueue.top());
                        localQueue.pop();
                    }
                }
            }
        }
    }
    };

    // Create and start worker threads
    std::vector<std::thread> threads;
    for (int i = 0; i < m_numThreads; i++) {
    threads.push_back(std::thread(workerFunction, i));
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
    thread.join();
    }
    }

    // GPU implementation of reliability-guided search
    void RGDIC::reliabilityGuidedSearchGPU(const cv::Mat& refImage, const cv::Mat& defImage, 
                                const cv::Mat& roi, cv::Point seedPoint, 
                                DisplacementResult& result)
    {
    // Create the GPU-enabled optimizer
    ICGNOptimizer optimizer(refImage, defImage, m_subsetRadius, m_order, 
                    m_convergenceThreshold, m_maxIterations, GPU_CUDA);

    // Initialize warp parameters matrix for seed point
    cv::Mat warpParams = cv::Mat::zeros(m_numParams, 1, CV_64F);
    double initialZNCC = 0.0;

    // Process seed point using GPU
    if (!optimizer.initialGuessGPU(seedPoint, warpParams, initialZNCC) || 
    !optimizer.optimizeGPU(seedPoint, warpParams, initialZNCC) ||
    initialZNCC > m_ccThreshold) {
    std::cerr << "Failed to process seed point on GPU." << std::endl;
    return;
    }

    // Save seed point results
    result.u.at<double>(seedPoint) = warpParams.at<double>(0);
    result.v.at<double>(seedPoint) = warpParams.at<double>(1);
    result.cc.at<double>(seedPoint) = initialZNCC;
    result.validMask.at<uchar>(seedPoint) = 1;

    // Define comparator for priority queue
    auto comparator = [](const std::pair<cv::Point, double>& a, const std::pair<cv::Point, double>& b) {
    return a.second > b.second;
    };

    // Create priority queue for reliability-guided search
    PriorityQueue queue(comparator);
    queue.push(std::make_pair(seedPoint, initialZNCC));

    // Create analyzed points tracker
    cv::Mat analyzedPoints = cv::Mat::zeros(roi.size(), CV_8U);
    analyzedPoints.at<uchar>(seedPoint) = 1;

    // Define 4-connected neighbors
    const cv::Point neighbors[] = {
    {0, 1}, {1, 0}, {0, -1}, {-1, 0}
    };

    // For GPU mode, we use a hybrid approach:
    // 1. CPU handles coordination of points and queue management
    // 2. GPU handles optimization computations for batches of points

    // Define a batch size for GPU processing
    const int batchSize = 64; // Adjust based on GPU capabilities

    // Create a pool of worker threads for GPU optimization tasks
    std::vector<std::thread> workerThreads;
    std::atomic<bool> stopWorkers(false);

    // Shared data structures for worker threads
    std::mutex batchMutex;
    std::condition_variable batchCondition;
    std::vector<cv::Point> pointBatch;
    std::vector<bool> resultFlags(batchSize, false);
    std::vector<cv::Mat> resultWarpParams(batchSize);
    std::vector<double> resultZncc(batchSize, 0.0);
    std::atomic<int> readyCount(0);
    std::atomic<int> processedCount(0);
    bool batchReady = false;

    // Initialize worker threads
    for (int t = 0; t < m_numThreads; ++t) {
    workerThreads.push_back(std::thread([&, t]() {
        // Create a local GPU optimizer for this thread
        ICGNOptimizer localOptimizer(refImage, defImage, m_subsetRadius, m_order,
                                    m_convergenceThreshold, m_maxIterations, GPU_CUDA);
        
        while (!stopWorkers) {
            // Wait for batch to be ready or for stop signal
            std::unique_lock<std::mutex> lock(batchMutex);
            batchCondition.wait(lock, [&]() { return batchReady || stopWorkers; });
            
            if (stopWorkers) break;
            
            // Get assigned range of points to process
            int startIdx = t * (pointBatch.size() / m_numThreads);
            int endIdx = (t == m_numThreads - 1) ? pointBatch.size() : (t + 1) * (pointBatch.size() / m_numThreads);
            
            // Process assigned points
            for (int i = startIdx; i < endIdx; ++i) {
                if (i >= pointBatch.size()) break;
                
                const cv::Point& point = pointBatch[i];
                cv::Mat localWarpParams = cv::Mat::zeros(m_numParams, 1, CV_64F);
                double localZncc = 0.0;
                
                // Find best neighbor for initial guess
                std::vector<cv::Point> validNeighbors;
                for (int j = 0; j < 4; j++) {
                    cv::Point checkPoint = point + neighbors[j];
                    if (checkPoint.x >= 0 && checkPoint.x < roi.cols &&
                        checkPoint.y >= 0 && checkPoint.y < roi.rows &&
                        result.validMask.at<uchar>(checkPoint) > 0) {
                        validNeighbors.push_back(checkPoint);
                    }
                }
                
                bool success = false;
                
                if (!validNeighbors.empty()) {
                    // Find neighbor with best correlation
                    cv::Point bestNeighbor = validNeighbors[0];
                    double bestCC = result.cc.at<double>(bestNeighbor);
                    
                    for (size_t j = 1; j < validNeighbors.size(); j++) {
                        double cc = result.cc.at<double>(validNeighbors[j]);
                        if (cc < bestCC) {
                            bestCC = cc;
                            bestNeighbor = validNeighbors[j];
                        }
                    }
                    
                    // Use warp parameters from best neighbor as initial guess
                    localWarpParams.at<double>(0) = result.u.at<double>(bestNeighbor);
                    localWarpParams.at<double>(1) = result.v.at<double>(bestNeighbor);
                    
                    // Run GPU optimization
                    success = localOptimizer.optimizeGPU(point, localWarpParams, localZncc);
                    success = success && (localZncc < m_ccThreshold);
                    
                    if (success) {
                        // Check for displacement jump
                        double du = localWarpParams.at<double>(0) - result.u.at<double>(bestNeighbor);
                        double dv = localWarpParams.at<double>(1) - result.v.at<double>(bestNeighbor);
                        double dispJump = std::sqrt(du*du + dv*dv);
                        
                        success = (dispJump <= m_deltaDispThreshold);
                    }
                }
                
                // Store results
                resultFlags[i] = success;
                if (success) {
                    resultWarpParams[i] = localWarpParams.clone();
                    resultZncc[i] = localZncc;
                }
                
                // Increment processed count
                processedCount++;
            }
            
            // Signal that this thread is done with its batch
            readyCount++;
            
            // Last thread to finish clears the batch ready flag
            if (readyCount == m_numThreads) {
                batchReady = false;
                lock.unlock();
                batchCondition.notify_one(); // Notify main thread
            }
        }
    }));
    }

    // Main processing loop
    while (!queue.empty()) {
    // Prepare batch of points from the queue
    pointBatch.clear();

    // Critical section: extract points from the queue
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        
        for (int i = 0; i < batchSize && !queue.empty(); ++i) {
            auto current = queue.top();
            queue.pop();
            cv::Point currentPoint = current.first;
            
            // For each point in batch, collect its unanalyzed neighbors
            for (int j = 0; j < 4; j++) {
                cv::Point neighborPoint = currentPoint + neighbors[j];
                
                // Check if within bounds and not yet analyzed
                if (neighborPoint.x >= 0 && neighborPoint.x < roi.cols &&
                    neighborPoint.y >= 0 && neighborPoint.y < roi.rows &&
                    roi.at<uchar>(neighborPoint) > 0 &&
                    analyzedPoints.at<uchar>(neighborPoint) == 0) {
                    
                    // Mark as analyzed to avoid duplicates
                    analyzedPoints.at<uchar>(neighborPoint) = 1;
                    
                    // Add to the batch
                    pointBatch.push_back(neighborPoint);
                }
            }
        }
    }

    // If no points to process, we're done
    if (pointBatch.empty()) {
        break;
    }

    // Resize result vectors to match batch size
    resultFlags.resize(pointBatch.size(), false);
    resultWarpParams.resize(pointBatch.size());
    resultZncc.resize(pointBatch.size(), 0.0);

    // Reset counters
    readyCount = 0;
    processedCount = 0;

    // Set batch ready flag and notify worker threads
    {
        std::unique_lock<std::mutex> lock(batchMutex);
        batchReady = true;
        lock.unlock();
        batchCondition.notify_all();
        
        // Wait for all threads to finish processing
        lock.lock();
        batchCondition.wait(lock, [&]() { return !batchReady; });
    }

    // Process results and update global data structures
    for (size_t i = 0; i < pointBatch.size(); ++i) {
        if (resultFlags[i]) {
            const cv::Point& point = pointBatch[i];
            
            // Update results
            {
                std::lock_guard<std::mutex> lock(m_resultMutex);
                result.u.at<double>(point) = resultWarpParams[i].at<double>(0);
                result.v.at<double>(point) = resultWarpParams[i].at<double>(1);
                result.cc.at<double>(point) = resultZncc[i];
                result.validMask.at<uchar>(point) = 1;
            }
            
            // Add to queue for further propagation
            {
                std::lock_guard<std::mutex> lock(m_queueMutex);
                queue.push(std::make_pair(point, resultZncc[i]));
            }
        }
    }
    }

    // Stop worker threads
    {
    std::lock_guard<std::mutex> lock(batchMutex);
    stopWorkers = true;
    }
    batchCondition.notify_all();

    // Join worker threads
    for (auto& thread : workerThreads) {
    thread.join();
    }
    }

    cv::Mat RGDIC::calculateSDA(const cv::Mat& roi) {
    cv::Mat dist;

    // Use GPU for distance transform if available
    if (m_mode == GPU_CUDA || m_mode == HYBRID) {
    // For GPU mode, we'll use CUDA kernels directly
    // This is a simplified implementation - in practice, you'd implement a CUDA kernel for the distance transform

    // Fall back to CPU implementation for now
    cv::distanceTransform(roi, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    } else {
    // For multi-threaded CPU mode, use parallel processing if available
    if (m_mode == CPU_MULTI_THREAD) {
        // Unfortunately, distanceTransform doesn't support parallel execution directly,
        // so we'll use the standard implementation
        cv::distanceTransform(roi, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    } else {
        // Single-threaded CPU mode
        cv::distanceTransform(roi, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    }
    }

    return dist;
    }

    cv::Point RGDIC::findSeedPoint(const cv::Mat& roi, const cv::Mat& sda) {
    cv::Point maxLoc;
    double maxVal;

    // Find point with maximum SDA value (furthest from boundaries)
    cv::minMaxLoc(sda, nullptr, &maxVal, nullptr, &maxLoc, roi);

    return maxLoc;
    }

    bool RGDIC::analyzePoint(const cv::Point& point, ICGNOptimizer& optimizer, 
                const cv::Mat& roi, DisplacementResult& result, 
                PriorityQueue& queue, cv::Mat& analyzedPoints)
    {
    // Get neighboring points that have already been analyzed successfully
    std::vector<cv::Point> validNeighbors;

    const cv::Point neighbors[] = {
    {0, 1}, {1, 0}, {0, -1}, {-1, 0}
    };

    // Thread-safe access to valid neighbors
    {
    std::lock_guard<std::mutex> lock(m_resultMutex);
    for (int i = 0; i < 4; i++) {
        cv::Point neighborPoint = point + neighbors[i];
        
        if (neighborPoint.x >= 0 && neighborPoint.x < roi.cols &&
            neighborPoint.y >= 0 && neighborPoint.y < roi.rows &&
            result.validMask.at<uchar>(neighborPoint) > 0) {
            
            validNeighbors.push_back(neighborPoint);
        }
    }
    }

    if (validNeighbors.empty()) {
    return false; // No valid neighbors to use as initial guess
    }

    // Find neighbor with best correlation coefficient
    cv::Point bestNeighbor = validNeighbors[0];
    double bestCC = 0.0;

    {
    std::lock_guard<std::mutex> lock(m_resultMutex);
    bestCC = result.cc.at<double>(bestNeighbor);

    for (size_t i = 1; i < validNeighbors.size(); i++) {
        double cc = result.cc.at<double>(validNeighbors[i]);
        if (cc < bestCC) { // Lower ZNCC value = better correlation
            bestCC = cc;
            bestNeighbor = validNeighbors[i];
        }
    }
    }

    // Use warp parameters from best neighbor as initial guess
    cv::Mat warpParams = cv::Mat::zeros(m_numParams, 1, CV_64F);

    {
    std::lock_guard<std::mutex> lock(m_resultMutex);
    warpParams.at<double>(0) = result.u.at<double>(bestNeighbor);
    warpParams.at<double>(1) = result.v.at<double>(bestNeighbor);
    }

    // Run ICGN optimization
    double zncc;
    bool success = false;

    if (m_mode == GPU_CUDA || m_mode == HYBRID) {
    success = optimizer.optimizeGPU(point, warpParams, zncc);
    } else {
    success = optimizer.optimize(point, warpParams, zncc);
    }

    if (success && zncc < m_ccThreshold) { // Lower ZNCC value = better correlation
    // Check for displacement jump
    double du = 0.0;
    double dv = 0.0;

    {
        std::lock_guard<std::mutex> lock(m_resultMutex);
        du = warpParams.at<double>(0) - result.u.at<double>(bestNeighbor);
        dv = warpParams.at<double>(1) - result.v.at<double>(bestNeighbor);
    }

    double dispJump = std::sqrt(du*du + dv*dv);

    if (dispJump <= m_deltaDispThreshold) {
        // Store results
        {
            std::lock_guard<std::mutex> lock(m_resultMutex);
            result.u.at<double>(point) = warpParams.at<double>(0);
            result.v.at<double>(point) = warpParams.at<double>(1);
            result.cc.at<double>(point) = zncc;
            result.validMask.at<uchar>(point) = 1;
        }
        
        // Add to queue for further propagation
        queue.push(std::make_pair(point, zncc));
        return true;
    }
    }

    return false;
    }

    // Divide ROI into segments for parallel processing
    std::vector<RGDIC::PointSegment> RGDIC::divideROIIntoSegments(const cv::Mat& roi, int numSegments) {
    // Count total ROI points
    int totalPoints = cv::countNonZero(roi);
    int pointsPerSegment = totalPoints / numSegments;

    std::vector<PointSegment> segments;

    // Collect all ROI points
    std::vector<cv::Point> allPoints;
    for (int y = 0; y < roi.rows; y++) {
    for (int x = 0; x < roi.cols; x++) {
        if (roi.at<uchar>(y, x) > 0) {
            allPoints.push_back(cv::Point(x, y));
        }
    }
    }

    // Divide points into segments
    for (int i = 0; i < numSegments; i++) {
    int startIdx = i * pointsPerSegment;
    int endIdx = (i == numSegments - 1) ? allPoints.size() : (i + 1) * pointsPerSegment;

    if (startIdx >= allPoints.size()) {
        break;
    }

    PointSegment segment;
    segment.points.assign(allPoints.begin() + startIdx, allPoints.begin() + endIdx);

    // Calculate bounding rectangle
    int minX = INT_MAX, minY = INT_MAX;
    int maxX = 0, maxY = 0;

    for (const auto& pt : segment.points) {
        minX = std::min(minX, pt.x);
        minY = std::min(minY, pt.y);
        maxX = std::max(maxX, pt.x);
        maxY = std::max(maxY, pt.y);
    }

    segment.bounds = cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
    segments.push_back(segment);
    }

    return segments;
    }

    void RGDIC::displayResults(const cv::Mat& refImage, const DisplacementResult& result, 
                    const cv::Mat& trueDispX, const cv::Mat& trueDispY) {
    // Create visualizations for displacement fields
    cv::Mat uViz, vViz;

    // Find min/max values for normalization
    double minU, maxU, minV, maxV;
    cv::minMaxLoc(result.u, &minU, &maxU, nullptr, nullptr, result.validMask);
    cv::minMaxLoc(result.v, &minV, &maxV, nullptr, nullptr, result.validMask);

    std::cout << "Displacement range: U [" << minU << ", " << maxU << "] V [" << minV << ", " << maxV << "]" << std::endl;

    // Normalize displacement fields for visualization
    cv::Mat uNorm, vNorm;
    cv::normalize(result.u, uNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
    cv::normalize(result.v, vNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);

    // Apply color map
    cv::Mat uColor, vColor;
    cv::applyColorMap(uNorm, uColor, cv::COLORMAP_JET);
    cv::applyColorMap(vNorm, vColor, cv::COLORMAP_JET);

    // Apply valid mask
    cv::Mat validMask3Ch;
    cv::cvtColor(result.validMask, validMask3Ch, cv::COLOR_GRAY2BGR);
    uColor = uColor.mul(validMask3Ch, 1.0/255.0);
    vColor = vColor.mul(validMask3Ch, 1.0/255.0);

    // Create displacement field visualization
    cv::Mat dispField;
    cv::cvtColor(refImage, dispField, cv::COLOR_GRAY2BGR);

    // Draw displacement vectors (subsampled)
    int step = 10;
    for (int y = 0; y < result.u.rows; y += step) {
    for (int x = 0; x < result.u.cols; x += step) {
        if (result.validMask.at<uchar>(y, x)) {
            double u = result.u.at<double>(y, x);
            double v = result.v.at<double>(y, x);
            
            // Scale displacements for visibility
            double scale = 5.0;
            cv::arrowedLine(dispField, cv::Point(x, y), 
                        cv::Point(x + u * scale, y + v * scale),
                        cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }
    }
    }

    // Display results
    cv::imshow("U Displacement", uColor);
    cv::imshow("V Displacement", vColor);
    cv::imshow("Displacement Field", dispField);
    cv::imshow("Valid Points", result.validMask * 255);

    // Save results
    cv::imwrite("u_displacement.png", uColor);
    cv::imwrite("v_displacement.png", vColor);
    cv::imwrite("displacement_field.png", dispField);
    cv::imwrite("valid_points.png", result.validMask * 255);

    // If ground truth is available, compute error maps
    if (!trueDispX.empty() && !trueDispY.empty()) {
    evaluateErrors(result, trueDispX, trueDispY);
    }

    cv::waitKey(0);
    }

    void RGDIC::evaluateErrors(const DisplacementResult& result, 
                    const cv::Mat& trueDispX, const cv::Mat& trueDispY) {
    // Convert data types if necessary - ensure all matrices are of the same type
    cv::Mat u, v, trueU, trueV;
    result.u.convertTo(u, CV_32F);
    result.v.convertTo(v, CV_32F);
    trueDispX.convertTo(trueU, CV_32F);
    trueDispY.convertTo(trueV, CV_32F);

    // Calculate error maps
    cv::Mat errorU, errorV;
    cv::subtract(u, trueU, errorU);
    cv::subtract(v, trueV, errorV);

    // Calculate absolute errors 
    cv::Mat absErrorU, absErrorV;
    cv::absdiff(u, trueU, absErrorU);
    cv::absdiff(v, trueV, absErrorV);

    // Convert valid mask to proper type for arithmetic operations
    cv::Mat validMaskFloat;
    result.validMask.convertTo(validMaskFloat, CV_32F, 1.0/255.0);

    // Compute statistics for valid points
    cv::Scalar meanErrorU = cv::mean(absErrorU, result.validMask);
    cv::Scalar meanErrorV = cv::mean(absErrorV, result.validMask);

    double meanU = meanErrorU[0];
    double meanV = meanErrorV[0];

    // Find max errors
    double minU, maxU, minV, maxV;
    cv::minMaxLoc(absErrorU, &minU, &maxU, nullptr, nullptr, result.validMask);
    cv::minMaxLoc(absErrorV, &minV, &maxV, nullptr, nullptr, result.validMask);

    // Calculate RMS error
    cv::Mat errorUSq, errorVSq;
    cv::multiply(errorU, errorU, errorUSq);
    cv::multiply(errorV, errorV, errorVSq);

    // Make sure to convert mask to float for multiplication
    cv::Mat errorUSqMasked, errorVSqMasked;
    cv::multiply(errorUSq, validMaskFloat, errorUSqMasked);
    cv::multiply(errorVSq, validMaskFloat, errorVSqMasked);

    cv::Scalar sumUSq = cv::sum(errorUSqMasked);
    cv::Scalar sumVSq = cv::sum(errorVSqMasked);

    int validPoints = cv::countNonZero(result.validMask);
    double rmsU = std::sqrt(sumUSq[0] / validPoints);
    double rmsV = std::sqrt(sumVSq[0] / validPoints);

    // Print error metrics
    std::cout << "Error Metrics:" << std::endl;
    std::cout << "  U displacement: mean = " << meanU << " px, max = " << maxU << " px, RMS = " << rmsU << " px" << std::endl;
    std::cout << "  V displacement: mean = " << meanV << " px, max = " << maxV << " px, RMS = " << rmsV << " px" << std::endl;

    // Visualize error maps
    cv::Mat errorUNorm, errorVNorm;
    cv::normalize(absErrorU, errorUNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
    cv::normalize(absErrorV, errorVNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);

    cv::Mat errorUColor, errorVColor;
    cv::applyColorMap(errorUNorm, errorUColor, cv::COLORMAP_JET);
    cv::applyColorMap(errorVNorm, errorVColor, cv::COLORMAP_JET);

    // Apply valid mask - convert to 3 channels first
    cv::Mat validMask3Ch;
    cv::cvtColor(result.validMask, validMask3Ch, cv::COLOR_GRAY2BGR);

    // Use bitwise AND instead of multiplication for masks
    cv::Mat errorUColorMasked, errorVColorMasked;
    cv::bitwise_and(errorUColor, validMask3Ch, errorUColorMasked);
    cv::bitwise_and(errorVColor, validMask3Ch, errorVColorMasked);

    // Display and save error maps
    cv::imshow("U Error Map", errorUColorMasked);
    cv::imshow("V Error Map", errorVColorMasked);

    cv::imwrite("u_error_map.png", errorUColorMasked);
    cv::imwrite("v_error_map.png", errorVColorMasked);
}

// Add this to rgdic.cpp:
cv::Mat RGDIC::visualizeDisplacement(const cv::Mat& u, const cv::Mat& v, const cv::Mat& mask) const {
    cv::Mat validMask;
    if (mask.empty()) {
        validMask = cv::Mat(u.size(), CV_8U, cv::Scalar(255));
    } else {
        validMask = mask;
    }
    
    // Create visualizations for displacement fields
    cv::Mat uColor, vColor, magnitude;
    
    // Find min/max values for normalization
    double minU, maxU, minV, maxV;
    cv::minMaxLoc(u, &minU, &maxU, nullptr, nullptr, validMask);
    cv::minMaxLoc(v, &minV, &maxV, nullptr, nullptr, validMask);
    
    // Compute magnitude
    cv::Mat uSquared, vSquared, magSquared;
    cv::multiply(u, u, uSquared);
    cv::multiply(v, v, vSquared);
    cv::add(uSquared, vSquared, magSquared);
    cv::sqrt(magSquared, magnitude);
    
    double minMag, maxMag;
    cv::minMaxLoc(magnitude, &minMag, &maxMag, nullptr, nullptr, validMask);
    
    // Normalize displacement fields for visualization
    cv::Mat uNorm, vNorm, magNorm;
    cv::normalize(u, uNorm, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);
    cv::normalize(v, vNorm, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);
    cv::normalize(magnitude, magNorm, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);
    
    // Apply color map
    cv::applyColorMap(uNorm, uColor, cv::COLORMAP_JET);
    cv::applyColorMap(vNorm, vColor, cv::COLORMAP_JET);
    cv::Mat magColor;
    cv::applyColorMap(magNorm, magColor, cv::COLORMAP_JET);
    
    // Apply valid mask if provided
    if (!mask.empty()) {
        cv::Mat validMask3Ch;
        cv::cvtColor(validMask, validMask3Ch, cv::COLOR_GRAY2BGR);
        uColor = uColor.mul(validMask3Ch, 1.0/255.0);
        vColor = vColor.mul(validMask3Ch, 1.0/255.0);
        magColor = magColor.mul(validMask3Ch, 1.0/255.0);
    }
    
    // Create a displacement field visualization with arrows
    cv::Mat dispField = cv::Mat::zeros(u.size(), CV_8UC3);
    cv::cvtColor(magNorm, dispField, cv::COLOR_GRAY2BGR);
    
    // Draw displacement vectors (subsampled)
    int step = std::max(u.rows, u.cols) / 50; // Adjust for visibility
    step = std::max(step, 10); // Minimum step size
    
    for (int y = 0; y < u.rows; y += step) {
        for (int x = 0; x < u.cols; x += step) {
            if (mask.empty() || mask.at<uchar>(y, x)) {
                double dx = u.at<double>(y, x);
                double dy = v.at<double>(y, x);
                
                // Scale displacements for visibility
                double scale = 2.0;
                cv::arrowedLine(dispField, cv::Point(x, y), 
                              cv::Point(x + dx * scale, y + dy * scale),
                              cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            }
        }
    }
    
    // Add text with min/max values
    std::stringstream ssU, ssV, ssMag;
    ssU << "U: " << std::fixed << std::setprecision(2) << minU << " to " << maxU;
    ssV << "V: " << std::fixed << std::setprecision(2) << minV << " to " << maxV;
    ssMag << "Mag: " << std::fixed << std::setprecision(2) << minMag << " to " << maxMag;
    
    cv::putText(uColor, ssU.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    cv::putText(vColor, ssV.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    cv::putText(magColor, ssMag.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    cv::putText(dispField, ssMag.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    
    // Create a combined visualization
    cv::Mat topRow, bottomRow, fullVis;
    cv::hconcat(uColor, vColor, topRow);
    cv::hconcat(magColor, dispField, bottomRow);
    cv::vconcat(topRow, bottomRow, fullVis);
    
    return fullVis;
}
