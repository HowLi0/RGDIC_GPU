#include "rgdic.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>

// Constructor with GPU memory initialization
RGDIC::ICGNOptimizer::ICGNOptimizer(const cv::Mat& refImage, const cv::Mat& defImage,
                                 int subsetRadius, ShapeFunctionOrder order,
                                 double convergenceThreshold, int maxIterations,
                                 ComputationMode mode)
    : m_refImage(refImage),
      m_defImage(defImage),
      m_subsetRadius(subsetRadius),
      m_order(order),
      m_convergenceThreshold(convergenceThreshold),
      m_maxIterations(maxIterations),
      m_mode(mode)
{
    // Set number of parameters based on shape function order
    m_numParams = (order == FIRST_ORDER) ? 6 : 12;
    
    // Initialize GPU memory for GPU modes
    if (mode == GPU_CUDA || mode == HYBRID) {
        initGPUMemory();
    } else {
        // Set GPU memory pointers to null for CPU modes
        d_refImage = nullptr;
        d_defImage = nullptr;
        d_warpedSubset = nullptr;
        d_refSubset = nullptr;
        d_steepestDescentImages = nullptr;
        d_hessian = nullptr;
        d_errorVector = nullptr;
        d_zncc = nullptr;
    }
}

// Destructor to clean up GPU resources
RGDIC::ICGNOptimizer::~ICGNOptimizer() {
    if (m_mode == GPU_CUDA || m_mode == HYBRID) {
        freeGPUMemory();
    }
}

// Initialize GPU memory
void RGDIC::ICGNOptimizer::initGPUMemory() {
    int subsetDiameter = 2 * m_subsetRadius + 1;
    int subsetSize = subsetDiameter * subsetDiameter;
    
    // Allocate and copy reference image to device
    size_t refPitchBytes;
    RGDIC_CUDA::gpuErrchk(cudaMallocPitch(&d_refImage, &refPitchBytes, m_refImage.cols * sizeof(unsigned char), m_refImage.rows));
    m_refPitch = refPitchBytes / sizeof(unsigned char);
    RGDIC_CUDA::gpuErrchk(cudaMemcpy2D(d_refImage, refPitchBytes, m_refImage.data, m_refImage.step,
                               m_refImage.cols * sizeof(unsigned char), m_refImage.rows, cudaMemcpyHostToDevice));
    
    // Allocate and copy deformed image to device
    size_t defPitchBytes;
    RGDIC_CUDA::gpuErrchk(cudaMallocPitch(&d_defImage, &defPitchBytes, m_defImage.cols * sizeof(unsigned char), m_defImage.rows));
    m_defPitch = defPitchBytes / sizeof(unsigned char);
    RGDIC_CUDA::gpuErrchk(cudaMemcpy2D(d_defImage, defPitchBytes, m_defImage.data, m_defImage.step,
                               m_defImage.cols * sizeof(unsigned char), m_defImage.rows, cudaMemcpyHostToDevice));
    
    // Allocate memory for subsets and other data structures
    RGDIC_CUDA::gpuErrchk(cudaMalloc(&d_warpedSubset, subsetSize * sizeof(float)));
    RGDIC_CUDA::gpuErrchk(cudaMalloc(&d_refSubset, subsetSize * sizeof(float)));
    RGDIC_CUDA::gpuErrchk(cudaMalloc(&d_steepestDescentImages, m_numParams * subsetSize * sizeof(float)));
    RGDIC_CUDA::gpuErrchk(cudaMalloc(&d_hessian, m_numParams * m_numParams * sizeof(float)));
    RGDIC_CUDA::gpuErrchk(cudaMalloc(&d_errorVector, m_numParams * sizeof(float)));
    RGDIC_CUDA::gpuErrchk(cudaMalloc(&d_zncc, sizeof(float)));
}

// Free GPU memory
void RGDIC::ICGNOptimizer::freeGPUMemory() {
    if (d_refImage != nullptr) cudaFree(d_refImage);
    if (d_defImage != nullptr) cudaFree(d_defImage);
    if (d_warpedSubset != nullptr) cudaFree(d_warpedSubset);
    if (d_refSubset != nullptr) cudaFree(d_refSubset);
    if (d_steepestDescentImages != nullptr) cudaFree(d_steepestDescentImages);
    if (d_hessian != nullptr) cudaFree(d_hessian);
    if (d_errorVector != nullptr) cudaFree(d_errorVector);
    if (d_zncc != nullptr) cudaFree(d_zncc);
    
    d_refImage = nullptr;
    d_defImage = nullptr;
    d_warpedSubset = nullptr;
    d_refSubset = nullptr;
    d_steepestDescentImages = nullptr;
    d_hessian = nullptr;
    d_errorVector = nullptr;
    d_zncc = nullptr;
}

// GPU version of initialGuess
bool RGDIC::ICGNOptimizer::initialGuessGPU(const cv::Point& refPoint, cv::Mat& warpParams, double& zncc) const {
    // Simple grid search for initial translation parameters
    int searchRadius = m_subsetRadius; // Search radius
    double bestZNCC = std::numeric_limits<double>::max(); // Lower ZNCC = better
    cv::Point bestOffset(0, 0);
    
    // Initialize warp params if not already initialized
    if (warpParams.empty()) {
        warpParams = cv::Mat::zeros(m_numParams, 1, CV_64F);
    }
    
    // Create grid of search points
    const int gridStepSize = 2;  // Step size for grid search
    const int gridSize = (2 * searchRadius / gridStepSize + 1) * (2 * searchRadius / gridStepSize + 1);
    
    // Allocate device memory for search results
    float* d_znccResults = nullptr;
    float* d_warpParams = nullptr;
    float* d_points = nullptr;
    
    RGDIC_CUDA::gpuErrchk(cudaMalloc(&d_znccResults, gridSize * sizeof(float)));
    RGDIC_CUDA::gpuErrchk(cudaMalloc(&d_warpParams, m_numParams * gridSize * sizeof(float)));
    RGDIC_CUDA::gpuErrchk(cudaMalloc(&d_points, gridSize * 2 * sizeof(float)));
    
    // Initialize warp parameters for each grid point on host
    std::vector<float> h_warpParams(m_numParams * gridSize, 0.0f);
    std::vector<float> h_points(gridSize * 2);
    
    int idx = 0;
    for (int dy = -searchRadius; dy <= searchRadius; dy += gridStepSize) {
        for (int dx = -searchRadius; dx <= searchRadius; dx += gridStepSize) {
            // Skip if out of bounds
            cv::Point curPoint = refPoint + cv::Point(dx, dy);
            if (curPoint.x < m_subsetRadius || curPoint.x >= m_defImage.cols - m_subsetRadius ||
                curPoint.y < m_subsetRadius || curPoint.y >= m_defImage.rows - m_subsetRadius) {
                continue;
            }
            
            // Set translation parameters for this grid point
            h_warpParams[idx * m_numParams + 0] = static_cast<float>(dx);
            h_warpParams[idx * m_numParams + 1] = static_cast<float>(dy);
            
            // Set warped point coordinates
            h_points[idx * 2 + 0] = static_cast<float>(curPoint.x);
            h_points[idx * 2 + 1] = static_cast<float>(curPoint.y);
            
            idx++;
        }
    }
    
    int actualGridSize = idx;
    
    if (actualGridSize == 0) {
        // No valid points to search
        cudaFree(d_znccResults);
        cudaFree(d_warpParams);
        cudaFree(d_points);
        return false;
    }
    
    // Copy data to device
    RGDIC_CUDA::gpuErrchk(cudaMemcpy(d_warpParams, h_warpParams.data(), m_numParams * actualGridSize * sizeof(float), cudaMemcpyHostToDevice));
    RGDIC_CUDA::gpuErrchk(cudaMemcpy(d_points, h_points.data(), actualGridSize * 2 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch multiple parallel ZNCC computations
    int threadsPerBlock = 256;
    int blocks = (actualGridSize + threadsPerBlock - 1) / threadsPerBlock;
    
    // For each grid point, compute ZNCC
    for (int i = 0; i < actualGridSize; i++) {
        RGDIC_CUDA::computeZNCCKernel<<<1, threadsPerBlock>>>(
            d_refImage, m_refPitch,
            d_defImage, m_defPitch,
            m_subsetRadius, refPoint.x, refPoint.y,
            d_warpParams + i * m_numParams, m_numParams,
            d_znccResults + i
        );
    }
    
    // Wait for all kernels to finish
    cudaDeviceSynchronize();
    
    // Copy results back to host
    std::vector<float> h_znccResults(actualGridSize);
    RGDIC_CUDA::gpuErrchk(cudaMemcpy(h_znccResults.data(), d_znccResults, actualGridSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Find best ZNCC value
    float bestZNCCValue = std::numeric_limits<float>::max();
    int bestIdx = -1;
    
    for (int i = 0; i < actualGridSize; i++) {
        if (h_znccResults[i] < bestZNCCValue) {
            bestZNCCValue = h_znccResults[i];
            bestIdx = i;
        }
    }
    
    // Free device memory
    cudaFree(d_znccResults);
    cudaFree(d_warpParams);
    cudaFree(d_points);
    
    // Return the best parameters
    if (bestIdx >= 0) {
        warpParams.at<double>(0) = static_cast<double>(h_warpParams[bestIdx * m_numParams + 0]);
        warpParams.at<double>(1) = static_cast<double>(h_warpParams[bestIdx * m_numParams + 1]);
        zncc = static_cast<double>(bestZNCCValue);
        return true;
    }
    
    return false;
}

// GPU version of optimize
bool RGDIC::ICGNOptimizer::optimizeGPU(const cv::Point& refPoint, cv::Mat& warpParams, double& zncc) const {
    // Initial guess if not provided
    if (warpParams.at<double>(0) == 0 && warpParams.at<double>(1) == 0) {
        if (!initialGuessGPU(refPoint, warpParams, zncc)) {
            return false;
        }
    }
    
    // Subset dimensions
    int subsetDiameter = 2 * m_subsetRadius + 1;
    int subsetSize = subsetDiameter * subsetDiameter;
    
    // Prepare reference subset
    std::vector<float> h_refSubset(subsetSize);
    for (int y = -m_subsetRadius; y <= m_subsetRadius; y++) {
        for (int x = -m_subsetRadius; x <= m_subsetRadius; x++) {
            int refX = refPoint.x + x;
            int refY = refPoint.y + y;
            
            if (refX >= 0 && refX < m_refImage.cols && refY >= 0 && refY < m_refImage.rows) {
                h_refSubset[(y + m_subsetRadius) * subsetDiameter + (x + m_subsetRadius)] = 
                    static_cast<float>(m_refImage.at<uchar>(refY, refX));
            } else {
                h_refSubset[(y + m_subsetRadius) * subsetDiameter + (x + m_subsetRadius)] = 0.0f;
            }
        }
    }
    
    // Copy reference subset to device
    RGDIC_CUDA::gpuErrchk(cudaMemcpy(d_refSubset, h_refSubset.data(), subsetSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // Pre-compute steepest descent images on GPU
    computeSteepestDescentImagesGPU(refPoint);
    
    // Pre-compute Hessian matrix on GPU
    computeHessianGPU();
    
    // Copy Hessian matrix from GPU to CPU for inversion
    std::vector<float> h_hessian(m_numParams * m_numParams);
    RGDIC_CUDA::gpuErrchk(cudaMemcpy(h_hessian.data(), d_hessian, m_numParams * m_numParams * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Convert to double for OpenCV inversion
    cv::Mat hessian(m_numParams, m_numParams, CV_32F, h_hessian.data());
    cv::Mat hessianDouble;
    hessian.convertTo(hessianDouble, CV_64F);
    
    // Check if Hessian is invertible
    cv::Mat hessianInv;
    if (cv::invert(hessianDouble, hessianInv, cv::DECOMP_CHOLESKY) == 0) {
        return false; // Not invertible
    }
    
    // ICGN main loop
    double prevZNCC = std::numeric_limits<double>::max();
    int iter = 0;
    
    // Convert warpParams to float for CUDA
    std::vector<float> h_warpParams(m_numParams);
    for (int i = 0; i < m_numParams; i++) {
        h_warpParams[i] = static_cast<float>(warpParams.at<double>(i));
    }
    
    // Allocate device memory for warp params
    float* d_currentWarpParams = nullptr;
    RGDIC_CUDA::gpuErrchk(cudaMalloc(&d_currentWarpParams, m_numParams * sizeof(float)));
    
    while (iter < m_maxIterations) {
        // Copy current warp parameters to device
        RGDIC_CUDA::gpuErrchk(cudaMemcpy(d_currentWarpParams, h_warpParams.data(), m_numParams * sizeof(float), cudaMemcpyHostToDevice));
        
        // Compute ZNCC for current parameters
        RGDIC_CUDA::computeZNCCKernel<<<1, 256>>>(
            d_refImage, m_refPitch,
            d_defImage, m_defPitch,
            m_subsetRadius, refPoint.x, refPoint.y,
            d_currentWarpParams, m_numParams,
            d_zncc
        );
        
        // Get ZNCC value
        float currentZNCC;
        RGDIC_CUDA::gpuErrchk(cudaMemcpy(&currentZNCC, d_zncc, sizeof(float), cudaMemcpyDeviceToHost));
        
        // Update zncc
        zncc = static_cast<double>(currentZNCC);
        
        // Check for convergence
        if (std::abs(zncc - prevZNCC) < m_convergenceThreshold) {
            break;
        }
        
        prevZNCC = zncc;
        
        // Warp subset with current parameters
        dim3 blockSize(16, 16);
        dim3 gridSize((subsetDiameter + blockSize.x - 1) / blockSize.x,
                      (subsetDiameter + blockSize.y - 1) / blockSize.y);
        
        RGDIC_CUDA::warpSubsetKernel<<<gridSize, blockSize>>>(
            d_defImage, m_defPitch,
            m_defImage.cols, m_defImage.rows,
            refPoint.x, refPoint.y,
            m_subsetRadius,
            d_currentWarpParams, m_numParams,
            d_warpedSubset
        );
        
        // Compute error vector
        RGDIC_CUDA::errorVectorKernel<<<m_numParams, 256>>>(
            d_refSubset,
            d_warpedSubset,
            d_steepestDescentImages,
            subsetDiameter,
            m_numParams,
            d_errorVector
        );
        
        // Copy error vector from device to host
        std::vector<float> h_errorVector(m_numParams);
        RGDIC_CUDA::gpuErrchk(cudaMemcpy(h_errorVector.data(), d_errorVector, m_numParams * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Convert to double for matrix operations
        cv::Mat errorVector(m_numParams, 1, CV_32F, h_errorVector.data());
        cv::Mat errorVectorDouble;
        errorVector.convertTo(errorVectorDouble, CV_64F);
        
        // Calculate parameter update: Δp = H⁻¹ * error
        cv::Mat deltaP = hessianInv * errorVectorDouble;
        
        // Update parameters
        for (int p = 0; p < m_numParams; p++) {
            h_warpParams[p] += static_cast<float>(deltaP.at<double>(p, 0));
            warpParams.at<double>(p) = static_cast<double>(h_warpParams[p]);
        }
        
        // Check convergence based on parameter update norm
        double deltaNorm = cv::norm(deltaP);
        if (deltaNorm < m_convergenceThreshold) {
            break;
        }
        
        iter++;
    }
    
    // Free temporary device memory
    cudaFree(d_currentWarpParams);
    
    // Final ZNCC calculation
    float finalZNCC;
    RGDIC_CUDA::gpuErrchk(cudaMemcpy(d_currentWarpParams, h_warpParams.data(), m_numParams * sizeof(float), cudaMemcpyHostToDevice));
    
    RGDIC_CUDA::computeZNCCKernel<<<1, 256>>>(
        d_refImage, m_refPitch,
        d_defImage, m_defPitch,
        m_subsetRadius, refPoint.x, refPoint.y,
        d_currentWarpParams, m_numParams,
        d_zncc
    );
    
    RGDIC_CUDA::gpuErrchk(cudaMemcpy(&finalZNCC, d_zncc, sizeof(float), cudaMemcpyDeviceToHost));
    zncc = static_cast<double>(finalZNCC);
    
    cudaFree(d_currentWarpParams);
    
    return true;
}

// GPU version of steepest descent images computation
void RGDIC::ICGNOptimizer::computeSteepestDescentImagesGPU(const cv::Point& refPoint) const {
    int subsetDiameter = 2 * m_subsetRadius + 1;
    
    dim3 blockSize(16, 16);
    dim3 gridSize((subsetDiameter + blockSize.x - 1) / blockSize.x,
                  (subsetDiameter + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel to compute steepest descent images
    RGDIC_CUDA::steepestDescentKernel<<<gridSize, blockSize>>>(
        d_refImage, m_refPitch,
        m_refImage.cols, m_refImage.rows,
        refPoint.x, refPoint.y,
        m_subsetRadius, m_numParams,
        d_steepestDescentImages
    );
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
}

// GPU version of Hessian computation
void RGDIC::ICGNOptimizer::computeHessianGPU() const {
    // Launch kernel to compute Hessian
    dim3 blockSize(m_numParams, m_numParams);
    dim3 gridSize(1, 1);
    
    int subsetDiameter = 2 * m_subsetRadius + 1;
    
    // For large parameter sets, switch to a different grid layout
    if (m_numParams > 16) {  // Adjust this threshold based on device capabilities
        blockSize = dim3(1, 1);
        gridSize = dim3(m_numParams, m_numParams);
    }
    
    RGDIC_CUDA::computeHessianKernel<<<gridSize, 256>>>(
        d_steepestDescentImages,
        subsetDiameter,
        m_numParams,
        d_hessian
    );
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
}

// GPU version of ZNCC computation
double RGDIC::ICGNOptimizer::computeZNCCGPU(const cv::Point& refPoint, const cv::Mat& warpParams) const {
    // Convert warpParams to float for CUDA
    std::vector<float> h_warpParams(m_numParams);
    for (int i = 0; i < m_numParams; i++) {
        h_warpParams[i] = static_cast<float>(warpParams.at<double>(i));
    }
    
    // Copy parameters to device
    float* d_params = nullptr;
    RGDIC_CUDA::gpuErrchk(cudaMalloc(&d_params, m_numParams * sizeof(float)));
    RGDIC_CUDA::gpuErrchk(cudaMemcpy(d_params, h_warpParams.data(), m_numParams * sizeof(float), cudaMemcpyHostToDevice));
    
    // Compute ZNCC
    RGDIC_CUDA::computeZNCCKernel<<<1, 256>>>(
        d_refImage, m_refPitch,
        d_defImage, m_defPitch,
        m_subsetRadius, refPoint.x, refPoint.y,
        d_params, m_numParams,
        d_zncc
    );
    
    // Get result
    float zncc_value;
    RGDIC_CUDA::gpuErrchk(cudaMemcpy(&zncc_value, d_zncc, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free temporary device memory
    cudaFree(d_params);
    
    return static_cast<double>(zncc_value);
}