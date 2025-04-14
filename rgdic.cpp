#include "RGDIC.h"
#include <iostream>
#include <functional>

RGDIC::RGDIC(int subsetRadius, double convergenceThreshold, int maxIterations,
           double ccThreshold, double deltaDispThreshold, ShapeFunctionOrder order)
    : m_subsetRadius(subsetRadius),
      m_convergenceThreshold(convergenceThreshold),
      m_maxIterations(maxIterations),
      m_ccThreshold(ccThreshold),
      m_deltaDispThreshold(deltaDispThreshold),
      m_order(order)
{
    // Set number of parameters based on shape function order
    m_numParams = (order == FIRST_ORDER) ? 6 : 12;
}

RGDIC::DisplacementResult RGDIC::compute(const cv::Mat& refImage, const cv::Mat& defImage, const cv::Mat& roi)
{
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
    
    // Create the ICGN optimizer
    ICGNOptimizer optimizer(refImage, defImage, m_subsetRadius, m_order, 
                          m_convergenceThreshold, m_maxIterations);
    
    // Initialize warp parameters matrix
    cv::Mat warpParams = cv::Mat::zeros(m_numParams, 1, CV_64F);
    
    // Compute initial ZNCC
    double initialZNCC = 0.0;
    
    // Try to compute initial guess for seed point
    if (!optimizer.initialGuess(seedPoint, warpParams, initialZNCC)) {
        std::cerr << "Failed to find initial guess for seed point." << std::endl;
        return result;
    }
    
    // Optimize seed point
    if (!optimizer.optimize(seedPoint, warpParams, initialZNCC)) {
        std::cerr << "Failed to optimize seed point." << std::endl;
        return result;
    }
    
    // Check if seed point has good correlation
    if (initialZNCC > m_ccThreshold) {
        std::cerr << "Seed point has poor correlation: " << initialZNCC << std::endl;
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
                if (analyzePoint(neighborPoint, optimizer, roi, result, queue, analyzedPoints)) {
                    // Point successfully analyzed and added to queue
                }
            }
        }
    }
    
    // Post-process to remove outliers
    // Create a copy of valid mask
    cv::Mat validMaskCopy = result.validMask.clone();
    
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
    
    // Update result with filtered mask
    result.validMask = validMaskCopy;
    
    return result;
}

cv::Mat RGDIC::calculateSDA(const cv::Mat& roi) {
    cv::Mat dist;
    cv::distanceTransform(roi, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
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
    
    for (int i = 0; i < 4; i++) {
        cv::Point neighborPoint = point + neighbors[i];
        
        if (neighborPoint.x >= 0 && neighborPoint.x < roi.cols &&
            neighborPoint.y >= 0 && neighborPoint.y < roi.rows &&
            result.validMask.at<uchar>(neighborPoint) > 0) {
            
            validNeighbors.push_back(neighborPoint);
        }
    }
    
    if (validNeighbors.empty()) {
        return false; // No valid neighbors to use as initial guess
    }
    
    // Find neighbor with best correlation coefficient
    cv::Point bestNeighbor = validNeighbors[0];
    double bestCC = result.cc.at<double>(bestNeighbor);
    
    for (size_t i = 1; i < validNeighbors.size(); i++) {
        double cc = result.cc.at<double>(validNeighbors[i]);
        if (cc < bestCC) { // Lower ZNCC value = better correlation
            bestCC = cc;
            bestNeighbor = validNeighbors[i];
        }
    }
    
    // Use warp parameters from best neighbor as initial guess
    cv::Mat warpParams = cv::Mat::zeros(m_numParams, 1, CV_64F);
    warpParams.at<double>(0) = result.u.at<double>(bestNeighbor);
    warpParams.at<double>(1) = result.v.at<double>(bestNeighbor);
    
    // For higher order parameters, we'd need to store them in the result
    // For simplicity, we're only storing and using u and v here
    
    // Run ICGN optimization
    double zncc;
    bool success = optimizer.optimize(point, warpParams, zncc);
    
    if (success && zncc < m_ccThreshold) { // Lower ZNCC value = better correlation
        // Check for displacement jump
        double du = warpParams.at<double>(0) - result.u.at<double>(bestNeighbor);
        double dv = warpParams.at<double>(1) - result.v.at<double>(bestNeighbor);
        double dispJump = std::sqrt(du*du + dv*dv);
        
        if (dispJump <= m_deltaDispThreshold) {
            // Store results
            result.u.at<double>(point) = warpParams.at<double>(0);
            result.v.at<double>(point) = warpParams.at<double>(1);
            result.cc.at<double>(point) = zncc;
            result.validMask.at<uchar>(point) = 1;
            
            // Add to queue for further propagation
            queue.push(std::make_pair(point, zncc));
            return true;
        }
    }
    
    return false;
}

void RGDIC::displayResults(const cv::Mat& refImage, const DisplacementResult& result, 
                         const cv::Mat& trueDispX, const cv::Mat& trueDispY) {
    // Create visualizations for displacement fields
    cv::Mat uViz, vViz;
    
    // Find min/max values for normalization
    double minU, maxU, minV, maxV;
    cv::minMaxLoc(result.u, &minU, &maxU, nullptr, nullptr, result.validMask);
    cv::minMaxLoc(result.v, &minV, &maxV, nullptr, nullptr, result.validMask);
    
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
    
    // Save results
    cv::imwrite("E:/code_C++/RGDIC/u_displacement.png", uColor);
    cv::imwrite("E:/code_C++/RGDIC/v_displacement.png", vColor);
    cv::imwrite("E:/code_C++/RGDIC/displacement_field.png", dispField);
    
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

    // Calculate absolute errors - use absdiff to avoid type issues
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
}

//--------------------------------------------------------------------
// ICGNOptimizer Implementation
//--------------------------------------------------------------------

RGDIC::ICGNOptimizer::ICGNOptimizer(const cv::Mat& refImage, const cv::Mat& defImage,
                                 int subsetRadius, ShapeFunctionOrder order,
                                 double convergenceThreshold, int maxIterations)
    : m_refImage(refImage),
      m_defImage(defImage),
      m_subsetRadius(subsetRadius),
      m_order(order),
      m_convergenceThreshold(convergenceThreshold),
      m_maxIterations(maxIterations)
{
    // Set number of parameters based on shape function order
    m_numParams = (order == FIRST_ORDER) ? 6 : 12;
}

bool RGDIC::ICGNOptimizer::initialGuess(const cv::Point& refPoint, cv::Mat& warpParams, double& zncc)const {
    // Simple grid search for initial translation parameters
    int searchRadius = m_subsetRadius; // Search radius
    double bestZNCC = std::numeric_limits<double>::max(); // Lower ZNCC = better
    cv::Point bestOffset(0, 0);
    
    // Initialize warp params if not already initialized
    if (warpParams.empty()) {
        warpParams = cv::Mat::zeros(m_numParams, 1, CV_64F);
    }
    
    // Perform grid search
    for (int dy = -searchRadius; dy <= searchRadius; dy += 2) {
        for (int dx = -searchRadius; dx <= searchRadius; dx += 2) {
            cv::Point curPoint = refPoint + cv::Point(dx, dy);
            
            // Check if within bounds
            if (curPoint.x >= m_subsetRadius && curPoint.x < m_defImage.cols - m_subsetRadius &&
                curPoint.y >= m_subsetRadius && curPoint.y < m_defImage.rows - m_subsetRadius) {
                
                // Create simple translation warp
                cv::Mat testParams = cv::Mat::zeros(m_numParams, 1, CV_64F);
                testParams.at<double>(0) = dx;
                testParams.at<double>(1) = dy;
                
                // Compute ZNCC
                double testZNCC = computeZNCC(refPoint, testParams);
                
                // Update best match
                if (testZNCC < bestZNCC) {
                    bestZNCC = testZNCC;
                    bestOffset = cv::Point(dx, dy);
                }
            }
        }
    }
    
    // If no good match found
    if (bestZNCC == std::numeric_limits<double>::max()) {
        return false;
    }
    
    // Set initial translation parameters
    warpParams.at<double>(0) = bestOffset.x;
    warpParams.at<double>(1) = bestOffset.y;
    zncc = bestZNCC;
    
    return true;
}

bool RGDIC::ICGNOptimizer::optimize(const cv::Point& refPoint, cv::Mat& warpParams, double& zncc) const {
    // Initial guess if not provided
    if (warpParams.at<double>(0) == 0 && warpParams.at<double>(1) == 0) {
        if (!initialGuess(refPoint, warpParams, zncc)) {
            return false;
        }
    }
    
    // Pre-compute steepest descent images (constant for IC algorithm)
    std::vector<cv::Mat> steepestDescentImages;
    computeSteepestDescentImages(refPoint, steepestDescentImages);
    
    // Pre-compute Hessian matrix (constant for IC algorithm)
    cv::Mat hessian;
    computeHessian(steepestDescentImages, hessian);
    
    // Check if Hessian is invertible
    cv::Mat hessianInv;
    if (cv::invert(hessian, hessianInv, cv::DECOMP_CHOLESKY) == 0) {
        return false; // Not invertible
    }
    
    // ICGN main loop
    double prevZNCC = std::numeric_limits<double>::max();
    int iter = 0;
    
    while (iter < m_maxIterations) {
        // Compute current ZNCC
        zncc = computeZNCC(refPoint, warpParams);
        
        // Check for convergence
        if (std::abs(zncc - prevZNCC) < m_convergenceThreshold) {
            break;
        }
        
        prevZNCC = zncc;
        
        // Prepare to calculate error vector
        cv::Mat errorVector = cv::Mat::zeros(m_numParams, 1, CV_64F);
        
        // For each pixel in subset
        for (int y = -m_subsetRadius; y <= m_subsetRadius; y++) {
            for (int x = -m_subsetRadius; x <= m_subsetRadius; x++) {
                // Reference subset coordinates
                cv::Point2f refSubsetPt(x, y);
                cv::Point refPixel = refPoint + cv::Point(x, y);
                
                // Check if within reference image bounds
                if (refPixel.x < 0 || refPixel.x >= m_refImage.cols ||
                    refPixel.y < 0 || refPixel.y >= m_refImage.rows) {
                    continue;
                }
                
                // Get reference intensity
                double refIntensity = m_refImage.at<uchar>(refPixel);
                
                // Warp point to get corresponding point in deformed image
                cv::Point2f defPt = warpPoint(refSubsetPt, warpParams);
                cv::Point2f defImgPt(refPoint.x + defPt.x, refPoint.y + defPt.y);
                
                // Check if within deformed image bounds
                if (defImgPt.x < 0 || defImgPt.x >= m_defImage.cols - 1 ||
                    defImgPt.y < 0 || defImgPt.y >= m_defImage.rows - 1) {
                    continue;
                }
                
                // Get deformed intensity (interpolated)
                double defIntensity = interpolate(m_defImage, defImgPt);
                
                // Calculate intensity error
                double error = refIntensity - defIntensity;
                
                // Update error vector
                for (int p = 0; p < m_numParams; p++) {
                    errorVector.at<double>(p) += error * steepestDescentImages[p].at<double>(y + m_subsetRadius, x + m_subsetRadius);
                }
            }
        }
        
        // Calculate parameter update: Δp = H⁻¹ * error
        cv::Mat deltaP = hessianInv * errorVector;
        
        // Update parameters (inverse compositional update)
        // For translation parameters, simple addition works
        warpParams.at<double>(0) += deltaP.at<double>(0);
        warpParams.at<double>(1) += deltaP.at<double>(1);
        
        // For deformation parameters, proper update uses the chain rule
        // This is a simplification - full IC update is more complex
        for (int p = 2; p < m_numParams; p++) {
            warpParams.at<double>(p) += deltaP.at<double>(p);
        }
        
        // Check convergence based on parameter update norm
        double deltaNorm = cv::norm(deltaP);
        if (deltaNorm < m_convergenceThreshold) {
            break;
        }
        
        iter++;
    }
    
    // Final ZNCC calculation
    zncc = computeZNCC(refPoint, warpParams);
    
    return true;
}

double RGDIC::ICGNOptimizer::computeZNCC(const cv::Point& refPoint, const cv::Mat& warpParams)const {
    double sumRef = 0, sumDef = 0;
    double sumRefSq = 0, sumDefSq = 0;
    double sumRefDef = 0;
    int count = 0;
    
    // For each pixel in subset
    for (int y = -m_subsetRadius; y <= m_subsetRadius; y++) {
        for (int x = -m_subsetRadius; x <= m_subsetRadius; x++) {
            // Reference subset coordinates
            cv::Point2f refSubsetPt(x, y);
            cv::Point refPixel = refPoint + cv::Point(x, y);
            
            // Check if within reference image bounds
            if (refPixel.x < 0 || refPixel.x >= m_refImage.cols ||
                refPixel.y < 0 || refPixel.y >= m_refImage.rows) {
                continue;
            }
            
            // Get reference intensity
            double refIntensity = m_refImage.at<uchar>(refPixel);
            
            // Warp point to get corresponding point in deformed image
            cv::Point2f defPt = warpPoint(refSubsetPt, warpParams);
            cv::Point2f defImgPt(refPoint.x + defPt.x, refPoint.y + defPt.y);
            
            // Check if within deformed image bounds
            if (defImgPt.x < 0 || defImgPt.x >= m_defImage.cols - 1 ||
                defImgPt.y < 0 || defImgPt.y >= m_defImage.rows - 1) {
                continue;
            }
            
            // Get deformed intensity (interpolated)
            double defIntensity = interpolate(m_defImage, defImgPt);
            
            // Update sums for ZNCC
            sumRef += refIntensity;
            sumDef += defIntensity;
            sumRefSq += refIntensity * refIntensity;
            sumDefSq += defIntensity * defIntensity;
            sumRefDef += refIntensity * defIntensity;
            count++;
        }
    }
    
    // Calculate ZNCC if we have enough points
    if (count > 0) {
        double meanRef = sumRef / count;
        double meanDef = sumDef / count;
        double varRef = sumRefSq / count - meanRef * meanRef;
        double varDef = sumDefSq / count - meanDef * meanDef;
        double covar = sumRefDef / count - meanRef * meanDef;
        
        if (varRef > 0 && varDef > 0) {
            // Return 1 - ZNCC to convert to minimization problem (0 is perfect match)
            return 1.0 - (covar / std::sqrt(varRef * varDef));
        }
    }
    
    return std::numeric_limits<double>::max(); // Error case
}

void RGDIC::ICGNOptimizer::computeSteepestDescentImages(const cv::Point& refPoint, 
                                                     std::vector<cv::Mat>& steepestDescentImages)const {
    // Calculate image gradients
    cv::Mat gradX, gradY;
    cv::Sobel(m_refImage, gradX, CV_64F, 1, 0, 3);
    cv::Sobel(m_refImage, gradY, CV_64F, 0, 1, 3);
    
    // Initialize steepest descent images
    steepestDescentImages.clear();
    for (int i = 0; i < m_numParams; i++) {
        steepestDescentImages.push_back(cv::Mat::zeros(2 * m_subsetRadius + 1, 2 * m_subsetRadius + 1, CV_64F));
    }
    
    // For each pixel in subset
    for (int y = -m_subsetRadius; y <= m_subsetRadius; y++) {
        for (int x = -m_subsetRadius; x <= m_subsetRadius; x++) {
            cv::Point pixel = refPoint + cv::Point(x, y);
            
            // Check if within image bounds
            if (pixel.x < 0 || pixel.x >= m_refImage.cols ||
                pixel.y < 0 || pixel.y >= m_refImage.rows) {
                continue;
            }
            
            // Get gradients at this pixel
            double dx = gradX.at<double>(pixel);
            double dy = gradY.at<double>(pixel);
            
            // Compute Jacobian matrix
            cv::Mat jacobian;
            computeWarpJacobian(cv::Point2f(x, y), jacobian);
            
            // Compute steepest descent images
            int row = y + m_subsetRadius;
            int col = x + m_subsetRadius;
            
            // First order parameters
            steepestDescentImages[0].at<double>(row, col) = dx; // du
            steepestDescentImages[1].at<double>(row, col) = dy; // dv
            steepestDescentImages[2].at<double>(row, col) = dx * x; // du/dx
            steepestDescentImages[3].at<double>(row, col) = dx * y; // du/dy
            steepestDescentImages[4].at<double>(row, col) = dy * x; // dv/dx
            steepestDescentImages[5].at<double>(row, col) = dy * y; // dv/dy
            
            // Second order parameters (if applicable)
            if (m_order == SECOND_ORDER) {
                steepestDescentImages[6].at<double>(row, col) = dx * x * x / 2.0; // d²u/dx²
                steepestDescentImages[7].at<double>(row, col) = dx * x * y; // d²u/dxdy
                steepestDescentImages[8].at<double>(row, col) = dx * y * y / 2.0; // d²u/dy²
                steepestDescentImages[9].at<double>(row, col) = dy * x * x / 2.0; // d²v/dx²
                steepestDescentImages[10].at<double>(row, col) = dy * x * y; // d²v/dxdy
                steepestDescentImages[11].at<double>(row, col) = dy * y * y / 2.0; // d²v/dy²
            }
        }
    }
}

void RGDIC::ICGNOptimizer::computeHessian(const std::vector<cv::Mat>& steepestDescentImages, 
                                        cv::Mat& hessian) const{
    // Initialize Hessian matrix
    hessian = cv::Mat::zeros(m_numParams, m_numParams, CV_64F);
    
    // For each parameter pair
    for (int i = 0; i < m_numParams; i++) {
        for (int j = i; j < m_numParams; j++) { // Take advantage of symmetry
            double sum = 0;
            
            // Sum over all pixels in subset
            for (int y = 0; y < 2 * m_subsetRadius + 1; y++) {
                for (int x = 0; x < 2 * m_subsetRadius + 1; x++) {
                    sum += steepestDescentImages[i].at<double>(y, x) * steepestDescentImages[j].at<double>(y, x);
                }
            }
            
            // Set Hessian element
            hessian.at<double>(i, j) = sum;
            
            // Set symmetric element
            if (i != j) {
                hessian.at<double>(j, i) = sum;
            }
        }
    }
}

cv::Point2f RGDIC::ICGNOptimizer::warpPoint(const cv::Point2f& pt, const cv::Mat& warpParams)const {
    double x = pt.x;
    double y = pt.y;
    
    // Extract parameters
    double u = warpParams.at<double>(0);
    double v = warpParams.at<double>(1);
    double dudx = warpParams.at<double>(2);
    double dudy = warpParams.at<double>(3);
    double dvdx = warpParams.at<double>(4);
    double dvdy = warpParams.at<double>(5);
    
    // First-order warp
    double warpedX = x + u + dudx * x + dudy * y;
    double warpedY = y + v + dvdx * x + dvdy * y;
    
    // Add second-order terms if using second-order shape function
    if (m_order == SECOND_ORDER && m_numParams >= 12) {
        double d2udx2 = warpParams.at<double>(6);
        double d2udxdy = warpParams.at<double>(7);
        double d2udy2 = warpParams.at<double>(8);
        double d2vdx2 = warpParams.at<double>(9);
        double d2vdxdy = warpParams.at<double>(10);
        double d2vdy2 = warpParams.at<double>(11);
        
        warpedX += 0.5 * d2udx2 * x * x + d2udxdy * x * y + 0.5 * d2udy2 * y * y;
        warpedY += 0.5 * d2vdx2 * x * x + d2vdxdy * x * y + 0.5 * d2vdy2 * y * y;
    }
    
    return cv::Point2f(warpedX, warpedY);
}

void RGDIC::ICGNOptimizer::computeWarpJacobian(const cv::Point2f& pt, cv::Mat& jacobian)const {
    double x = pt.x;
    double y = pt.y;
    
    // For first-order shape function
    if (m_order == FIRST_ORDER) {
        jacobian = (cv::Mat_<double>(2, 6) << 
                   1, 0, x, y, 0, 0,
                   0, 1, 0, 0, x, y);
    }
    // For second-order shape function
    else {
        jacobian = (cv::Mat_<double>(2, 12) << 
                   1, 0, x, y, 0, 0, 0.5*x*x, x*y, 0.5*y*y, 0, 0, 0,
                   0, 1, 0, 0, x, y, 0, 0, 0, 0.5*x*x, x*y, 0.5*y*y);
    }
}

double RGDIC::ICGNOptimizer::interpolate(const cv::Mat& image, const cv::Point2f& pt)const {
    // Bounds check
    if (pt.x < 0 || pt.x >= image.cols - 1 || pt.y < 0 || pt.y >= image.rows - 1) {
        return 0;
    }
    
    // Get integer and fractional parts
    int x1 = static_cast<int>(pt.x);
    int y1 = static_cast<int>(pt.y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;
    
    double fx = pt.x - x1;
    double fy = pt.y - y1;
    
    // Bilinear interpolation
    double val = (1 - fx) * (1 - fy) * image.at<uchar>(y1, x1) +
                fx * (1 - fy) * image.at<uchar>(y1, x2) +
                (1 - fx) * fy * image.at<uchar>(y2, x1) +
                fx * fy * image.at<uchar>(y2, x2);
    
    return val;
}