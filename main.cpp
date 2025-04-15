#include "rgdic.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <string>

// Function to log messages with timestamp
void logMessage(const std::string& message) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    
    std::cout << "[" << ss.str() << "] " << message << std::endl;
}

// Function to create synthetic test images
void createSyntheticImages(cv::Mat& refImage, cv::Mat& defImage, cv::Mat& trueDispX, cv::Mat& trueDispY, int width = 1000, int height = 1000) {
    logMessage("Creating synthetic test images...");
    
    // Create reference image with random speckle pattern
    refImage = cv::Mat::zeros(height, width, CV_8UC1);
    
    // Create speckle pattern (random dots)
    cv::RNG rng(12345);
    for (int i = 0; i < 50000; i++) {
        int x = rng.uniform(0, width);
        int y = rng.uniform(0, height);
        int radius = rng.uniform(2, 6);
        cv::circle(refImage, cv::Point(x, y), radius, cv::Scalar(255), -1);
    }
    
    // Apply Gaussian blur to make it more realistic
    cv::GaussianBlur(refImage, refImage, cv::Size(5, 5), 1.0);
    
    // Create true displacement fields (analytical functions)
    trueDispX = cv::Mat::zeros(height, width, CV_64F);
    trueDispY = cv::Mat::zeros(height, width, CV_64F);
    
    // Create a complex displacement field (a combination of translation, rotation, and sinusoidal patterns)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Normalized coordinates (0 to 1)
            double nx = static_cast<double>(x) / width;
            double ny = static_cast<double>(y) / height;
            
            // Displacement components
            double dispX = 5.0 * sin(nx * 2 * CV_PI) * cos(ny * 3 * CV_PI) + 3.0;  // x displacement
            double dispY = 4.0 * cos(nx * 3 * CV_PI) * sin(ny * 2 * CV_PI) + 2.0;  // y displacement
            
            trueDispX.at<double>(y, x) = dispX;
            trueDispY.at<double>(y, x) = dispY;
        }
    }
    
    // Create deformed image using the true displacement field
    defImage = cv::Mat::zeros(height, width, CV_8UC1);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Get true displacement at this pixel
            double dx = trueDispX.at<double>(y, x);
            double dy = trueDispY.at<double>(y, x);
            
            // Source pixel in reference image
            double srcX = x - dx;
            double srcY = y - dy;
            
            // Check if source is within image bounds
            if (srcX >= 0 && srcX < width - 1 && srcY >= 0 && srcY < height - 1) {
                // Bilinear interpolation
                int x1 = static_cast<int>(srcX);
                int y1 = static_cast<int>(srcY);
                int x2 = x1 + 1;
                int y2 = y1 + 1;
                
                double fx = srcX - x1;
                double fy = srcY - y1;
                
                double val = (1 - fx) * (1 - fy) * refImage.at<uchar>(y1, x1) +
                            fx * (1 - fy) * refImage.at<uchar>(y1, x2) +
                            (1 - fx) * fy * refImage.at<uchar>(y2, x1) +
                            fx * fy * refImage.at<uchar>(y2, x2);
                
                defImage.at<uchar>(y, x) = static_cast<uchar>(val);
            }
        }
    }
    
    logMessage("Synthetic test images created successfully.");
}

// Function to load a pair of real images
bool loadRealImagePair(const std::string& refPath, const std::string& defPath, 
                    cv::Mat& refImage, cv::Mat& defImage) {
    logMessage("Loading real image pair...");
    
    refImage = cv::imread(refPath, cv::IMREAD_GRAYSCALE);
    if (refImage.empty()) {
        logMessage("Error: Could not load reference image from " + refPath);
        return false;
    }
    
    defImage = cv::imread(defPath, cv::IMREAD_GRAYSCALE);
    if (defImage.empty()) {
        logMessage("Error: Could not load deformed image from " + defPath);
        return false;
    }
    
    logMessage("Real image pair loaded successfully.");
    return true;
}

// Main function for running the RGDIC analysis
void runRGDICAnalysis(const cv::Mat& refImage, const cv::Mat& defImage, 
                    const cv::Mat& trueDispX, const cv::Mat& trueDispY) {
    logMessage("Starting RGDIC analysis...");
    
    // Create a region of interest (ROI) - here we use the entire image except for a small border
    cv::Mat roi = cv::Mat::ones(refImage.size(), CV_8UC1) * 255;
    int borderSize = 30;
    cv::rectangle(roi, cv::Rect(borderSize, borderSize, 
                             roi.cols - 2 * borderSize, roi.rows - 2 * borderSize), 
                cv::Scalar(255), -1);
    
    // Run RGDIC with different computation modes for comparison
    std::vector<RGDIC::ComputationMode> modes = {
        RGDIC::CPU_SINGLE_THREAD,
        RGDIC::CPU_MULTI_THREAD,
        RGDIC::GPU_CUDA
    };
    
    std::vector<std::string> modeNames = {
        "CPU Single-threaded",
        "CPU Multi-threaded",
        "GPU CUDA"
    };
    
    // Parameters for RGDIC
    int subsetRadius = 15;
    double convergenceThreshold = 0.001;
    int maxIterations = 30;
    double ccThreshold = 0.8;
    double deltaDispThreshold = 1.0;
    RGDIC::ShapeFunctionOrder order = RGDIC::SECOND_ORDER;
    
    // Run RGDIC with different modes
    for (size_t i = 0; i < modes.size(); i++) {
        logMessage("Running RGDIC with " + modeNames[i] + " mode...");
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Create RGDIC instance with current mode
        RGDIC rgdic(subsetRadius, convergenceThreshold, maxIterations,
                  ccThreshold, deltaDispThreshold, order, modes[i]);
        
        // Compute displacement fields
        RGDIC::DisplacementResult result = rgdic.compute(refImage, defImage, roi);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        logMessage(modeNames[i] + " mode completed in " + std::to_string(duration.count() / 1000.0) + " seconds.");
        
        // Count valid points
        int validPoints = cv::countNonZero(result.validMask);
        int totalROIPoints = cv::countNonZero(roi);
        double coverage = 100.0 * validPoints / totalROIPoints;
        
        logMessage("Coverage: " + std::to_string(validPoints) + " out of " + 
                std::to_string(totalROIPoints) + " (" + std::to_string(coverage) + "%)");
        
        // Display results
        std::string windowName = "RGDIC Results - " + modeNames[i];
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(windowName, 1200, 800);
        
        // Create a visualization of the results
        cv::Mat resultVis;
        
        // If we have ground truth, display error maps
        if (!trueDispX.empty() && !trueDispY.empty()) {
            rgdic.evaluateErrors(result, trueDispX, trueDispY);
            
            // Display true vs computed displacement fields
            cv::Mat trueVsComputed;
            cv::hconcat(rgdic.visualizeDisplacement(trueDispX, trueDispY),
                       rgdic.visualizeDisplacement(result.u, result.v),
                       trueVsComputed);
                       
            // Save results
            cv::imwrite("results_" + std::to_string(i) + "_mode_" + modeNames[i] + ".png", trueVsComputed);
            
            // Show results
            cv::imshow(windowName, trueVsComputed);
        } else {
            // Just display computed displacement fields
            resultVis = rgdic.visualizeDisplacement(result.u, result.v);
            cv::imshow(windowName, resultVis);
            cv::imwrite("results_" + std::to_string(i) + "_mode_" + modeNames[i] + ".png", resultVis);
        }
        
        cv::waitKey(100); // Brief pause to allow display
    }
    
    logMessage("RGDIC analysis completed.");
    cv::waitKey(0); // Wait for user to close windows
}

int main(int argc, char* argv[]) {
    logMessage("RGDIC_GPU Demo Application Starting...");
    
    // Check CUDA availability
    if (RGDIC_CUDA::getGPUDeviceCount() > 0) {
        logMessage("CUDA device detected!");
        RGDIC_CUDA::printGPUInfo();
    } else {
        logMessage("No CUDA device detected. Will use CPU modes only.");
    }
    
    // Command line arguments
    // 1. Synthetic or real mode: 0 for synthetic, 1 for real
    // 2. For real mode: path to reference image
    // 3. For real mode: path to deformed image
    
    bool useSyntheticData = true;
    std::string refImagePath, defImagePath;
    
    if (argc > 1) {
        int mode = std::stoi(argv[1]);
        useSyntheticData = (mode == 0);
        
        if (!useSyntheticData && argc >= 4) {
            refImagePath = argv[2];
            defImagePath = argv[3];
        }
    }
    
    // Image data
    cv::Mat refImage, defImage;
    cv::Mat trueDispX, trueDispY; // Ground truth displacement (only for synthetic data)
    
    if (useSyntheticData) {
        // Create synthetic test images
        createSyntheticImages(refImage, defImage, trueDispX, trueDispY);
        
        // Save synthetic images for inspection
        cv::imwrite("synthetic_reference.png", refImage);
        cv::imwrite("synthetic_deformed.png", defImage);
        
        logMessage("Synthetic images saved to disk.");
    } else {
        // Load real images
        if (!loadRealImagePair(refImagePath, defImagePath, refImage, defImage)) {
            logMessage("Error loading images. Exiting.");
            return -1;
        }
        
        // For real images, we don't have ground truth
        trueDispX = cv::Mat();
        trueDispY = cv::Mat();
    }
    
    // Run the analysis
    runRGDICAnalysis(refImage, defImage, trueDispX, trueDispY);
    
    logMessage("RGDIC_GPU Demo Application Completed.");
    return 0;
}