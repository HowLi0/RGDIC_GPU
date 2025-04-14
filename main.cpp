#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "RGDIC.h"

// Global variables for ROI drawing
cv::Mat roiImage;
cv::Mat roi;
std::vector<cv::Point> roiPoints;
bool roiFinished = false;

// Mouse callback function for drawing ROI
void drawROI(int event, int x, int y, int flags, void* userdata) {
    if (roiFinished)
        return;
        
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Add point to the ROI
        roiPoints.push_back(cv::Point(x, y));
        
        // Draw a circle at the clicked point
        cv::circle(roiImage, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
        
        // If we have at least two points, draw a line between the last two points
        if (roiPoints.size() > 1) {
            cv::line(roiImage, roiPoints[roiPoints.size() - 2], roiPoints[roiPoints.size() - 1], 
                    cv::Scalar(0, 0, 255), 2);
        }
        
        cv::imshow("Draw ROI", roiImage);
    }
    else if (event == cv::EVENT_MOUSEMOVE && !roiPoints.empty()) {
        // Show a temporary line from the last point to the current position
        cv::Mat tempImage = roiImage.clone();
        cv::line(tempImage, roiPoints.back(), cv::Point(x, y), cv::Scalar(0, 0, 255), 2);
        cv::imshow("Draw ROI", tempImage);
    }
}

// Function to generate synthetic speckle pattern images for testing
void generateSyntheticImages(cv::Mat& refImg, cv::Mat& defImg, 
                           cv::Mat& trueDispX, cv::Mat& trueDispY,
                           int width = 500, int height = 500) {
    // Create reference image with speckle pattern
    refImg = cv::Mat::zeros(height, width, CV_8UC1);
    cv::RNG rng(12345);
    
    // Generate random speckle pattern
    for (int i = 0; i <4000; i++) {
        int x = rng.uniform(0, width);
        int y = rng.uniform(0, height);
        int radius = rng.uniform(2, 4);
        cv::circle(refImg, cv::Point(x, y), radius, cv::Scalar(255), -1);
    }
    
    // Apply Gaussian blur for more realistic speckles
    cv::GaussianBlur(refImg, refImg, cv::Size(3, 3), 0.8);
    
    // Create true displacement field (sinusoidal displacement)
    trueDispX = cv::Mat::zeros(height, width, CV_32F);
    trueDispY = cv::Mat::zeros(height, width, CV_32F);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Generate displacement field
            double dx = 3.0 * sin(2.0 * CV_PI * y / height);
            double dy = 2.0 * cos(2.0 * CV_PI * x / width);
            
            trueDispX.at<float>(y, x) = dx;
            trueDispY.at<float>(y, x) = dy;
        }
    }
    
    // Create deformed image using the displacement field
    defImg = cv::Mat::zeros(height, width, CV_8UC1);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Get displacement at this point
            float dx = trueDispX.at<float>(y, x);
            float dy = trueDispY.at<float>(y, x);
            
            // Source position in reference image
            float srcX = x - dx;
            float srcY = y - dy;
            
            // Check if within bounds
            if (srcX >= 0 && srcX < width - 1 && srcY >= 0 && srcY < height - 1) {
                // Bilinear interpolation
                int x1 = floor(srcX);
                int y1 = floor(srcY);
                int x2 = x1 + 1;
                int y2 = y1 + 1;
                
                float wx = srcX - x1;
                float wy = srcY - y1;
                
                float val = (1 - wx) * (1 - wy) * refImg.at<uchar>(y1, x1) +
                           wx * (1 - wy) * refImg.at<uchar>(y1, x2) +
                           (1 - wx) * wy * refImg.at<uchar>(y2, x1) +
                           wx * wy * refImg.at<uchar>(y2, x2);
                
                defImg.at<uchar>(y, x) = static_cast<uchar>(val);
            }
        }
    }
    
    // Fill any holes in the deformed image
    cv::Mat mask = (defImg == 0);
    cv::inpaint(defImg, mask, defImg, 5, cv::INPAINT_TELEA);
}

// Function to let user draw ROI manually
cv::Mat createManualROI(const cv::Mat& image) {
    // Create ROI mask
    cv::Mat manualROI = cv::Mat::zeros(image.size(), CV_8UC1);
    
    // Create an image for drawing
    if (image.channels() == 1) {
        cv::cvtColor(image, roiImage, cv::COLOR_GRAY2BGR);
    } else {
        roiImage = image.clone();
    }
    
    // Set up window and mouse callback
    cv::namedWindow("Draw ROI", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Draw ROI", drawROI);
    
    // Instructions
    std::cout << "Draw ROI: Click to add points, press Enter to complete, Esc to cancel" << std::endl;
    
    // Wait for ROI drawing to complete
    roiPoints.clear();
    roiFinished = false;
    
    while (true) {
        char key = (char)cv::waitKey(10);
        
        // Enter key completes the ROI
        if (key == 13 && roiPoints.size() >= 3) {
            // Complete the polygon by connecting to the first point
            cv::line(roiImage, roiPoints.back(), roiPoints.front(), cv::Scalar(0, 0, 255), 2);
            cv::imshow("Draw ROI", roiImage);
            
            // Fill the ROI polygon
            std::vector<std::vector<cv::Point>> contours = { roiPoints };
            cv::fillPoly(manualROI, contours, cv::Scalar(255));
            
            roiFinished = true;
            break;
        }
        // Escape key cancels
        else if (key == 27) {
            manualROI = cv::Mat::ones(image.size(), CV_8UC1);  // Default to full image
            break;
        }
        
        cv::imshow("Draw ROI", roiImage);
    }
    
    cv::destroyWindow("Draw ROI");
    return manualROI;
}

// Function to export displacement data to CSV file
void exportToCSV(const cv::Mat& u, const cv::Mat& v, const cv::Mat& validMask, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "x,y,u_displacement,v_displacement" << std::endl;
    
    // Write data points
    for (int y = 0; y < u.rows; y++) {
        for (int x = 0; x < u.cols; x++) {
            if (validMask.at<uchar>(y, x)) {
                file << x << "," << y << "," << u.at<double>(y, x) << "," << v.at<double>(y, x) << std::endl;
            }
        }
    }
    
    std::cout << "Displacement data exported to: " << filename << std::endl;
}

// Function to create a color map visualization with a scale bar
cv::Mat visualizeDisplacementWithScaleBar(const cv::Mat& displacement, const cv::Mat& validMask, 
                                      double minVal, double maxVal, 
                                      const std::string& title,
                                      int colorMap = cv::COLORMAP_JET) {
    // Normalize displacement for visualization
    cv::Mat dispNorm;
    cv::normalize(displacement, dispNorm, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);
    
    // Apply color map
    cv::Mat colorDisp;
    cv::applyColorMap(dispNorm, colorDisp, colorMap);
    
    // Apply valid mask
    cv::Mat background = cv::Mat::zeros(displacement.size(), CV_8UC3);
    colorDisp.copyTo(background, validMask);
    
    // Add border around the image for the scale bar and title
    int topBorder = 40;  // Space for title
    int bottomBorder = 70;  // Space for scale bar
    int leftRightBorder = 30;
    
    cv::Mat result;
    cv::copyMakeBorder(background, result, topBorder, bottomBorder, leftRightBorder, leftRightBorder, 
                     cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    
    // Add title
    cv::putText(result, title, cv::Point(leftRightBorder, 30), cv::FONT_HERSHEY_SIMPLEX, 
               0.8, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    
    // Add scale bar
    int barWidth = background.cols - 60;
    int barHeight = 20;
    int barX = leftRightBorder + 30;
    int barY = result.rows - bottomBorder + 20;
    
    // Create gradient for scale bar
    cv::Mat scaleBar(barHeight, barWidth, CV_8UC3);
    for (int x = 0; x < barWidth; x++) {
        double value = (double)x / barWidth * 255.0;
        cv::Mat color;
        cv::Mat temp(1, 1, CV_8UC1, cv::Scalar(value));
        cv::applyColorMap(temp, color, colorMap);
        cv::rectangle(scaleBar, cv::Point(x, 0), cv::Point(x, barHeight), color.at<cv::Vec3b>(0, 0), 1);
    }
    
    // Place scale bar on the result image
    scaleBar.copyTo(result(cv::Rect(barX, barY, barWidth, barHeight)));
    
    // Add min and max values as text
    std::stringstream ssMin, ssMax;
    ssMin << std::fixed << std::setprecision(2) << minVal;
    ssMax << std::fixed << std::setprecision(2) << maxVal;
    
    cv::putText(result, ssMin.str(), cv::Point(barX - 5, barY + barHeight + 15), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    cv::putText(result, ssMax.str(), cv::Point(barX + barWidth - 10, barY + barHeight + 15), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
               
    // Add "pixels" unit text
    cv::putText(result, "[pixels]", cv::Point(barX + barWidth/2 - 20, barY + barHeight + 15), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    
    return result;
}

int main(int argc, char** argv) {
    // Flags to control behavior
    bool useSyntheticImages = true;
    bool useFirstOrderShapeFunction = false; // Set to false to use second-order
    bool useManualROI = true; // Set to true to manually draw ROI
    
    cv::Mat refImage, defImage;
    cv::Mat trueDispX, trueDispY;
    
    if (useSyntheticImages) {
        std::cout << "Generating synthetic speckle pattern images..." << std::endl;
        generateSyntheticImages(refImage, defImage, trueDispX, trueDispY);
        
        // Save the generated images
        cv::imwrite("E:/code_C++/RGDIC/synthetic_reference.png", refImage);
        cv::imwrite("E:/code_C++/RGDIC/synthetic_deformed.png", defImage);
        
        // Display images
        cv::imshow("Reference Image", refImage);
        cv::imshow("Deformed Image", defImage);
        cv::waitKey(100); // Brief pause to ensure windows are displayed
        
        // Visualize true displacement fields
        cv::Mat trueDispXViz, trueDispYViz;
        
        // Find min/max values for visualization
        double minX, maxX, minY, maxY;
        cv::minMaxLoc(trueDispX, &minX, &maxX);
        cv::minMaxLoc(trueDispY, &minY, &maxY);
        
        // Create visualizations with scale bars
        cv::Mat trueXViz = visualizeDisplacementWithScaleBar(trueDispX, cv::Mat::ones(trueDispX.size(), CV_8UC1),
                                                          minX, maxX, "True X Displacement");
        cv::Mat trueYViz = visualizeDisplacementWithScaleBar(trueDispY, cv::Mat::ones(trueDispY.size(), CV_8UC1),
                                                          minY, maxY, "True Y Displacement");
        
        cv::imshow("True X Displacement", trueXViz);
        cv::imshow("True Y Displacement", trueYViz);
        
        cv::imwrite("E:/code_C++/RGDIC/true_disp_x.png", trueXViz);
        cv::imwrite("E:/code_C++/RGDIC/true_disp_y.png", trueYViz);
    } else {
        // Load real images if provided
        if (argc < 3) {
            std::cout << "Usage: " << argv[0] << " <reference_image> <deformed_image>" << std::endl;
            return -1;
        }
        
        refImage = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        defImage = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
        
        if (refImage.empty() || defImage.empty()) {
            std::cerr << "Error loading images!" << std::endl;
            return -1;
        }
        
        // Display images
        cv::imshow("Reference Image", refImage);
        cv::imshow("Deformed Image", defImage);
        cv::waitKey(100);
    }
    
    // Create ROI
    cv::Mat roi;
    if (useManualROI) {
        // Let user draw ROI
        roi = createManualROI(refImage);
        
        // Display the ROI
        cv::Mat roiViz;
        cv::cvtColor(refImage, roiViz, cv::COLOR_GRAY2BGR);
        roiViz.setTo(cv::Scalar(0, 0, 255), roi);
        cv::addWeighted(roiViz, 0.3, cv::Mat(roiViz.size(), roiViz.type(), cv::Scalar(0)), 0.7, 0, roiViz);
        
        // Convert reference image to color and copy only the ROI area
        cv::Mat colorRef;
        cv::cvtColor(refImage, colorRef, cv::COLOR_GRAY2BGR);
        colorRef.copyTo(roiViz, roi);
        
        cv::imshow("Selected ROI", roiViz);
        cv::waitKey(100);
        cv::imwrite("E:/code_C++/RGDIC/selected_roi.png", roiViz);
    } else {
        // Create automatic ROI (exclude border regions)
        int borderWidth = 25;
        roi = cv::Mat::ones(refImage.size(), CV_8UC1);
        cv::rectangle(roi, cv::Point(0, 0), cv::Point(roi.cols-1, roi.rows-1), 0, borderWidth);
    }
    
    // Create RGDIC object
    RGDIC::ShapeFunctionOrder order = useFirstOrderShapeFunction ? 
                                    RGDIC::FIRST_ORDER : RGDIC::SECOND_ORDER;
    
    // Subset radius of 31 means a 63x63 subset size
    RGDIC dic(15, 0.001, 30, 0.8, 1.0, order);
    
    std::cout << "Running RGDIC algorithm with " 
              << (useFirstOrderShapeFunction ? "first" : "second") 
              << "-order shape function..." << std::endl;
    
    // Measure execution time
    double t = (double)cv::getTickCount();
    
    // Run RGDIC algorithm
    auto result = dic.compute(refImage, defImage, roi);
    
    // Calculate execution time
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "RGDIC computation completed in " << t << " seconds." << std::endl;
    
    // Count valid points
    int validPoints = cv::countNonZero(result.validMask);
    int totalRoiPoints = cv::countNonZero(roi);
    double coverage = 100.0 * validPoints / totalRoiPoints;
    
    std::cout << "Analysis coverage: " << coverage << "% (" 
              << validPoints << " of " << totalRoiPoints << " points)" << std::endl;
    
    // Find min/max values of computed displacement
    double minU, maxU, minV, maxV;
    cv::minMaxLoc(result.u, &minU, &maxU, nullptr, nullptr, result.validMask);
    cv::minMaxLoc(result.v, &minV, &maxV, nullptr, nullptr, result.validMask);
    
    // Create visualizations with scale bars
    cv::Mat uViz = visualizeDisplacementWithScaleBar(result.u, result.validMask, 
                                                   minU, maxU, "X Displacement");
    cv::Mat vViz = visualizeDisplacementWithScaleBar(result.v, result.validMask, 
                                                   minV, maxV, "Y Displacement");
    
    // Display results
    cv::imshow("X Displacement", uViz);
    cv::imshow("Y Displacement", vViz);
    
    // Save results to disk
    cv::imwrite("E:/code_C++/RGDIC/computed_disp_x.png", uViz);
    cv::imwrite("E:/code_C++/RGDIC/computed_disp_y.png", vViz);
    
    // Export displacement data to CSV
    exportToCSV(result.u, result.v, result.validMask, "E:/code_C++/RGDIC/displacement_results.csv");
    
    // Calculate vector magnitude of displacement (for visualization)
    cv::Mat dispMag = cv::Mat::zeros(result.u.size(), CV_64F);
    for (int y = 0; y < dispMag.rows; y++) {
        for (int x = 0; x < dispMag.cols; x++) {
            if (result.validMask.at<uchar>(y, x)) {
                double dx = result.u.at<double>(y, x);
                double dy = result.v.at<double>(y, x);
                dispMag.at<double>(y, x) = std::sqrt(dx*dx + dy*dy);
            }
        }
    }
    
    // Find min/max of magnitude
    double minMag, maxMag;
    cv::minMaxLoc(dispMag, &minMag, &maxMag, nullptr, nullptr, result.validMask);
    
    // Create visualization of magnitude
    cv::Mat magViz = visualizeDisplacementWithScaleBar(dispMag, result.validMask, 
                                                    minMag, maxMag, "Displacement Magnitude");
    
    cv::imshow("Displacement Magnitude", magViz);
    cv::imwrite("E:/code_C++/RGDIC/computed_disp_magnitude.png", magViz);
    
    // Create vector field visualization on reference image
    cv::Mat vectorField;
    cv::cvtColor(refImage, vectorField, cv::COLOR_GRAY2BGR);
    
    // Draw displacement vectors (subsampled for clarity)
    int step = 5; // Sample every 10 pixels
    for (int y = 0; y < result.u.rows; y += step) {
        for (int x = 0; x < result.u.cols; x += step) {
            if (result.validMask.at<uchar>(y, x)) {
                double dx = result.u.at<double>(y, x);
                double dy = result.v.at<double>(y, x);
                
                // Scale for visibility (adjust this as needed)
                double scale = 5.0;
                cv::arrowedLine(vectorField, 
                              cv::Point(x, y), 
                              cv::Point(x + dx * scale, y + dy * scale), 
                              cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            }
        }
    }
    
    cv::imshow("Vector Field", vectorField);
    cv::imwrite("E:/code_C++/RGDIC/vector_field.png", vectorField);
    
    // If we have ground truth, calculate errors
    if (useSyntheticImages) {
        // Convert types for calculation
        cv::Mat uError(result.u.size(), CV_64F);
        cv::Mat vError(result.v.size(), CV_64F);
        
        for (int y = 0; y < result.u.rows; y++) {
            for (int x = 0; x < result.u.cols; x++) {
                if (result.validMask.at<uchar>(y, x)) {
                    uError.at<double>(y, x) = std::abs(result.u.at<double>(y, x) - trueDispX.at<float>(y, x));
                    vError.at<double>(y, x) = std::abs(result.v.at<double>(y, x) - trueDispY.at<float>(y, x));
                }
            }
        }
        
        // Find min/max error
        double minErrU, maxErrU, minErrV, maxErrV;
        cv::minMaxLoc(uError, &minErrU, &maxErrU, nullptr, nullptr, result.validMask);
        cv::minMaxLoc(vError, &minErrV, &maxErrV, nullptr, nullptr, result.validMask);
        
        // Create error visualizations
        cv::Mat uErrViz = visualizeDisplacementWithScaleBar(uError, result.validMask, 
                                                         minErrU, maxErrU, "X Displacement Error");
        cv::Mat vErrViz = visualizeDisplacementWithScaleBar(vError, result.validMask, 
                                                         minErrV, maxErrV, "Y Displacement Error");
        
        cv::imshow("X Displacement Error", uErrViz);
        cv::imshow("Y Displacement Error", vErrViz);
        
        cv::imwrite("E:/code_C++/RGDIC/error_disp_x.png", uErrViz);
        cv::imwrite("E:/code_C++/RGDIC/error_disp_y.png", vErrViz);
        
        // Calculate error statistics
        cv::Scalar meanErrU = cv::mean(uError, result.validMask);
        cv::Scalar meanErrV = cv::mean(vError, result.validMask);
        
        std::cout << "Error Statistics:" << std::endl;
        std::cout << "  X Displacement: Mean Error = " << meanErrU[0] 
                  << " pixels, Max Error = " << maxErrU << " pixels" << std::endl;
        std::cout << "  Y Displacement: Mean Error = " << meanErrV[0] 
                  << " pixels, Max Error = " << maxErrV << " pixels" << std::endl;
    }
    
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
    return 0;
}