#ifndef RGDIC_H
#define RGDIC_H

#include <vector>
#include <queue>
#include <cmath>
#include <opencv2/opencv.hpp>

class RGDIC {
public:
    // Enumeration for shape function order
    enum ShapeFunctionOrder {
        FIRST_ORDER = 1,  // 6 parameters: u, v, du/dx, du/dy, dv/dx, dv/dy
        SECOND_ORDER = 2  // 12 parameters: first order + second derivatives
    };
    
    // Result structure to hold displacement fields
    struct DisplacementResult {
        cv::Mat u;           // x-displacement field
        cv::Mat v;           // y-displacement field
        cv::Mat cc;          // correlation coefficient (ZNCC)
        cv::Mat validMask;   // valid points mask
    };
    
    // Constructor
    RGDIC(int subsetRadius = 15, 
          double convergenceThreshold = 0.001,
          int maxIterations = 30,
          double ccThreshold = 0.8,
          double deltaDispThreshold = 1.0,
          ShapeFunctionOrder order = SECOND_ORDER);
    
    // Main function to perform RGDIC analysis
    DisplacementResult compute(const cv::Mat& refImage, 
                              const cv::Mat& defImage,
                              const cv::Mat& roi);
    
    // Utility function to display results
    void displayResults(const cv::Mat& refImage, const DisplacementResult& result, 
                      const cv::Mat& trueDispX = cv::Mat(), const cv::Mat& trueDispY = cv::Mat());
    
    // Evaluate errors if ground truth is available
    void evaluateErrors(const DisplacementResult& result, 
                      const cv::Mat& trueDispX, const cv::Mat& trueDispY);

private:
    // Algorithm parameters
    int m_subsetRadius;
    double m_convergenceThreshold;
    int m_maxIterations;
    double m_ccThreshold;
    double m_deltaDispThreshold;
    ShapeFunctionOrder m_order;
    
    // Number of shape function parameters (6 for first-order, 12 for second-order)
    int m_numParams;
    
    // Representation of warp parameters
    // For first-order: p = [u, v, du/dx, du/dy, dv/dx, dv/dy]
    // For second-order: p = [u, v, du/dx, du/dy, dv/dx, dv/dy, 
    //                        d²u/dx², d²u/dxdy, d²u/dy², d²v/dx², d²v/dxdy, d²v/dy²]
    
    // Class for ICGN (Inverse Compositional Gauss-Newton) optimization
    class ICGNOptimizer {
    public:
        ICGNOptimizer(const cv::Mat& refImage, const cv::Mat& defImage,
                    int subsetRadius, ShapeFunctionOrder order,
                    double convergenceThreshold, int maxIterations);
        
        // Finds optimal deformation parameters for a point
        bool optimize(const cv::Point& refPoint, cv::Mat& warpParams, double& zncc)const ;
        
        // Initial guess using ZNCC grid search
        bool initialGuess(const cv::Point& refPoint, cv::Mat& warpParams, double& zncc)const ;
        
    private:
        const cv::Mat& m_refImage;
        const cv::Mat& m_defImage;
        int m_subsetRadius;
        ShapeFunctionOrder m_order;
        double m_convergenceThreshold;
        int m_maxIterations;
        int m_numParams;
        
        // Computes ZNCC between reference subset and warped current subset
        double computeZNCC(const cv::Point& refPoint, const cv::Mat& warpParams)const;
        
        // Computes steepest descent images
        void computeSteepestDescentImages(const cv::Point& refPoint, 
                                         std::vector<cv::Mat>& steepestDescentImages)const;
        
        // Computes Hessian matrix
        void computeHessian(const std::vector<cv::Mat>& steepestDescentImages, cv::Mat& hessian)const;
        
        // Warps a point using shape function parameters
        cv::Point2f warpPoint(const cv::Point2f& pt, const cv::Mat& warpParams)const;
        
        // Computes warp Jacobian
        void computeWarpJacobian(const cv::Point2f& pt, cv::Mat& jacobian)const;
        
        // Interpolates image intensity at non-integer coordinates
        double interpolate(const cv::Mat& image, const cv::Point2f& pt)const;
    };
    
    // Calculate signed distance array for ROI (used for seed point selection)
    cv::Mat calculateSDA(const cv::Mat& roi);
    
    // Find initial seed point using SDA
    cv::Point findSeedPoint(const cv::Mat& roi, const cv::Mat& sda);
    
    using PriorityQueue = std::priority_queue<
        std::pair<cv::Point, double>,
        std::vector<std::pair<cv::Point, double>>,
        std::function<bool(const std::pair<cv::Point, double>&, const std::pair<cv::Point, double>&)>>;
    
    // Updated function signature
    bool analyzePoint(const cv::Point& point, ICGNOptimizer& optimizer, 
                    const cv::Mat& roi, DisplacementResult& result, 
                    PriorityQueue& queue, cv::Mat& analyzedPoints);
};

#endif // RGDIC_H