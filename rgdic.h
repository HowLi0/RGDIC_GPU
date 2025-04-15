#ifndef RGDIC_H
#define RGDIC_H

#include <vector>
#include <queue>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thread>
#include <mutex>
#include <future>
#include <omp.h>

class RGDIC {
public:
    // Enumeration for shape function order
    enum ShapeFunctionOrder {
        FIRST_ORDER = 1,  // 6 parameters: u, v, du/dx, du/dy, dv/dx, dv/dy
        SECOND_ORDER = 2  // 12 parameters: first order + second derivatives
    };
    
    // Enumeration for computation mode
    enum ComputationMode {
        CPU_SINGLE_THREAD = 0,  // Original implementation
        CPU_MULTI_THREAD = 1,   // CPU with multi-threading
        GPU_CUDA = 2,           // CUDA GPU acceleration
        HYBRID = 3              // Hybrid CPU/GPU approach
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
          ShapeFunctionOrder order = SECOND_ORDER,
          ComputationMode mode = CPU_MULTI_THREAD,
          int numThreads = -1);  // Default to use all available cores
    
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

    cv::Mat visualizeDisplacement(const cv::Mat& u, const cv::Mat& v, const cv::Mat& mask = cv::Mat()) const;

private:
    // Algorithm parameters
    int m_subsetRadius;
    double m_convergenceThreshold;
    int m_maxIterations;
    double m_ccThreshold;
    double m_deltaDispThreshold;
    ShapeFunctionOrder m_order;
    ComputationMode m_mode;
    int m_numThreads;
    
    // Number of shape function parameters (6 for first-order, 12 for second-order)
    int m_numParams;
    
    // Class for ICGN (Inverse Compositional Gauss-Newton) optimization
    class ICGNOptimizer {
    public:
        ICGNOptimizer(const cv::Mat& refImage, const cv::Mat& defImage,
                    int subsetRadius, ShapeFunctionOrder order,
                    double convergenceThreshold, int maxIterations,
                    ComputationMode mode);
        
        ~ICGNOptimizer();
        
        // Finds optimal deformation parameters for a point
        bool optimize(const cv::Point& refPoint, cv::Mat& warpParams, double& zncc) const;
        
        // Initial guess using ZNCC grid search
        bool initialGuess(const cv::Point& refPoint, cv::Mat& warpParams, double& zncc) const;
        
        // GPU version of the optimization function
        bool optimizeGPU(const cv::Point& refPoint, cv::Mat& warpParams, double& zncc) const;

        // GPU version of the initial guess function
        bool initialGuessGPU(const cv::Point& refPoint, cv::Mat& warpParams, double& zncc) const;
        
    private:
        const cv::Mat& m_refImage;
        const cv::Mat& m_defImage;
        unsigned char* d_refImage;  // Device pointer for reference image
        unsigned char* d_defImage;  // Device pointer for deformed image
        float* d_warpedSubset;      // Device pointer for warped subset
        float* d_refSubset;         // Device pointer for reference subset
        float* d_steepestDescentImages;  // Device pointer for steepest descent images
        float* d_hessian;           // Device pointer for Hessian matrix
        float* d_errorVector;       // Device pointer for error vector
        float* d_zncc;              // Device pointer for ZNCC values
        size_t m_refPitch;          // Pitch for reference image
        size_t m_defPitch;          // Pitch for deformed image
        int m_subsetRadius;
        ShapeFunctionOrder m_order;
        double m_convergenceThreshold;
        int m_maxIterations;
        int m_numParams;
        ComputationMode m_mode;
        
        // Computes ZNCC between reference subset and warped current subset
        double computeZNCC(const cv::Point& refPoint, const cv::Mat& warpParams) const;
        
        // GPU version of ZNCC computation using custom CUDA kernel
        double computeZNCCGPU(const cv::Point& refPoint, const cv::Mat& warpParams) const;
        
        // Computes steepest descent images
        void computeSteepestDescentImages(const cv::Point& refPoint, 
                                         std::vector<cv::Mat>& steepestDescentImages) const;
        
        // GPU version of steepest descent images computation
        void computeSteepestDescentImagesGPU(const cv::Point& refPoint) const;
        
        // Computes Hessian matrix
        void computeHessian(const std::vector<cv::Mat>& steepestDescentImages, cv::Mat& hessian) const;
        
        // GPU version of Hessian computation
        void computeHessianGPU() const;
        
        // Warps a point using shape function parameters
        cv::Point2f warpPoint(const cv::Point2f& pt, const cv::Mat& warpParams) const;
        
        // Computes warp Jacobian
        void computeWarpJacobian(const cv::Point2f& pt, cv::Mat& jacobian) const;
        
        // Interpolates image intensity at non-integer coordinates
        double interpolate(const cv::Mat& image, const cv::Point2f& pt) const;
        
        // Initialize GPU memory
        void initGPUMemory();
        
        // Free GPU memory
        void freeGPUMemory();
    };
    
    // Calculate signed distance array for ROI (used for seed point selection)
    cv::Mat calculateSDA(const cv::Mat& roi);
    
    // Find initial seed point using SDA
    cv::Point findSeedPoint(const cv::Mat& roi, const cv::Mat& sda);
    
    using PriorityQueue = std::priority_queue<
        std::pair<cv::Point, double>,
        std::vector<std::pair<cv::Point, double>>,
        std::function<bool(const std::pair<cv::Point, double>&, const std::pair<cv::Point, double>&)>>;
    
    // Function to analyze a point in the ROI
    bool analyzePoint(const cv::Point& point, ICGNOptimizer& optimizer, 
                    const cv::Mat& roi, DisplacementResult& result, 
                    PriorityQueue& queue, cv::Mat& analyzedPoints);
                    
    // Multi-threaded implementation of reliability-guided search
    void reliabilityGuidedSearch(const cv::Mat& refImage, const cv::Mat& defImage, 
                                const cv::Mat& roi, cv::Point seedPoint, 
                                DisplacementResult& result);
                                
    // GPU implementation of reliability-guided search
    void reliabilityGuidedSearchGPU(const cv::Mat& refImage, const cv::Mat& defImage, 
                                  const cv::Mat& roi, cv::Point seedPoint, 
                                  DisplacementResult& result);
                                  
    // Structure to hold segment data for parallel processing
    struct PointSegment {
        std::vector<cv::Point> points;
        cv::Rect bounds;
    };
    
    // Divide ROI into segments for parallel processing
    std::vector<PointSegment> divideROIIntoSegments(const cv::Mat& roi, int numSegments);
    
    // Thread-safe queue operations
    std::mutex m_queueMutex;
    std::mutex m_resultMutex;
};

// CUDA kernels and helper functions
namespace RGDIC_CUDA {
    // CUDA error checking
    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
        if (code != cudaSuccess) {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }
    
    // Initialize CUDA device
    void initializeGPU();
    
    // Get CUDA device information
    int getGPUDeviceCount();
    bool checkGPUCompatibility();
    void printGPUInfo();
    
    // CUDA kernel declarations
    __global__ void computeZNCCKernel(const unsigned char* refImage, size_t refPitch,
                                    const unsigned char* defImage, size_t defPitch,
                                    int subsetRadius, int refPointX, int refPointY,
                                    float* warpParams, int numParams, float* result);
                                    
    __global__ void interpolateKernel(const unsigned char* image, size_t pitch,
                                    int width, int height,
                                    float* points, int numPoints,
                                    float* results);
                                    
    __global__ void steepestDescentKernel(const unsigned char* refImage, size_t refPitch,
                                        int width, int height,
                                        int refPointX, int refPointY,
                                        int subsetRadius, int numParams,
                                        float* steepestDescentImages);
                                        
    __global__ void computeHessianKernel(float* steepestDescentImages,
                                       int subsetDiameter,
                                       int numParams,
                                       float* hessian);
                                       
    __global__ void warpSubsetKernel(const unsigned char* defImage, size_t defPitch,
                                   int width, int height,
                                   int refPointX, int refPointY,
                                   int subsetRadius,
                                   float* warpParams, int numParams,
                                   float* warpedSubset);
                                   
    __global__ void errorVectorKernel(float* refSubset,
                                    float* warpedSubset,
                                    float* steepestDescentImages,
                                    int subsetDiameter,
                                    int numParams,
                                    float* errorVector);
}


#endif // RGDIC_H