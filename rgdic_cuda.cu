#include "rgdic.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <crt/math_functions.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace RGDIC_CUDA {

void initializeGPU() {
    // Set device to 0 by default
    cudaSetDevice(0);
    
    // Reset device to clear any previous state
    cudaDeviceReset();
}

int getGPUDeviceCount() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

bool checkGPUCompatibility() {
    int deviceCount = getGPUDeviceCount();
    if (deviceCount <= 0) {
        return false;
    }
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    // Check compute capability (minimum required is 3.0)
    if (deviceProp.major < 3) {
        return false;
    }
    
    return true;
}

void printGPUInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return;
    }
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        printf("GPU Device %d: \"%s\"\n", i, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %.2f GB\n", 
               static_cast<float>(deviceProp.totalGlobalMem) / (1024.0f * 1024.0f * 1024.0f));
        printf("  Multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  Max threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max thread dimensions: (%d, %d, %d)\n", 
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max grid dimensions: (%d, %d, %d)\n", 
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Warp size: %d\n", deviceProp.warpSize);
        printf("\n");
    }
}

// Bilinear interpolation kernel
__device__ float interpolateGPU(const unsigned char* image, size_t pitch,
                              int width, int height, float x, float y) {
    // Bounds check
    if (x < 0 || x >= width - 1 || y < 0 || y >= height - 1) {
        return 0.0f;
    }
    
    // Get integer and fractional parts
    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;
    
    float fx = x - x1;
    float fy = y - y1;
    
    // Get pointers to rows
    const unsigned char* row1 = (const unsigned char*)((const char*)image + y1 * pitch);
    const unsigned char* row2 = (const unsigned char*)((const char*)image + y2 * pitch);
    
    // Bilinear interpolation
    float val = (1 - fx) * (1 - fy) * row1[x1] +
               fx * (1 - fy) * row1[x2] +
               (1 - fx) * fy * row2[x1] +
               fx * fy * row2[x2];
    
    return val;
}

// Device function for point warping
__device__ void warpPointGPU(float x, float y, float* warpParams, int numParams,
                           float& warpedX, float& warpedY) {
    // Extract parameters
    float u = warpParams[0];
    float v = warpParams[1];
    float dudx = warpParams[2];
    float dudy = warpParams[3];
    float dvdx = warpParams[4];
    float dvdy = warpParams[5];
    
    // First-order warp
    warpedX = x + u + dudx * x + dudy * y;
    warpedY = y + v + dvdx * x + dvdy * y;
    
    // Add second-order terms if using second-order shape function
    if (numParams >= 12) {
        float d2udx2 = warpParams[6];
        float d2udxdy = warpParams[7];
        float d2udy2 = warpParams[8];
        float d2vdx2 = warpParams[9];
        float d2vdxdy = warpParams[10];
        float d2vdy2 = warpParams[11];
        
        warpedX += 0.5f * d2udx2 * x * x + d2udxdy * x * y + 0.5f * d2udy2 * y * y;
        warpedY += 0.5f * d2vdx2 * x * x + d2vdxdy * x * y + 0.5f * d2vdy2 * y * y;
    }
}

// Kernel for computing ZNCC
__global__ void computeZNCCKernel(const unsigned char* refImage, size_t refPitch,
                                const unsigned char* defImage, size_t defPitch,
                                int subsetRadius, int refPointX, int refPointY,
                                float* warpParams, int numParams, float* result) {
    // Each thread block will compute part of the sums
    __shared__ float sumRef[32];
    __shared__ float sumDef[32];
    __shared__ float sumRefSq[32];
    __shared__ float sumDefSq[32];
    __shared__ float sumRefDef[32];
    __shared__ float count[32];
    
    const int tid = threadIdx.x;
    
    // Initialize shared memory
    if (tid < 32) {
        sumRef[tid] = 0.0f;
        sumDef[tid] = 0.0f;
        sumRefSq[tid] = 0.0f;
        sumDefSq[tid] = 0.0f;
        sumRefDef[tid] = 0.0f;
        count[tid] = 0.0f;
    }
    __syncthreads();
    
    // Compute subset size
    const int subsetDiameter = 2 * subsetRadius + 1;
    const int totalPixels = subsetDiameter * subsetDiameter;
    
    // Each thread processes multiple pixels
    const int pixelsPerThread = (totalPixels + blockDim.x - 1) / blockDim.x;
    const int start = tid * pixelsPerThread;
    const int end = min(start + pixelsPerThread, totalPixels);
    
    // Process assigned pixels
    for (int idx = start; idx < end; idx++) {
        int x = idx % subsetDiameter - subsetRadius;
        int y = idx / subsetDiameter - subsetRadius;
        
        // Reference subset coordinates
        int refPixelX = refPointX + x;
        int refPixelY = refPointY + y;
        
        // Check if within reference image bounds
        if (refPixelX >= 0 && refPixelX < blockDim.y && 
            refPixelY >= 0 && refPixelY < blockDim.z) {
            
            // Get reference intensity
            const unsigned char* refRow = (const unsigned char*)((const char*)refImage + refPixelY * refPitch);
            float refIntensity = static_cast<float>(refRow[refPixelX]);
            
            // Warp point to get corresponding point in deformed image
            float warpedX, warpedY;
            warpPointGPU(static_cast<float>(x), static_cast<float>(y), warpParams, numParams, warpedX, warpedY);
            
            float defImgX = refPointX + warpedX;
            float defImgY = refPointY + warpedY;
            
            // Check if within deformed image bounds
            if (defImgX >= 0 && defImgX < blockDim.y - 1 &&
                defImgY >= 0 && defImgY < blockDim.z - 1) {
                
                // Get deformed intensity (interpolated)
                float defIntensity = interpolateGPU(defImage, defPitch, blockDim.y, blockDim.z, defImgX, defImgY);
                
                // Update sums for ZNCC
                atomicAdd(&sumRef[tid % 32], refIntensity);
                atomicAdd(&sumDef[tid % 32], defIntensity);
                atomicAdd(&sumRefSq[tid % 32], refIntensity * refIntensity);
                atomicAdd(&sumDefSq[tid % 32], defIntensity * defIntensity);
                atomicAdd(&sumRefDef[tid % 32], refIntensity * defIntensity);
                atomicAdd(&count[tid % 32], 1.0f);
            }
        }
    }
    
    __syncthreads();
    
    // Reduce results from all threads
    if (tid == 0) {
        float totalRef = 0.0f;
        float totalDef = 0.0f;
        float totalRefSq = 0.0f;
        float totalDefSq = 0.0f;
        float totalRefDef = 0.0f;
        float totalCount = 0.0f;
        
        for (int i = 0; i < 32; ++i) {
            totalRef += sumRef[i];
            totalDef += sumDef[i];
            totalRefSq += sumRefSq[i];
            totalDefSq += sumDefSq[i];
            totalRefDef += sumRefDef[i];
            totalCount += count[i];
        }
        
        if (totalCount > 0) {
            float meanRef = totalRef / totalCount;
            float meanDef = totalDef / totalCount;
            float varRef = totalRefSq / totalCount - meanRef * meanRef;
            float varDef = totalDefSq / totalCount - meanDef * meanDef;
            float covar = totalRefDef / totalCount - meanRef * meanDef;
            
            if (varRef > 0 && varDef > 0) {
                // Return 1 - ZNCC to convert to minimization problem
                *result = 1.0f - (covar / sqrtf(varRef * varDef));
            } else {
                *result = 1000000.0f;  // Error case
            }
        } else {
            *result = 1000000.0f;  // Error case
        }
    }
}

// Kernel to warp and extract subset
__global__ void warpSubsetKernel(const unsigned char* defImage, size_t defPitch,
                               int width, int height,
                               int refPointX, int refPointY,
                               int subsetRadius,
                               float* warpParams, int numParams,
                               float* warpedSubset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x - subsetRadius;
    int y = blockIdx.y * blockDim.y + threadIdx.y - subsetRadius;
    
    if (abs(x) <= subsetRadius && abs(y) <= subsetRadius) {
        // Warp point coordinates
        float warpedX, warpedY;
        warpPointGPU(static_cast<float>(x), static_cast<float>(y), warpParams, numParams, warpedX, warpedY);
        
        float defImgX = refPointX + warpedX;
        float defImgY = refPointY + warpedY;
        
        // Calculate index in output warped subset
        int outputIdx = (y + subsetRadius) * (2 * subsetRadius + 1) + (x + subsetRadius);
        
        // Check if within deformed image bounds
        if (defImgX >= 0 && defImgX < width - 1 &&
            defImgY >= 0 && defImgY < height - 1) {
            // Get interpolated intensity
            warpedSubset[outputIdx] = interpolateGPU(defImage, defPitch, width, height, defImgX, defImgY);
        } else {
            warpedSubset[outputIdx] = 0.0f;
        }
    }
}

// Kernel for computing steepest descent images
__global__ void steepestDescentKernel(const unsigned char* refImage, size_t refPitch,
                                    int width, int height,
                                    int refPointX, int refPointY,
                                    int subsetRadius, int numParams,
                                    float* steepestDescentImages) {
    int x = blockIdx.x * blockDim.x + threadIdx.x - subsetRadius;
    int y = blockIdx.y * blockDim.y + threadIdx.y - subsetRadius;
    
    if (abs(x) <= subsetRadius && abs(y) <= subsetRadius) {
        int pixelX = refPointX + x;
        int pixelY = refPointY + y;
        
        // Check if within image bounds
        if (pixelX >= 1 && pixelX < width - 1 &&
            pixelY >= 1 && pixelY < height - 1) {
            
            // Get image gradients using central difference
            const unsigned char* row_prev = (const unsigned char*)((const char*)refImage + (pixelY-1) * refPitch);
            const unsigned char* row_curr = (const unsigned char*)((const char*)refImage + pixelY * refPitch);
            const unsigned char* row_next = (const unsigned char*)((const char*)refImage + (pixelY+1) * refPitch);
            
            float dx = (row_curr[pixelX+1] - row_curr[pixelX-1]) * 0.5f;
            float dy = (row_next[pixelX] - row_prev[pixelX]) * 0.5f;
            
            // Calculate output index
            int idx = (y + subsetRadius) * (2 * subsetRadius + 1) + (x + subsetRadius);
            int subsetSize = (2 * subsetRadius + 1) * (2 * subsetRadius + 1);
            
            // First order parameters
            steepestDescentImages[0 * subsetSize + idx] = dx;                  // du
            steepestDescentImages[1 * subsetSize + idx] = dy;                  // dv
            steepestDescentImages[2 * subsetSize + idx] = dx * x;              // du/dx
            steepestDescentImages[3 * subsetSize + idx] = dx * y;              // du/dy
            steepestDescentImages[4 * subsetSize + idx] = dy * x;              // dv/dx
            steepestDescentImages[5 * subsetSize + idx] = dy * y;              // dv/dy
            
            // Second order parameters (if applicable)
            if (numParams == 12) {
                steepestDescentImages[6 * subsetSize + idx] = dx * x * x * 0.5f;  // d²u/dx²
                steepestDescentImages[7 * subsetSize + idx] = dx * x * y;         // d²u/dxdy
                steepestDescentImages[8 * subsetSize + idx] = dx * y * y * 0.5f;  // d²u/dy²
                steepestDescentImages[9 * subsetSize + idx] = dy * x * x * 0.5f;  // d²v/dx²
                steepestDescentImages[10 * subsetSize + idx] = dy * x * y;        // d²v/dxdy
                steepestDescentImages[11 * subsetSize + idx] = dy * y * y * 0.5f; // d²v/dy²
            }
        }
    }
}

// Kernel for computing Hessian matrix
__global__ void computeHessianKernel(float* steepestDescentImages,
                                   int subsetDiameter,
                                   int numParams,
                                   float* hessian) {
    const int i = blockIdx.x;
    const int j = blockIdx.y;
    
    if (i < numParams && j < numParams) {
        const int subsetSize = subsetDiameter * subsetDiameter;
        const int numThreads = blockDim.x;
        const int tid = threadIdx.x;
        
        __shared__ float partialSums[256];  // Adjust based on max block size
        partialSums[tid] = 0.0f;
        
        // Each thread processes multiple pixels
        const int pixelsPerThread = (subsetSize + numThreads - 1) / numThreads;
        const int start = tid * pixelsPerThread;
        const int end = min(start + pixelsPerThread, subsetSize);
        
        // Compute partial dot product
        for (int idx = start; idx < end; idx++) {
            partialSums[tid] += steepestDescentImages[i * subsetSize + idx] * 
                               steepestDescentImages[j * subsetSize + idx];
        }
        
        __syncthreads();
        
        // Reduction in shared memory
        for (int stride = numThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialSums[tid] += partialSums[tid + stride];
            }
            __syncthreads();
        }
        
        // Write result to global memory
        if (tid == 0) {
            hessian[i * numParams + j] = partialSums[0];
        }
    }
}

// Kernel for computing error vector
__global__ void errorVectorKernel(float* refSubset,
                                float* warpedSubset,
                                float* steepestDescentImages,
                                int subsetDiameter,
                                int numParams,
                                float* errorVector) {
    const int paramIdx = blockIdx.x;
    
    if (paramIdx < numParams) {
        const int subsetSize = subsetDiameter * subsetDiameter;
        const int numThreads = blockDim.x;
        const int tid = threadIdx.x;
        
        __shared__ float partialSum[256];  // Adjust based on max block size
        partialSum[tid] = 0.0f;
        
        // Each thread processes multiple pixels
        const int pixelsPerThread = (subsetSize + numThreads - 1) / numThreads;
        const int start = tid * pixelsPerThread;
        const int end = min(start + pixelsPerThread, subsetSize);
        
        // Compute partial sum for error vector
        for (int idx = start; idx < end; idx++) {
            float error = refSubset[idx] - warpedSubset[idx];
            partialSum[tid] += error * steepestDescentImages[paramIdx * subsetSize + idx];
        }
        
        __syncthreads();
        
        // Reduction in shared memory
        for (int stride = numThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialSum[tid] += partialSum[tid + stride];
            }
            __syncthreads();
        }
        
        // Write result to global memory
        if (tid == 0) {
            errorVector[paramIdx] = partialSum[0];
        }
    }
}

// Batch version of interpolation kernel for initial guess search
__global__ void interpolateKernel(const unsigned char* image, size_t pitch,
                                int width, int height,
                                float* points, int numPoints,
                                float* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numPoints) {
        float x = points[idx * 2];
        float y = points[idx * 2 + 1];
        
        results[idx] = interpolateGPU(image, pitch, width, height, x, y);
    }
}

} // namespace RGDIC_CUDA