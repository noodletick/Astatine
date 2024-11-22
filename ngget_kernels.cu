//
// Created by matthewjouffray on 11/11/24.
//
#include "NUGGET.cuh"
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

 __global__ void ReLu_kernel(float* A, float* B, int cols, int rows) {

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        if (A[row * cols + col] > 0) {
            B[row * cols + col] = A[row * cols + col];
        }
        else {
            B[row * cols + col] = 0;
        }

    }
}

__global__ void ReLuPr_kernel(float* A, float* B, int cols, int rows) {

     unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
     unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
     if (row < rows && col < cols) {
         if (A[row * cols + col] > 0) {
             B[row * cols + col] = 1;
         }
         else {
             B[row * cols + col] = 0;
         }

     }
 }

__global__ void Leaky_ReLu_kernel(float* A, float* B, float c, int cols, int rows) {

     unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
     unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
     if (row < rows && col < cols) {
         if (A[row * cols + col] > 0) {
             B[row * cols + col] = A[row * cols + col];
         }
         else {
             B[row * cols + col] = A[row * cols + col]*c;
         }

     }
 }

__global__ void Leaky_ReLuPr_kernel(float* A, float* B, float c, int cols, int rows) {

     unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
     unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
     if (row < rows && col < cols) {
         if (A[row * cols + col] > 0) {
             B[row * cols + col] = 1;
         }
         else {
             B[row * cols + col] = c;
         }

     }
 }

__global__ void sigmoid_kernel(float* A, float* B, int cols, int rows) {

     unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
     unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
     if (row < rows && col < cols) {
         B[row * cols + col] = sigm(A[row * cols + col]);

     }
 }

__global__ void sigmoidPr_kernel(float* A, float* B, int cols, int rows) {

     unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
     unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
     if (row < rows && col < cols) {
         float tempsig = sigm(A[row * cols + col]);
         B[row * cols + col] = tempsig*(1- tempsig);

     }
 }
__device__ float sigm(float x) {
     if (x >= 0) {
         float ex = expf(-x);
         return 1.0f / (1.0f + ex);
     }
     else {
         float ex = expf(x);
         return ex / (1.0f + ex);
     }
 }