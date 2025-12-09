// ------------------------------------------------------------------------------------------------//
//	
//	header file containing all CUDA kernels used by the Matrix class
// ------------------------------------------------------------------------------------------------//
#pragma once
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <iostream>
#ifndef CUTRIX_H
#define CUTRIX_H
template <typename U>
__global__ void matmult_K(U* A, U* B, U* C, int width, int rowC, int colC);
template <typename U>
void matmult(U* A, const U* B, U* C, int arows, int acols, int brows, int bcols);
template <typename U>
__global__ void matadd_K(U* A, U* B, U* C, int cols, int rows);
template <typename U>
void matop(int a, U* A, const U* B, U* C, int arows, int acols);
template <typename U>
__global__ void matsub_K(U* A, U* B, U* C, int cols, int rows);
template <typename U>
__global__ void mat_Hadam_K(U* A, U* B, U* C, int cols, int rows);
template <typename U>
__global__ void mat_broad_sum_col_K(U* A, U* B, U* C, int cols, int rows);
template <typename U>
__global__ void mat_broad_sum_row_K(U* A, U* B, U* C, int cols, int rows);
template <typename U>
__global__ void mat_broad_sub_col_K(U* A, U* B, U* C, int cols, int rows);
template <typename U>
__global__ void mat_broad_sub_row_K(U* A, U* B, U* C, int cols, int rows);
template <typename U>
__global__ void mat_broad_mult_col_K(U* A, U* B, U* C, int cols, int rows);
template <typename U>
__global__ void mat_broad_mult_row_K(U* A, U* B, U* C, int cols, int rows);
template <typename U>
void matscal(int a, U b, const U* A, U* C, int arows, int acols);
template <typename U>
__global__ void mat_scal_mult_K(U a, U* A, U* C, int cols, int rows);
template <typename U>
__global__ void mat_scal_div_K(U a, U* A, U* C, int cols, int rows);
template <typename U>
__global__ void mat_sum_row_K(U* A, U* C, int cols, int rows);
template <typename U>
__global__ void mat_sum_col_K(U* A, U* C, int cols, int rows);
template <typename U>
void mat_indx(U* A, U* C, int arows, int acols);
template <typename U>
__global__ void mat_transpose_K(U* A, U* C, int cols, int rows);

template <typename U>
__global__ void matmult_K(U* A, U* B, U* C, int width, int rowC, int colC) {//CUDA kernel for matrix multiplication

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rowC && col < colC) {
		U sum = 0;
		for (int i = 0; i < width; i++) {
			sum += A[row * width + i] * B[i * colC + col];
		}
		C[row * colC + col] = sum;
	}
}
template <typename U>
__global__ void matadd_K(U* A, U* B, U* C, int cols, int rows) {//CUDA kernel for matrix addition

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
	}
}
template <typename U>
__global__ void matsub_K(U* A, U* B, U* C, int cols, int rows) {//CUDA kernel for matrix subtraction

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] - B[row * cols + col];
	}
}
template <typename U>
__global__ void mat_Hadam_K(U* A, U* B, U* C, int cols, int rows) {//CUDA kernel for Hadamard product

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] * B[row * cols + col];
	}
}
template <typename U>
__global__ void mat_broad_sum_col_K(U* A, U* B, U* C, int cols, int rows) {//CUDA kernel for broadcasting sum along columns

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] + B[row];
	}
}
template <typename U>
__global__ void mat_broad_sum_row_K(U* A, U* B, U* C, int cols, int rows) {//CUDA kernel for broadcasting sum along rows

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] + B[col];
	}
}
template <typename U>
__global__ void mat_broad_sub_col_K(U* A, U* B, U* C, int cols, int rows) {//CUDA kernel for broadcasting subtraction along columns

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] - B[row];
	}
}
template <typename U>
__global__ void mat_broad_sub_row_K(U* A, U* B, U* C, int cols, int rows) {//CUDA kernel for broadcasting subtraction along rows

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] - B[col];
	}
}
template <typename U>
__global__ void mat_broad_mult_col_K(U* A, U* B, U* C, int cols, int rows) {//CUDA kernel for broadcasting multiplication along columns

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] * B[row];
	}
}
template <typename U>
__global__ void mat_broad_mult_row_K(U* A, U* B, U* C, int cols, int rows) {//CUDA kernel for broadcasting multiplication along rows

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] * B[col];
	}
}
template <typename U>
__global__ void mat_div_K(U* A, U* B, U* C, int cols, int rows) {//CUDA kernel for element wise division

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] * B[row * cols + col];
	}
}
template <typename U>
__global__ void mat_broad_div_col_K(U* A, U* B, U* C, int cols, int rows) {//CUDA kernel for broadcasting division along columns

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] * B[row];
	}
}
template <typename U>
__global__ void mat_broad_div_row_K(U* A, U* B, U* C, int cols, int rows) {//CUDA kernel for broadcasting division along rows

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] * B[col];
	}
}
template <typename U>
__global__ void mat_sum_col_K(U* A, U* C, int cols, int rows) {//CUDA kernel for matrix sum along columns

	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int s = 1; s < rows; s *= 2) {
		unsigned int indx = row * s * 2;
		if (indx + s < rows && col < cols){
			A[indx*cols + col] += A[(indx+s)*cols + col];
		}
	}
	if (row == 0 && col < cols) {
		C[col]=A[col];
	}

}
template <typename U>
__global__ void mat_sum_row_K(U* A, U* C, int cols, int rows) {//CUDA kernel for matrix sum along rows

	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	for (int s = 1; s < cols; s *= 2) {
		unsigned int indx = col * s * 2;
		if (indx + s < cols && row < rows){
			A[row*cols + indx] += A[row*cols + indx + s];
		}

	}
	if (col == 0 && row < rows) {
		C[row]=A[row*cols];
	}
}

template <typename U>
__global__ void mat_transpose_K(U* A, U* C, int cols, int rows) {
	// Calculate row and column index in the output matrix
	constexpr unsigned int TILE_WIDTH = 16;
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

	// Allocate shared memory for the tile
	__shared__ U tile[TILE_WIDTH][TILE_WIDTH + 1]; // +1 to avoid bank conflicts

	// Load data into shared memory
	if (row < rows && col < cols) {
		tile[threadIdx.y][threadIdx.x] = A[row * cols + col];
	}

	// Synchronize to ensure all threads have loaded their data
	__syncthreads();

	// Calculate transposed indices
	int transposed_row = blockIdx.x * TILE_WIDTH + threadIdx.y;
	int transposed_col = blockIdx.y * TILE_WIDTH + threadIdx.x;

	// Write the transposed data to the output matrix
	if (transposed_row < cols && transposed_col < rows) {
		C[transposed_row * rows + transposed_col] = tile[threadIdx.x][threadIdx.y];
	}
}
template <typename U>
__global__ void mat_scal_mult_K(U a, U* A, U* C, int cols, int rows) {//CUDA kernel for scalar multiplication

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] * a;
	}
}
template <typename U>
__global__ void mat_scal_div_K(U a, U* A, U* C, int cols, int rows) {//CUDA kernel for scalar division

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] / a;
	}
}
template <typename U>
void matmult(U* A, const U* B, U* C, int arows, int acols, int brows, int bcols) {//CUDA wrapper function for matrix multiplication
	U* c_A, * c_B, * c_C;
	cudaMallocManaged(&c_A, arows * acols * sizeof(U));
	cudaMallocManaged(&c_B, brows * bcols * sizeof(U));
	cudaMallocManaged(&c_C, arows * bcols * sizeof(U));

	cudaMemcpy(c_A, A, arows * acols * sizeof(U), cudaMemcpyHostToDevice);
	cudaMemcpy(c_B, B, brows * bcols * sizeof(U), cudaMemcpyHostToDevice);

	dim3 grid(bcols / 32 + 1, arows / 32 + 1, 1);
	dim3 block(32, 32, 1);
	matmult_K << <grid, block >> > (c_A, c_B, c_C, acols, arows, bcols);
	cudaMemcpy(C, c_C, arows * bcols * sizeof(U), cudaMemcpyDeviceToHost);
	cudaFree(c_A);
	cudaFree(c_B);
	cudaFree(c_C);

}
template <typename U>
void matop(int a, U* A, const U* B, U* C, int arows, int acols) {//CUDA wrapper function for matrix operations
	//operations selection:
	//0- classic addition
	//1 - classic subtraction
	//2 - Hadamard product
	//3 - broadcast addition (columns)
	//4 - broadcast addition (rows)
	//5 - broadcast subtraction (columns)
	//6 - broadcast subtraction (rows)
	//7 - broadcast multiplication (columns)
	//8 - broadcast multiplication (rows)
	//9 - element wise division
	//10 - broadcasting element wise division (columns)
	//11 - broadcasting element wise division (rows)
	//12 - sum matrix along columns
	//13 - sum matrix along rows
	if (a < 0 || a > 11) {
		std::cout << "input argument for CUDA matrix op wrapper out of range, should be 0-11, is "<<a<<std::endl;
		exit(0);
	}
	U* c_A, * c_B, * c_C;
	cudaMallocManaged(&c_A, arows * acols * sizeof(U));

	cudaMallocManaged(&c_C, arows * acols * sizeof(U));

	cudaMemcpy(c_A, A, arows * acols * sizeof(U), cudaMemcpyHostToDevice);

	dim3 grid(acols / 32 + 1, arows / 32 + 1, 1);
	dim3 block(32, 32, 1);
	switch (a){// switch to invoke proper CUDA kernel
	case 0:
		cudaMallocManaged(&c_B, arows * acols * sizeof(U));
		cudaMemcpy(c_B, B, arows * acols * sizeof(U), cudaMemcpyHostToDevice);
		matadd_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 1:
		cudaMallocManaged(&c_B, arows * acols * sizeof(U));
		cudaMemcpy(c_B, B, arows * acols * sizeof(U), cudaMemcpyHostToDevice);
		matsub_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 2:
		cudaMallocManaged(&c_B, arows * acols * sizeof(U));
		cudaMemcpy(c_B, B, arows * acols * sizeof(U), cudaMemcpyHostToDevice);
		mat_Hadam_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 3:
		cudaMallocManaged(&c_B, arows* sizeof(U));
		cudaMemcpy(c_B, B, arows * sizeof(U), cudaMemcpyHostToDevice);
		mat_broad_sum_col_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 4:
		cudaMallocManaged(&c_B, acols * sizeof(U));
		cudaMemcpy(c_B, B, acols * sizeof(U), cudaMemcpyHostToDevice);
		mat_broad_sum_row_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 5:
		cudaMallocManaged(&c_B, arows * sizeof(U));
		cudaMemcpy(c_B, B, arows * sizeof(U), cudaMemcpyHostToDevice);
		mat_broad_sub_col_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 6:
		cudaMallocManaged(&c_B, acols * sizeof(U));
		cudaMemcpy(c_B, B, acols * sizeof(U), cudaMemcpyHostToDevice);
		mat_broad_sub_row_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 7:
		cudaMallocManaged(&c_B, arows * sizeof(U));
		cudaMemcpy(c_B, B, arows * sizeof(U), cudaMemcpyHostToDevice);
		mat_broad_mult_col_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 8:
		cudaMallocManaged(&c_B, acols * sizeof(U));
		cudaMemcpy(c_B, B, acols * sizeof(U), cudaMemcpyHostToDevice);
		mat_broad_mult_row_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 9:
		cudaMallocManaged(&c_B, arows * acols * sizeof(U));
		cudaMemcpy(c_B, B, arows * acols * sizeof(U), cudaMemcpyHostToDevice);
		mat_div_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 10:
		cudaMallocManaged(&c_B, arows * sizeof(U));
		cudaMemcpy(c_B, B, arows * sizeof(U), cudaMemcpyHostToDevice);
		mat_broad_div_col_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 11:
		cudaMallocManaged(&c_B, acols * sizeof(U));
		cudaMemcpy(c_B, B, acols * sizeof(U), cudaMemcpyHostToDevice);
		mat_broad_div_row_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	default:
		std::cout << "Switch error.\n";
		exit(0);
	}

	cudaMemcpy(C, c_C, arows * acols * sizeof(U), cudaMemcpyDeviceToHost);
	cudaFree(c_A);
	cudaFree(c_B);
	cudaFree(c_C);
}
template <typename U>
void matscal(int a, U b, const U* A, U* C, int arows, int acols) {//CUDA wrapper function for scalar operations
	//the numeral b is the scalar by which to multiply the matrix elements

	if (a < 0 || a > 3) {
		std::cout << "input argument for CUDA scalar wrapper out of range, should be 0-8, is " << a << std::endl;
		exit(0);
	}

	U* c_A, * c_C;
	cudaMallocManaged(&c_A, arows * acols * sizeof(U));

	cudaMemcpy(c_A, A, arows * acols * sizeof(U), cudaMemcpyHostToDevice);

	dim3 grid(acols / 32 + 1, arows / 32 + 1, 1);
	dim3 block(32, 32, 1);

	switch (a) // switch to invoke proper CUDA kernel
	{
	case 0:
		cudaMallocManaged(&c_C, arows * acols * sizeof(U));
		mat_scal_mult_K << <grid, block >> > (b, c_A, c_C, acols, arows);
		cudaMemcpy(C, c_C, arows * acols * sizeof(U), cudaMemcpyDeviceToHost);
		break;
	case 1:
		cudaMallocManaged(&c_C, arows * acols * sizeof(U));
		mat_scal_div_K << <grid, block >> > (b, c_A, c_C, acols, arows);
		cudaMemcpy(C, c_C, arows * acols * sizeof(U), cudaMemcpyDeviceToHost);
		break;
	case 2:
		cudaMallocManaged(&c_C, acols * sizeof(U));
		mat_sum_col_K << <grid, block >> > (c_A, c_C, acols, arows);
		cudaMemcpy(C, c_C, acols * sizeof(U), cudaMemcpyDeviceToHost);
		break;
	case 3:
		cudaMallocManaged(&c_C, arows * sizeof(U));
		mat_sum_row_K << <grid, block >> > (c_A, c_C, acols, arows);
		cudaMemcpy(C, c_C, arows * sizeof(U), cudaMemcpyDeviceToHost);
		break;
	default:
		std::cout << "Switch error in matscal().\n";
		exit(0);
	}
	// check for error
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));

	}
	cudaFree(c_A);
	cudaFree(c_C);
}
template <typename U>
void mat_indx(U* A, U* C, int arows, int acols) {//CUDA wrapper function for scalar operations
	//only used to call the transpose kernel for now
	U* c_A, * c_C;
	constexpr unsigned int WIDTH = 16;
	cudaMallocManaged(&c_A, arows * acols * sizeof(U));
	cudaMallocManaged(&c_C, arows * acols * sizeof(U));
	cudaMemcpy(c_A, A, arows * acols * sizeof(U), cudaMemcpyHostToDevice);
	dim3 grid((acols+WIDTH-1) / WIDTH, (arows+WIDTH-1) / WIDTH, 1);
	dim3 block(WIDTH, WIDTH, 1);
	mat_transpose_K << <grid, block >> > (c_A, c_C, acols, arows);
	cudaMemcpy(C, c_C, arows * acols * sizeof(U), cudaMemcpyDeviceToHost);

	// check for error
	if(const cudaError_t error = cudaGetLastError(); error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));

	}
	cudaFree(c_A);
	cudaFree(c_C);
}


#endif
