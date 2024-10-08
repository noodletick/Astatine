
#include "cuda_runtime.h"
#include "CUTRIX.cuh"
#include <device_launch_parameters.h>
#include <iostream>

__global__ void matmult_K(double* A, double* B, double* C, int width, int rowC, int colC) {//CUDA kernel for matrix multiplication

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rowC && col < colC) {
		double sum = 0;
		for (int i = 0; i < width; i++) {
			sum += A[row * width + i] * B[i * colC + col];
		}
		C[row * colC + col] = sum;
	}
}

__global__ void matadd_K(double* A, double* B, double* C, int cols, int rows) {//CUDA kernel for matrix addition

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		
		C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
	}
}

__global__ void matsub_K(double* A, double* B, double* C, int cols, int rows) {//CUDA kernel for matrix substraction

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] - B[row * cols + col];
	}
}

__global__ void mat_Hadam_K(double* A, double* B, double* C, int cols, int rows) {//CUDA kernel for Hadamard product

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] * B[row * cols + col];
	}
}

__global__ void mat_broad_sum_col_K(double* A, double* B, double* C, int cols, int rows) {//CUDA kernel for broadcasting sum along columns

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] + B[row];
	}
}

__global__ void mat_broad_sum_row_K(double* A, double* B, double* C, int cols, int rows) {//CUDA kernel for broadcasting sum along rows

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] + B[col];
	}
}

__global__ void mat_broad_sub_col_K(double* A, double* B, double* C, int cols, int rows) {//CUDA kernel for broadcasting substraction along columns

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] - B[row];
	}
}

__global__ void mat_broad_sub_row_K(double* A, double* B, double* C, int cols, int rows) {//CUDA kernel for broadcasting substraction along rows

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] - B[col];
	}
}

__global__ void mat_broad_mult_col_K(double* A, double* B, double* C, int cols, int rows) {//CUDA kernel for broadcasting multiplication along columns

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] * B[row];
	}
}

__global__ void mat_broad_mult_row_K(double* A, double* B, double* C, int cols, int rows) {//CUDA kernel for broadcasting multiplication along rows

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] * B[col];
	}
}
__global__ void mat_div_K(double* A, double* B, double* C, int cols, int rows) {//CUDA kernel for element wise division

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] * B[row * cols + col];
	}
}

__global__ void mat_broad_div_col_K(double* A, double* B, double* C, int cols, int rows) {//CUDA kernel for broadcasting division along columns

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] * B[row];
	}
}

__global__ void mat_broad_div_row_K(double* A, double* B, double* C, int cols, int rows) {//CUDA kernel for broadcasting division along rows

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] * B[col];
	}
}

__global__ void mat_sum_col_K(double* A, double* C, int cols, int rows) {//CUDA kernel for matrix sum along columns

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double temp_sum=0;
	if (row < rows && col < cols) {
		for (int i = 0; i < rows; i++) {
			temp_sum += A[i * cols + col];
		}
		C[col] = temp_sum;
	}
}

__global__ void mat_sum_row_K(double* A, double* C, int cols, int rows) {//CUDA kernel for matrix sum along rows

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double temp_sum = 0;
	if (row < rows && col < cols) {
		for (int i = 0; i < cols; i++) {
			temp_sum += A[row * cols + i];
		}
		C[row] = temp_sum;
	}
}

__global__ void mat_scal_mult_K(double a, double* A, double* C, int cols, int rows) {//CUDA kernel for scalar multiplication

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] * a;
	}
}

__global__ void mat_scal_div_K(double a, double* A, double* C, int cols, int rows) {//CUDA kernel for scalar division

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {

		C[row * cols + col] = A[row * cols + col] / a;
	}
}

void matmult(double* A, const double* B, double* C, int arows, int acols, int brows, int bcols) {//CUDA wrapper function for matrix multiplication
	double* c_A, * c_B, * c_C;
	cudaMallocManaged(&c_A, arows * acols * sizeof(double));
	cudaMallocManaged(&c_B, brows * bcols * sizeof(double));
	cudaMallocManaged(&c_C, arows * bcols * sizeof(double));

	cudaMemcpy(c_A, A, arows * acols * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(c_B, B, brows * bcols * sizeof(double), cudaMemcpyHostToDevice);

	dim3 grid(bcols / 32 + 1, arows / 32 + 1, 1);
	dim3 block(32, 32, 1);
	matmult_K << <grid, block >> > (c_A, c_B, c_C, acols, arows, bcols);
	cudaMemcpy(C, c_C, arows * bcols * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(c_A);
	cudaFree(c_B);
	cudaFree(c_C);

}

void matop(int a, double* A, const double* B, double* C, int arows, int acols) {//CUDA wrapper function for matrix operations
	//opeartions selection:
	//0- classic addition
	//1 - classic substraction
	//2 - Hadamar product
	//3 - broadcast addition (columns)
	//4 - broadcast addition (rows)
	//5 - broadcast substraction (columns)
	//6 - broadcast substraction (rows)
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
	double* c_A, * c_B, * c_C;
	cudaMallocManaged(&c_A, arows * acols * sizeof(double));
	
	cudaMallocManaged(&c_C, arows * acols * sizeof(double));

	cudaMemcpy(c_A, A, arows * acols * sizeof(double), cudaMemcpyHostToDevice);
	
	dim3 grid(acols / 32 + 1, arows / 32 + 1, 1);
	dim3 block(32, 32, 1);
	switch (a){// switch to invoke proper CUDA kernel
	case 0:
		cudaMallocManaged(&c_B, arows * acols * sizeof(double));
		cudaMemcpy(c_B, B, arows * acols * sizeof(double), cudaMemcpyHostToDevice);
		matadd_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 1:
		cudaMallocManaged(&c_B, arows * acols * sizeof(double));
		cudaMemcpy(c_B, B, arows * acols * sizeof(double), cudaMemcpyHostToDevice);
		matsub_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 2:
		cudaMallocManaged(&c_B, arows * acols * sizeof(double));
		cudaMemcpy(c_B, B, arows * acols * sizeof(double), cudaMemcpyHostToDevice);
		mat_Hadam_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 3:
		cudaMallocManaged(&c_B, arows* sizeof(double));
		cudaMemcpy(c_B, B, arows * sizeof(double), cudaMemcpyHostToDevice);
		mat_broad_sum_col_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 4:
		cudaMallocManaged(&c_B, acols * sizeof(double));
		cudaMemcpy(c_B, B, acols * sizeof(double), cudaMemcpyHostToDevice);
		mat_broad_sum_row_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 5:
		cudaMallocManaged(&c_B, arows * sizeof(double));
		cudaMemcpy(c_B, B, arows * sizeof(double), cudaMemcpyHostToDevice);
		mat_broad_sub_col_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 6:
		cudaMallocManaged(&c_B, acols * sizeof(double));
		cudaMemcpy(c_B, B, acols * sizeof(double), cudaMemcpyHostToDevice);
		mat_broad_sub_row_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 7:
		cudaMallocManaged(&c_B, arows * sizeof(double));
		cudaMemcpy(c_B, B, arows * sizeof(double), cudaMemcpyHostToDevice);
		mat_broad_mult_col_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 8:
		cudaMallocManaged(&c_B, acols * sizeof(double));
		cudaMemcpy(c_B, B, acols * sizeof(double), cudaMemcpyHostToDevice);
		mat_broad_mult_row_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 9:
		cudaMallocManaged(&c_B, arows * acols * sizeof(double));
		cudaMemcpy(c_B, B, arows * acols * sizeof(double), cudaMemcpyHostToDevice);
		mat_div_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 10:
		cudaMallocManaged(&c_B, arows * sizeof(double));
		cudaMemcpy(c_B, B, arows * sizeof(double), cudaMemcpyHostToDevice);
		mat_broad_div_col_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	case 11:
		cudaMallocManaged(&c_B, acols * sizeof(double));
		cudaMemcpy(c_B, B, acols * sizeof(double), cudaMemcpyHostToDevice);
		mat_broad_div_row_K << <grid, block >> > (c_A, c_B, c_C, acols, arows);
		break;
	default:
		std::cout << "Switch error.\n";
		exit(0);
	}

	cudaMemcpy(C, c_C, arows * acols * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(c_A);
	cudaFree(c_B);
	cudaFree(c_C);
}
void matscal(int a, double b, const double* A, double* C, int arows, int acols) {//CUDA wrapper function for scalar operations
	//the integer b is the scalar by which to multiply the matrix elements

	if (a < 0 || a > 3) {
		std::cout << "input argument for CUDA scalar wrapper out of range, should be 0-8, is " << a << std::endl;
		exit(0);
	}

	double* c_A, * c_C;
	cudaMallocManaged(&c_A, arows * acols * sizeof(double));

	cudaMemcpy(c_A, A, arows * acols * sizeof(double), cudaMemcpyHostToDevice);

	dim3 grid(acols / 32 + 1, arows / 32 + 1, 1);
	dim3 block(32, 32, 1);

	switch (a) // switch to invoke proper CUDA kernel
	{
	case 0:
		cudaMallocManaged(&c_C, arows * acols * sizeof(double));
		mat_scal_mult_K << <grid, block >> > (b, c_A, c_C, acols, arows);
		cudaDeviceSynchronize();
		cudaMemcpy(C, c_C, arows * acols * sizeof(double), cudaMemcpyDeviceToHost);
		break;
	case 1:
		cudaMallocManaged(&c_C, arows * acols * sizeof(double));
		mat_scal_div_K << <grid, block >> > (b, c_A, c_C, acols, arows);
		cudaDeviceSynchronize();
		cudaMemcpy(C, c_C, arows * acols * sizeof(double), cudaMemcpyDeviceToHost);
		break;
	case 2:
		cudaMallocManaged(&c_C, acols * sizeof(double));
		mat_sum_col_K << <grid, block >> > (c_A, c_C, acols, arows);
		cudaDeviceSynchronize();
		cudaMemcpy(C, c_C, acols * sizeof(double), cudaMemcpyDeviceToHost);
		break;
	case 3:
		cudaMallocManaged(&c_C, arows * sizeof(double));
		mat_sum_row_K << <grid, block >> > (c_A, c_C, acols, arows);
		cudaDeviceSynchronize();
		cudaMemcpy(C, c_C, arows * sizeof(double), cudaMemcpyDeviceToHost);
		break;
	}

	cudaMemcpy(C, c_C, arows * acols * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(c_A);
	cudaFree(c_C);
}