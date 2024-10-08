#include <device_launch_parameters.h>

__global__ void matmult_K(double* A, double* B, double* C, int width, int rowC, int colC);
void matmult(double* A, const double* B, double* C, int arows, int acols, int brows, int bcols);
__global__ void matadd_K(double* A, double* B, double* C, int cols, int rows);
void matop(int a, double* A, const double* B, double* C, int arows, int acols);
__global__ void matsub_K(double* A, double* B, double* C, int cols, int rows);
__global__ void mat_Hadam_K(double* A, double* B, double* C, int cols, int rows);
__global__ void mat_broad_sum_col_K(double* A, double* B, double* C, int cols, int rows);
__global__ void mat_broad_sum_row_K(double* A, double* B, double* C, int cols, int rows);
__global__ void mat_broad_sub_col_K(double* A, double* B, double* C, int cols, int rows);
__global__ void mat_broad_sub_row_K(double* A, double* B, double* C, int cols, int rows);
__global__ void mat_broad_mult_col_K(double* A, double* B, double* C, int cols, int rows);
__global__ void mat_broad_mult_row_K(double* A, double* B, double* C, int cols, int rows);
void matscal(int a, double b, const double* A, double* C, int arows, int acols);
__global__ void mat_scal_mult_K(int a, double* A, double* C, int cols, int rows);
__global__ void mat_scal_div_K(int a, double* A, double* C, int cols, int rows);
__global__ void mat_sum_row_K(double* A, double* C, int cols, int rows);
__global__ void mat_sum_col_K(double* A, double* C, int cols, int rows);