#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <iomanip>
#include "./CUTRIX.cuh"


#ifndef MAT_H
#define MAT_H

class mat {

private:
	unsigned int n, m;
	std::vector<double> matrix;

public:
	// -- constructors --
	mat(std::string, unsigned int);
	mat(std::string, unsigned int, unsigned int);
	mat(std::string, double, double, unsigned int, unsigned int);
	mat(double, unsigned int, unsigned int);
	mat(std::vector<std::vector<double>>&);
	mat(std::vector<double>&, unsigned int, unsigned int);
	mat(std::vector<std::vector<unsigned int>>&);
	mat(std::vector<std::vector<int>>&);
	mat(std::vector<std::vector<float>>&);
	mat(std::vector<double>&);
	mat(std::vector<unsigned int>&);
	mat(std::vector<int>&);
	mat(std::vector<float>&);
	mat();
	// -- operators --
	// matrix operations
	mat operator+(const mat&); // addition
	mat operator-(const mat&); // substraction
	mat operator*(const mat&); // multiplication
	mat operator^(const mat&); // broadcasting/Hadamard product
	mat operator/(const mat&); // dividing matrix by 1D vector by row or column 
	// matrix index
	double& operator()(const unsigned int&, const unsigned int&);
	// matrix comparison
	bool operator==(const mat&);
	// matrix assignment
	void operator=(const mat&);
	// scalar operations
	friend mat operator*(const double a,const mat& A);
	friend mat operator*(const mat& A, const double a);
	mat operator/(const double);
	// -- utilities -- 
	mat T(); // transpose
	void print(); // print matrix
	unsigned int rows(); // returns number of rows
	unsigned int cols(); // returns number of column
	mat sum(std::string); // sum along axis
	double sum(); // sum all cells
	double max(); // returns the largest element of the matrix
	double min(); // returns the smallest element of the matrix
	mat xp(); // returns e^x of every matrix elements
};

mat::mat(std::vector<std::vector<double>>& M) { // contructor which accepts 2D std::vectors for 2D matrix
	m = M.size();
	n = M[0].size();
	this->matrix.resize(m * n);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			this->matrix[i * n + j] = M[i][j];
			
		}
	}
}


mat::mat(std::vector<double>& A, unsigned int b, unsigned int c) { // contructor which accepts a 1D std::vector for 2D matrix

	this->m = b;
	this->n = c;
	this->matrix = A;

}

mat::mat(std::string arg, unsigned int M, unsigned int N) { // contructor for initializing matrix as zeros or identity matrix
	// read the argument
	std::vector<double> vec(M * N, 0);
	if (arg == "zeros") { // null matrix
		
		matrix = vec;
		m = M;
		n = N;
	}
	else if (arg == "I") { // identity matrix
		if (N != M) {
			std::cout << "Cannot generate identity rectangular matrix, m must be equal to n." << std::endl;
			vec.clear();
			exit(0);
		}
		m = M;
		n = M;
		
		for (unsigned int i = 0; i < m; i++) {
			for (unsigned int j = 0; j < n; j++) {
				if (i == j) {
					vec[i*n+j] = 1;
				}

			}
		}

		matrix = vec;
	}
	else { // error
		std::cout << "invalid matrix constructor argument" << std::endl;
		vec.clear();
		exit(0);
	}


}

mat::mat(std::string arg, double a, double b, unsigned int M, unsigned int N) { // constructor to intialize matrix with random numbers
	// read the argument

	if (arg == "rand") { 
		std::vector<double> vec(M*N, 0);

		m = M;
		n = N;

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(a, b);

		for (unsigned int i = 0; i < m; i++) {
			for (unsigned int j = 0; j < n; j++) {
				vec[i*n+j] = dis(gen);
			}
		}

		matrix = vec;
	}

	else { // error
		std::cout << "invalid matrix constructor arguments" << std::endl;
		exit(0);
	}


}

mat::mat(double a, unsigned int M, unsigned int N) { // constructor to initialize all values as 'a' for given matrix size

	std::vector<double> vec(M * N, a);

	m = M;
	n = N;

	matrix = vec;

}

mat::mat(std::vector<std::vector<unsigned int>>& M) { // contructor which accepts std::vector<unsigned int>
	m = M.size();
	n = M[0].size();
	matrix.resize(m*n);
	for (unsigned int i = 0; i < m; i++) {
		for (unsigned int j = 0; j < n; j++) {
			matrix[i*n+j] = double(M[i][j]);
		}
	}

}

mat::mat(std::vector<std::vector<int>>& M) { // contructor which accepts std::vector<int>
	m = M.size();
	n = M[0].size();
	matrix.resize(m * n);
	for (unsigned int i = 0; i < m; i++) {
		for (unsigned int j = 0; j < n; j++) {
			matrix[i * n + j] = double(M[i][j]);
		}
	}

}

mat::mat(std::vector<std::vector<float>>& M) { // contructor which accepts std::vector<float>
	m = M.size();
	n = M[0].size();
	matrix.resize(m * n);
	for (unsigned int i = 0; i < m; i++) {
		for (unsigned int j = 0; j < n; j++) {
			matrix[i * n + j] = double(M[i][j]);
		}
	}

}

mat::mat(std::vector<float>& M) { // contructor which accepts 1D std::vector<float>
	m = M.size();
	n = 1;
	matrix.resize(m);
	for (unsigned int i = 0; i < m; i++) {

		matrix[i] = double(M[i]);

	}

}

mat::mat(std::vector<int>& M) { // contructor which accepts 1D std::vector<float>
	m = M.size();
	n = 1;
	matrix.resize(m);
	for (unsigned int i = 0; i < m; i++) {

		matrix[i] = double(M[i]);

	}

}

mat::mat(std::vector<unsigned int>& M) { // contructor which accepts 1D std::vector<float>
	m = M.size();
	n = 1;
	matrix.resize(m);
	for (unsigned int i = 0; i < m; i++) {

		matrix[i] = double(M[i]);

	}

}

mat::mat(std::vector<double>& M) { // contructor which accepts 1D std::vector<float>
	m = M.size();
	n = 1;
	matrix.resize(m);
	for (unsigned int i = 0; i < m; i++) {

		matrix[i] = double(M[i]);

	}

}

mat::mat() { //default constructor

	std::vector<double> vec;
	matrix = vec;
	m = 0;
	n = 0;

}


double& mat::operator()(const unsigned int& i, const unsigned int& j) {
	return this->matrix[i*this->n+j];
}

void mat::operator=(const mat& B) { // overload of assignment operator
	this->m = B.m;
	this->n = B.n;
	this->matrix = B.matrix;
}

mat mat::operator*(const mat& B) { // matrix multiplication
	if (this->n != B.m) {
		std::cout << "matrix multiplication error, matrix dimension mismatch.\n";
		exit(0);
	}

	std::vector<double> retrn(this->rows()* B.n);

	int a = this->rows();
	int b = this->cols();
	
	int c = B.m;
	int d = B.n;

	double* ptr3 = new double[a * d];
	matmult(this->matrix.data(), B.matrix.data(), retrn.data(), a, b, c, d);
	mat mult(retrn, this->m, B.n);
	retrn.clear();
	
	return mult;
}

bool mat::operator==(const mat& B) { // overlaod of comparison operator
	if (this->matrix == B.matrix) { return true; }
	else { return false; }
}

mat mat::operator+(const mat& B) { // matrix addition & broadcasting

	mat sum("zeros", this->m, this->n);
	
	if (this->m != B.m && this->n != B.n) {
		std::cout << "matrix addition error, matrix dimension mismatch.\n";
		exit(0);
	}
	else if (this->m == B.m && this->n == B.n) { // classic matrix addition
		
		matop(0, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->m == B.m && B.n == 1) { // extrude the vetor B to match columns
		matop(3, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->n == B.n && B.m == 1) {// extrude the vetor B to match rows
		matop(4, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else {
		std::cout << "matrix addition error, matrix dimension mismatch.\n";
		exit(0);
	}
	
	return sum;
}

mat mat::operator-(const mat& B) { // matrix substraction & broadcasting

	mat sum("zeros", this->m, this->n);

	if (this->m != B.m && this->n != B.n) {
		std::cout << "matrix substraction error, matrix dimension mismatch.\n";
		exit(0);
	}
	else if (this->m == B.m && this->n == B.n) { // classic matrix substraction
		matop(1, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->m == B.m && B.n == 1) { // extrude the vetor B to match columns
		matop(5, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->n == B.n && B.m == 1) {// extrude the vetor B to match rows
		matop(6, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else {
		std::cout << "matrix substraction error, matrix dimension mismatch.\n";
		exit(0);
	}

	return sum;
}

mat mat::operator^(const mat& B) { // matrix Hadamard product and broadcasting

	mat sum("zeros", this->m, this->n);

	if (this->m != B.m && this->n != B.n) {
		std::cout << "matrix broadcasting error, matrix dimension mismatch.\n";
		exit(0);
	}
	else if (this->m == B.m && this->n == B.n) { // Hadamard Product
		matop(2, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->m == B.m && B.n == 1) { // extrude the vetor B to match columns
		matop(7, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->n == B.n && B.m == 1) {// extrude the vetor B to match rows
		matop(8, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else {
		std::cout << "matrix broadcasting error, matrix dimension mismatch.\n";
		exit(0);
	}

	return sum;
}

mat mat::operator/(const mat& B) { // matrix element by element division and broadcasting

	for (int i = 0; i < B.matrix.size(); i++) {
		if (B.matrix[i] == 0){
			std::cout << "matrix element division error, dividing by zero.\n";
			exit(0);
		}
	}

	mat sum("zeros", this->m, this->n);

	if (this->m != B.m && this->n != B.n) {
		std::cout << "matrix element division error, matrix dimension mismatch.\n";
		exit(0);
	}
	else if (this->m == B.m && this->n == B.n) { // naive division
		matop(9, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->m == B.m && B.n == 1) { // extrude the vetor B to match columns
		matop(10, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->n == B.n && B.m == 1) {// extrude the vetor B to match rows
		matop(11, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else {
		std::cout << "matrix element division error, matrix dimension mismatch.\n";
		exit(0);
	}

	return sum;
}

mat operator*(const mat& A, const double a) { // scalar multiplication

	mat sub("zeros", A.m, A.n);

	matscal(0, a, A.matrix.data(), sub.matrix.data(), A.m, A.n);

	return sub;
}

mat operator*(const double a, const mat& A) { // scalar multiplication

	mat sub("zeros", A.m, A.n);
	
	matscal(0, a, A.matrix.data(), sub.matrix.data(), A.m, A.n);

	return sub;
}

mat mat::operator/(const double a) { // scalar division
	if (a == 0) {
		std::cout << "error: matrix divided by scalar 0.\n";
		exit(0);
	}
	mat sub("zeros", m, n);

	matscal(1, a, matrix.data(), sub.matrix.data(), m, n);

	return sub;
}

mat mat::sum(std::string arg) { // summing columns or rows 
	if (arg == "rows") {
		mat SUM("zeros", this->m, 1);
		matscal(3, 0, matrix.data(), SUM.matrix.data(), m, n);
		return SUM;
	}
	else if (arg == "cols") {
		mat SUM("zeros", 1, this->n);
		matscal(2, 0, matrix.data(), SUM.matrix.data(), m, n);
		return SUM;
	}
	else {
		std::cout << "improper argument for 'sum()' please use 'cols' to sum columns, or 'rows' to sum rows.\n";
		exit(0);
	}


}

double mat::sum() { // summing all matrix elements into a scalar value
	double sum = 0;
#pragma omp parallel for reduction (+:sum)
	for (int i = 0; i < this->m; i++) {
		for (int j = 0; j < this->n; j++) {
			sum += this->matrix[i*n+j];
		}
	}
	return sum;
}

double mat::max() {
	if (this->m == 0 && this->n == 0) { return 0; }
	else if (matrix.size() == 1) {
		return this->matrix[0];
	}
	else {
		double max = this->matrix[0];
		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				if (max < this->matrix[i * n + j]) {
					max = this->matrix[i * n + j];
				}
			}
		}
		return max;
	}
	

}

double mat::min() {
	if (this->m == 0 && this->n == 0) { return 0; }
	else if (matrix.size() == 1) {
		return this->matrix[0];
	}
	else {
		double min = this->matrix[0];
		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				if (min > this->matrix[i * n + j]) {
					min = this->matrix[i * n + j];
				}
			}
		}
		return min;
	}


}

mat mat::xp() {
	mat temp("zeros", m, n);
	#pragma omp parallel for
	for (int i = 0; i < this->m; i++) {
		for (int j = 0; j < this->n; j++) {
			temp(i, j) = exp(this->matrix[i * n + j]);
		}
	}
	return temp;
}


mat mat::T() { // matrix transpose function
	mat transpose("zeros", this->n, this->m); // needs destructor
	#pragma omp parallel for
	for (int i = 0; i < this->m; i++) {
		for (int j = 0; j < this->n; j++) {
			transpose(j, i) = this->matrix[i * n + j];
		}
	}

	return transpose;
}


unsigned int mat::rows() {
	return this->m;
}

unsigned int mat::cols() {
	return this->n;
}

void mat::print() { // matrix print function
	std::cout << std::setprecision(3);
	unsigned int N;
	if (n > 20) {
		std::cout << "the matrix it too wide to print, printing only the first 20 columns: \n";
		N = 20;
	}
	else {
		N = n;
	}

	for (unsigned int i = 0; i < m; i++) {
		for (unsigned int j = 0; j < N; j++) {
			if (j == 0) {
				std::cout << "\n| " << std::setw(8) << matrix[i * N + j] << (j == N - 1 ? " |" : "");
			}
			else {
				std::cout << std::setw(8) << matrix[i * N + j] << (j == N - 1 ? " |" : "");
			}

		}
	}
	std::cout << '\n';
}

#endif
