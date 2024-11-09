#pragma once
#include "MATRIX.cuh"
#include <vector>
#include <string>
#include <iomanip>

#ifndef NUG_H
#define NUG_H
class nugget {

private:
	
	std::vector<mat<float>> weight; // vector of matrices for the weights of each layer
	std::vector<mat<float>> bias; // vector of matrices for the biases of each layer
	std::vector<int> layer; // vector of ints describing the hidden layers
	//std::string activation; // Type of activation function for hidden layers
	//std::string out_activation; // typer of activation for output layer
	//std::string init; // type of weight initialization
	int Ninput, Noutput; // number of inputs and outputs neurons
	int activF; //Type of activation function used during training (set during training)
	static mat<float> xav_uni(int sizerow, int sizecol, int inputs, int outputs); // uniform xavier initialization
	static mat<float> xav_norm(int sizerow, int sizecol, int inputs, int outputs); // normal xavier initialization
	static mat<float> ReLu(mat<float> a); // ReLu function for forward propagation
	static mat<float> ReLuPr(mat<float> a); // Derivative of ReLu for back propagation
	static mat<float> sigmoid(mat<float> a); // Sigmoid function for forward propagation
	static mat<float> sigmoidPr(mat<float> a); // Derivative of the sigmoid function for the back propagation
	friend float sig(float);
	static mat<float> OneHT(mat<float> y);
	static mat<float> softmax(mat<float> A);
	static void accuracy(mat<float> A, mat<float> y);
	void save(const std::string&);
	void save();

public:
	// -- constructors --
	nugget(int inputs, int outputs, std::vector<int> hid_layers, const std::string& init);
	nugget(const std::string& read); // initialize using a save file of a previously trained NN
	// -- methods --

	void train(const mat<float>& data, const mat<float>& labels, int it, const std::string& activ, const std::string& o_activ, float alpha);
	void train(const mat<float>& data, const mat<float>& labels, int it, const std::string& activ, const std::string& o_activ, float alpha, const std::string &);
	void run(const mat<float>& data, const mat<float>& labels);
	void run(const mat<float>& data);
	

};

#endif
