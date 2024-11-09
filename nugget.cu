#include <cmath>
#include <iostream>
#include "NUGGET.cuh"
#include <vector>
#include <string>
#include <random>
#include <iomanip>
#include <fstream>
#include <chrono>

nugget::nugget(const std::string& read) {
	// constructor to initialize the neural network using a save file
	/*
	 *----------------------------------------------------------------------------------------------------------
	 * arguments:
	 * read: path and name (or name if in working directory) to save file
	 * ---------------------------------------------------------------------------------------------------------
	 */
	std::vector<float> tempvec;
	std::vector<std::vector<float>> tempmat;
	std::ifstream savefile;
	savefile.open(read);
	int tempInt, temp2;
	savefile >> tempInt;
	this->activF = tempInt;
	savefile >> tempInt;
	this->Ninput = tempInt;
	savefile >> tempInt;
	this->Noutput = tempInt;
	savefile >> temp2;
	//reading layer map
	for (int i = 0; i < temp2; i++) {
		savefile >> tempInt;
		this->layer.push_back(tempInt);
	}

	//Reading weights

	tempvec.resize(this->Ninput);
	tempmat.resize(this->layer[0]);
	for (int i = 0; i < this->layer[0]; i++) {
		for (int j = 0; j < this->Ninput; j++) {
			savefile >> tempvec[j];
		}
		tempmat[i] = tempvec;
	}
	this->weight.emplace_back(tempmat);
	tempvec.clear();
	tempmat.clear();


	for (int l = 1; l < layer.size(); l++) {
		tempvec.resize(this->layer[l-1]);
		tempmat.resize(this->layer[l]);
		for (int i = 0; i < this->layer[l]; i++) {
			for (int j = 0; j < this->layer[l - 1]; j++) {
				savefile >> tempvec[j];
			}
			tempmat[i] = tempvec;
		}
		this->weight.emplace_back(tempmat);
		tempvec.clear();
		tempmat.clear();
	}


	tempvec.resize(this->layer.back());
	tempmat.resize(this->Noutput);
	for (int i = 0; i < this->Noutput; i++) {
		for (int j = 0; j < this->layer.back(); j++) {
			savefile >> tempvec[j];
		}
		tempmat[i] = tempvec;
	}
	this->weight.emplace_back(tempmat);
	tempvec.clear();
	tempmat.clear();

	// reading biases

	tempvec.resize(this->layer[0]);
	for (int i = 0; i < this->layer[0]; i++) {
		savefile >> tempvec[i];
	}
	this->bias.emplace_back(tempvec);
	tempvec.clear();


	for (int l = 1; l < layer.size(); l++) {
		tempvec.resize(this->layer[l]);

		for (int i = 0; i < this->layer[l]; i++) {
			savefile >> tempvec[i];

		}
		this->bias.emplace_back(tempvec);
		tempvec.clear();

	}

	tempvec.resize(this->Noutput);
	for (int i = 0; i < this->Noutput; i++) {
		savefile >> tempvec[i];
	}
	this->bias.emplace_back(tempvec);
	tempvec.clear();

	savefile.close();
}

nugget::nugget(int inputs, int outputs, std::vector<int> hid_layers, const std::string& init) {
// constructor to initialize the neural network with specified shape
	/*
	 *----------------------------------------------------------------------------------------------------------
	 * arguments:
	 * inputs: size of the input layer
	 * outputs: size of the output layer
	 * hid_layers: vector representing the hidden layers, with each element representing a layer of size equal
	 * to the element.
	 * init: type of random initialization for weights (uniform or normal distribution)
	 * ---------------------------------------------------------------------------------------------------------
	 */
	bool uni = true;
	if (init == "normal") {
		uni = false;
	}
	else if (init != "uniform") {
		std::cout << "invalid weight initialization constructor argument" << std::endl;
		exit(0);
	}
	this->activF = -1; //placeholder value until the activation function is set in .train()
	this->layer = hid_layers;
	this->Ninput = inputs;
	this->Noutput = outputs;
	this->weight.resize(hid_layers.size() + 1);
	this->bias.resize(hid_layers.size() + 1);

	if (uni) {
		this->weight[0] = xav_uni(hid_layers[0], inputs, inputs, hid_layers[0]);
	}
	else {
		this->weight[0] = xav_norm(hid_layers[0], inputs, inputs, hid_layers[0]);
	}
	mat<float> fstb("zeros", hid_layers[0], 1);

	this -> bias[0] = fstb;

	// creating weight and bias matrix for each hidden layer past layer one
	for (int i = 1; i < hid_layers.size(); i++) {
		if (uni) {
			this->weight[i] = xav_uni(hid_layers[i],
				hid_layers[i - 1],
				hid_layers[i - 1],
				hid_layers[i]);
		}
		else {
			this->weight[i] = xav_norm(hid_layers[i],
				hid_layers[i - 1],
				hid_layers[i - 1],
				hid_layers[i]);
		}
		mat<float> btemp("zeros", hid_layers[i], 1);
		this->bias[i] = btemp;

	}
	// weights and bias to go from the last hidden layer to the output layer
	if (uni) {
		this->weight[hid_layers.size()] = xav_uni(outputs, hid_layers[hid_layers.size() - 1],
		                                          hid_layers[hid_layers.size() - 1], outputs);
	}
	else {
		this->weight[hid_layers.size()] = xav_norm(outputs, hid_layers[hid_layers.size() - 1],
		                                           hid_layers[hid_layers.size() - 1], outputs);
	}
	mat<float> lastb("zeros", outputs, 1);

	this->bias[hid_layers.size()] = (lastb);

}

mat<float> nugget::xav_uni(int sizerow, int sizecol, int inputs, int outputs) {
	//uniform xavier initialization

	float xavier = std::sqrt(6/(static_cast<float>(inputs) + static_cast<float>(outputs)));
	mat W1("rand", -xavier, xavier, sizerow, sizecol);
	return W1;
}

mat<float> nugget::xav_norm(int sizerow, int sizecol, int inputs, int outputs) {
	//normal xavier initialization

	float xavier = std::sqrt(2 / (static_cast<float>(inputs) + static_cast<float>(outputs)));
	mat<float> W1("randN", 0, xavier, sizerow, sizecol);
	return W1;
}

mat<float> nugget::OneHT(mat<float> y, const unsigned int& m) {
	//transforms label 1D array into 2D matrix of depth m
	mat<float> onehot("zeros", m, y.rows());
	for (int i = 0; i < y.rows(); i++) {
		onehot(static_cast<int>(y(i, 0)), i) = 1;
	}

	return onehot;
}

mat<float> nugget::ReLu(mat<float> a) {
	//applies the ReLu activation function on all elements of matrix a
	mat<float> temp = a;
	#pragma omp parallel for
	for (int i = 0; i < a.rows(); i++) {
		for (int j = 0; j < a.cols(); j++) {
			if (a(i, j) <= 0) {
				temp(i, j) = 0;
			}
			else {
				temp(i, j) = a(i, j);
			}
		}
	}

	return temp;
}

mat<float> nugget::ReLuPr(mat<float> a) {
	//applies teh derivative of the ReLu activation function to every element in matrix a
	mat<float> temp = a;
	#pragma omp parallel for
	for (int i = 0; i < a.rows(); i++) {
		for (int j = 0; j < a.cols(); j++) {
			if (a(i, j) <= 0) {
				temp(i, j) = 0;
			}
			else {
				temp(i, j) = 1;
			}
		}
	}

	return temp;
}

float sig(float a) {
	//returns the sigmoid of a
	float b = 1 / (1 + (std::exp(-a)));
	return b;
}

mat<float> nugget::sigmoid(mat<float> a) {
	//applies the sigmoid activation function to every element of matrix a
	mat<float> temp = a;
	#pragma omp parallel for
	for (int i = 0; i < a.rows(); i++) {
		for (int j = 0; j < a.cols(); j++) {

			temp(i, j) = sig(a(i, j));

		}
	}

	return temp;

}

mat<float> nugget::sigmoidPr(mat<float> a) {
	//applies the derivative of the sigmoid function to every element of matrix a
	mat<float> temp = a;
#pragma omp parallel for
	for (int i = 0; i < a.rows(); i++) {
		for (int j = 0; j < a.cols(); j++) {

			temp(i, j) = sig(a(i, j))*(1- sig(a(i, j)));

		}
	}

	return temp;

}

mat<float> nugget::softmax(mat<float> A) {
	//output layer activation function to normalize output and help calculate the loss function
	mat<float> B = A.xp() / A.xp().sum("cols");
	return B;
}

void nugget::accuracy(mat<float> A, mat<float> Y) {
	//calculates the accuracy of the output compared to the labels
	unsigned int m = A.cols();
	float sum = 0;
	float tempmax = 0;
	int indxmax1 = -1;
	int indxmax2 = 0;
	//#pragma omp parallel for
	for (int i = 0; i < A.cols(); i++) {
		tempmax = 0;
		for (int j = 0; j < A.rows(); j++) {
			if (A(j, i) > tempmax) {
				tempmax = A(j, i);
				indxmax1 = j;
			}
			if (Y(j, i) == 1) {

				indxmax2 = j;
			}
		}
		if (indxmax1 == indxmax2) {
			sum++;

		}
	}
	float accuracy = sum / static_cast<float>(m);
	std::cout << "accuracy is " << accuracy << "\n\n";
}

void nugget::save() {
	//generic save function that writes all the neural net parameters to a default ascii file "save.nn"
	char tab = 9;
	std::ofstream savefile;
	savefile.open("save.nn");
	savefile << this->activF << "\n";
	savefile << this->Ninput << "\n";
	savefile << this->Noutput << "\n";
	savefile << this->layer.size() << "\n";
	for (int i = 0; i < this->layer.size(); i++) {
		savefile << this->layer[i];
		if (i < this->layer.size() - 1) {
			savefile << tab;
		}

	}
	savefile << "\n";

	for (int i = 0; i < this->weight.size(); i++) { //Write weight matrices
		if (i != 0) {
			savefile << "\n";
		}
		for (int j = 0; j < this->weight[i].rows(); j++) {
			if (j != 0) {
				savefile << "\n";
			}
			for (int k = 0; k < this->weight[i].cols(); k++) {
				if (k != 0) {
					savefile << tab;
				}
				savefile << this->weight[i](j, k);
			}
		}
	}
	for (auto & bia : this->bias) { //Write weight matrices

		savefile << "\n";

		for (int j = 0; j < bia.rows(); j++) {
			if (j != 0) {
				savefile << tab;
			}
			savefile << bia(j, 0);

		}
	}
	savefile.close();
}

void nugget::save(const std::string& filename) {
	//overloaded save function that takes in a file as parameter
	char tab = 9;
	std::ofstream savefile;
	savefile.open(filename);
	savefile << this->activF << "\n";
	savefile << this->Ninput << "\n";
	savefile << this->Noutput << "\n";
	savefile << this->layer.size() << "\n";
	for (int i = 0; i < this->layer.size(); i++) {
		savefile << this->layer[i];
		if (i < this->layer.size() - 1) {
			savefile << tab;
		}

	}
	savefile << "\n";

	for (int i = 0; i < this->weight.size(); i++) { //Write weight matrices
		if (i != 0) {
			savefile << "\n";
		}
		for (int j = 0; j < this->weight[i].rows(); j++) {
			if (j != 0) {
				savefile << "\n";
			}
			for (int k = 0; k < this->weight[i].cols(); k++) {
				if (k != 0) {
					savefile << tab;
				}
				savefile << this->weight[i](j, k);
			}
		}
	}
	for (auto & bia : this->bias) { //Write weight matrices

		savefile << "\n";

		for (int j = 0; j < bia.rows(); j++) {
			if (j != 0) {
				savefile << tab;
			}
			savefile << bia(j, 0);

		}
	}
	savefile.close();
}


void nugget::train(const mat<float>& raw_data,
	const mat<float>& labels,
	int it,
	const std::string& activ,
	const std::string& o_activ,
	float alpha)
{
	unsigned int m; // size of data sample
	int activation;

	// seting condition for selected activation function
	if (activ == "ReLu") {
		activation = 0;
		this->activF = 0;
	}
	else if (activ == "sigmoid") {
		activation = 1;
		this->activF = 1;
	}
	else {
		std::cout << "invalid activation function argument for training function, please select either 'ReLu' or 'sigmoid'." << std::endl;
		exit(0);
	}
	// normalize and process input data

	mat<float> data = raw_data;
	if (data.rows() == this->Ninput) {
		m = data.cols();
		data = data / data.max();
	}
	else if (data.cols() == this->Ninput) {
		data = data.T();
		m = data.cols();
		data = data / data.max();
	}
	else {
		std::cout << "Dimensions of the data matrix are " << data.rows() << " by " << data.cols() << ", at least one of those dimensions should match the specificed input layer size for this nugget which is" << this->Ninput << "." << std::endl;
		exit(0);
	}


	// 1-hot labels

	mat lab = OneHT(labels, Noutput);

	std::vector<mat<float>> A, Z, dW, dZ, db;
	A.resize(layer.size() + 1);
	Z.resize(layer.size() + 1);
	dW.resize(layer.size() + 1);
	dZ.resize(layer.size() + 1);
	db.resize(layer.size() + 1);
	int epoch = 0;

	while (epoch < it) {
		// forward propagation
		std::chrono::steady_clock::time_point epoch_start = std::chrono::steady_clock::now();

		Z[0] = this->weight[0] * data + this->bias[0];
		if (activation) {

			A[0] = sigmoid(Z[0]);
		}
		else {

			A[0] = ReLu(Z[0]);

		}
		for (int i = 1; i < this->layer.size(); i++) {
			Z[i] = this->weight[i] * A[i - 1] + this->bias[i];

			if (activation) {

				A[i] = sigmoid(Z[i]);
			}
			else {

				A[i] = ReLu(Z[i]);

			}

		}
		/*std::cout << "W3 is "<< this->weight[layer.size()].rows() << " by " << this->weight[layer.size()].cols() <<"\n\n";
		std::cout << "A2 is " << A[layer.size() - 1].rows() << " by " << A[layer.size() - 1].cols() << "\n\n";
		std::cout << "b3 is " << this->bias[layer.size()].rows() << " by " << this->bias[layer.size()].cols() << "\n\n";*/

		Z[layer.size()] = this->weight[layer.size()] * A[layer.size() - 1] + this->bias[layer.size()];

		A[layer.size()] = softmax(Z[layer.size()]); // output layer
				// back propagation

		dZ[layer.size()] = A[layer.size()] - lab;
		dW[layer.size()] = 1 / (float)m * dZ[layer.size()] * A[layer.size() - 1].T();
		db[layer.size()] = 1 / (float)m * dZ[layer.size()].sum("rows");
		for (int i = static_cast<int>(layer.size()) - 1; i > 0; i--) {

			if (activation) {

				dZ[i] = this->weight[i + 1].T() * dZ[i + 1] ^ sigmoidPr(Z[i]);
			}
			else {

				dZ[i] = this->weight[i + 1].T() * dZ[i + 1] ^ ReLuPr(Z[i]);

			}

			//std::cout << "mk3\n\n";
			dW[i] = 1 / (float)m * dZ[i] * A[i - 1].T();
			//std::cout << "mk4\n\n";
			db[i] = 1 / (float)m * dZ[i].sum("rows");
		}

		if (activation) {

			dZ[0] = this->weight[1].T() * dZ[1] ^ sigmoidPr(Z[0]);
		}
		else {

			dZ[0] = this->weight[1].T() * dZ[1] ^ ReLuPr(Z[0]);

		}
		//std::cout << "mk3\n\n";
		dW[0] = 1 / (float)m * dZ[0] * data.T();
		//std::cout << "mk4\n\n"
		db[0] = 1 / (float)m * dZ[0].sum("rows");

		for (int i = 0; i < this->layer.size() + 1; i++) {

			this->weight[i] = weight[i] - alpha * dW[i];

			this->bias[i] = bias[i] - alpha * db[i];
		}
				if (epoch % 10 == 0) {
			std::cout << "Epoch: " << epoch << "\n";
			accuracy(A[layer.size()], lab);


		}
		std::chrono::steady_clock::time_point epoch_end = std::chrono::steady_clock::now();
		std::cout << "epoch time = " << std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start).count() << "[ms]" << std::endl;
		epoch++;
	}

}

void nugget::train(const mat<float>& raw_data,
	const mat<float>& labels,
	int it,
	const std::string& activ,
	const std::string& o_activ,
	float alpha,
	const std::string &filename)
{
	nugget::train(raw_data, labels, it, activ, o_activ, alpha);
	std::cout << "Saving model. "<< "\n";
	save(filename);

}

void nugget::run(const mat<float>& raw_data, const mat<float>& labels) {
	// runing the model is just the forward propagation step with a new dataset. This function takes labels and performs an accuracy calculation at the end.

	// normalize and process input data

	mat<float> data = raw_data;
	if (data.rows() == this->Ninput) {
		data = data / data.max();
	}
	else if (data.cols() == this->Ninput) {
		data = data.T();
		data = data / data.max();
	}
	else {
		std::cout << "Dimensions of the data matrix are " << data.rows() << " by " << data.cols() << ", at least one of those dimensions should match the specificed input layer size for this nugget which is" << this->Ninput << "." << std::endl;
		exit(0);
	}


	// 1-hot labels

	mat lab = OneHT(labels, Noutput);

	std::vector<mat<float>> A, Z;
	A.resize(layer.size() + 1);
	Z.resize(layer.size() + 1);



	// forward propagation
	Z[0] = this->weight[0] * data + this->bias[0];
	if (this->activF) {

		A[0] = sigmoid(Z[0]);
	}
	else {

		A[0] = ReLu(Z[0]);

	}


	for (int i = 1; i < this->layer.size(); i++) {
		Z[i] = this->weight[i] * A[i - 1] + this->bias[i];

		if (this->activF) {

			A[i] = sigmoid(Z[i]);
		}
		else {

			A[i] = ReLu(Z[i]);

		}

	}


	Z[layer.size()] = this->weight[layer.size()] * A[layer.size() - 1] + this->bias[layer.size()];

	A[layer.size()] = softmax(Z[layer.size()]); // output layer

	accuracy(A[layer.size()], lab);

}

void nugget::run(const mat<float>& raw_data) {
	// runing the model is just the forward propagation step with a new dataset. This function takes labels and performs an accuracy calculation at the end.

	mat<float> data = raw_data;
	if (data.rows() == this->Ninput) {

		data = data / data.max();
	}
	else if (data.cols() == this->Ninput) {
		data = data.T();

		data = data / data.max();
	}
	else {
		std::cout << "Dimensions of the data matrix are " << data.rows() << " by " << data.cols() << ", at least one of those dimensions should match the specificed input layer size for this nugget which is" << this->Ninput << "." << std::endl;
		exit(0);
	}


	std::vector<mat<float>> A, Z;
	A.resize(layer.size() + 1);
	Z.resize(layer.size() + 1);



	// forward propagation
	Z[0] = this->weight[0] * data + this->bias[0];
	if (this->activF) {

		A[0] = sigmoid(Z[0]);
	}
	else {

		A[0] = ReLu(Z[0]);

	}


	for (int i = 1; i < this->layer.size(); i++) {
		Z[i] = this->weight[i] * A[i - 1] + this->bias[i];

		if (this->activF) {

			A[i] = sigmoid(Z[i]);
		}
		else {

			A[i] = ReLu(Z[i]);

		}

	}

	Z[layer.size()] = this->weight[layer.size()] * A[layer.size() - 1] + this->bias[layer.size()];
	A[layer.size()] = Z[layer.size()]; // output layer

}
