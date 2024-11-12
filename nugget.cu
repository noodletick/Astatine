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
	this->L_ReLu_factor = 0;
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

mat<float> nugget::normalize(const mat<float>& a, const unsigned int& inputs) {

	mat<float> normalized = a;
	if (normalized.rows() == inputs) {
		normalized  = normalized / normalized.max();
	}
	else if (normalized.cols() == inputs) {
		normalized = normalized.T();
		normalized  = normalized / normalized.max();
	}
	else {
		std::cout << "Dimensions of the data matrix are " << normalized.rows() << " by " << normalized.cols() <<
				", at least one of those dimensions should match the specified input layer size for this nugget which is"
				<< inputs << "." << std::endl;
		exit(0);
	}
	return normalized;
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

mat<float> nugget::leaky_ReLu(mat<float> a, const float& b) {
	//applies the Leaky ReLu activation function on all elements of matrix a with factor b for x < 0
	mat<float> temp = a;
	#pragma omp parallel for
	for (int i = 0; i < a.rows(); i++) {
		for (int j = 0; j < a.cols(); j++) {
			if (a(i, j) <= 0) {
				temp(i, j) = a(i, j)*b;
			}
			else {
				temp(i, j) = a(i, j);
			}
		}
	}

	return temp;
}

mat<float> nugget::leaky_ReLuPr(mat<float> a, const float& b) {
	//applies the derivative of the Leaky ReLu on all elements of matrix a with factor b for x < 0
	mat<float> temp = a;
	#pragma omp parallel for
	for (int i = 0; i < a.rows(); i++) {
		for (int j = 0; j < a.cols(); j++) {
			if (a(i, j) <= 0) {
				temp(i, j) = b;
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


void nugget::core_train(const mat<float>& raw_data,
	const mat<float>& labels,
	const int it,
	const std::string& o_activ,
	const std::string& activ,
	const float L_RelU_factor,
	const std::string& learning_schedule,
	float lschedule1,
	float lschedule2)
{
	// determine size of data set
	unsigned int m;
	if (raw_data.cols() == this->Ninput) {
		m = raw_data.rows();
	}
	else {
		m = raw_data.cols();
	}


	// setting condition for selected activation function
	if (activ == "ReLu") {
		this->activF = 0;
	}
	else if (activ == "sigmoid") {
		this->activF = 1;
	}
	else if (activ == "leaky ReLu") {
		this->activF = 2;
		if (L_RelU_factor < 0) {
			std::cout<<"WARNING: Leaky ReLu factor is smaller than 0\n";
		}
	}
	else {
		std::cout <<
				"invalid activation function argument for training function, please select either 'ReLu', 'leaky ReLu, or 'sigmoid'."
				<< std::endl;
		exit(0);
	}

	// validating learning schedule arguments
	if (learning_schedule == "fixed" && lschedule1 != 0) {
		lschedule2 = 0;
	}
	else if (learning_schedule == "exponential" && lschedule1 != 0) {
		// all good
	}
	else {
		std::cout<<"invalid learning schedule parameters\n";
		exit(0);
	}
	// normalize and process input data
	mat<float> data = normalize(raw_data, Ninput);

	// 1-hot labels
	mat lab = OneHT(labels, Noutput);

	std::vector<mat<float>> A, Z, dW, dZ, db;
	A.resize(layer.size() + 1);
	Z.resize(layer.size() + 1);
	dW.resize(layer.size() + 1);
	dZ.resize(layer.size() + 1);
	db.resize(layer.size() + 1);
	int epoch = 0;

	while (epoch < it) {//training loop
		//learning rate
		float alpha = lschedule1 * std::exp(static_cast<float>(epoch)/static_cast<float>(it)*lschedule2);
		// ------------------ forward propagation --------------------------
		std::chrono::steady_clock::time_point epoch_start = std::chrono::steady_clock::now();
		Z[0] = this->weight[0] * data + this->bias[0];

		switch (this->activF) {
			case 0:
				A[0] = ReLu(Z[0]);
				break;
			case 1:
				A[0] = sigmoid(Z[0]);
				break;
			case 2:
				A[0] = leaky_ReLu(Z[0], L_RelU_factor);
				break;
			default:
				std::cout<<"invalid switch case\n";
				exit(0);
		}

		for (int i = 1; i < this->layer.size(); i++) {
			Z[i] = this->weight[i] * A[i - 1] + this->bias[i];

			switch (this->activF) {
				case 0:
					A[i] = ReLu(Z[i]);
					break;
				case 1:
					A[i] = sigmoid(Z[i]);
					break;
				case 2:
					A[i] = leaky_ReLu(Z[i], L_RelU_factor);
					break;
				default:
					std::cout<<"invalid switch case\n";
					exit(0);
			}
		}
		Z[layer.size()] = this->weight[layer.size()] * A[layer.size() - 1] + this->bias[layer.size()];
		A[layer.size()] = softmax(Z[layer.size()]); // output layer

		//----------------------- back propagation ---------------------------

		dZ[layer.size()] = A[layer.size()] - lab;
		dW[layer.size()] = 1 / static_cast<float>(m) * dZ[layer.size()] * A[layer.size() - 1].T();
		db[layer.size()] = 1 / static_cast<float>(m) * dZ[layer.size()].sum("rows");

		for (int i = static_cast<int>(layer.size()) - 1; i > 0; i--) {
			switch (this->activF) {
				case 0:
					dZ[i] = this->weight[i + 1].T() * dZ[i + 1] ^ ReLuPr(Z[i]);
					break;
				case 1:
					dZ[i] = this->weight[i + 1].T() * dZ[i + 1] ^ sigmoidPr(Z[i]);
					break;
				case 2:
					dZ[i] =	this->weight[i + 1].T() * dZ[i + 1] ^ leaky_ReLuPr(Z[i], L_RelU_factor);
					break;
				default:
					std::cout<<"invalid switch case\n";
					exit(0);
			}

			dW[i] = 1 / static_cast<float>(m) * dZ[i] * A[i - 1].T();
			db[i] = 1 / static_cast<float>(m) * dZ[i].sum("rows");
		}

		switch (this->activF) {
			case 0:
				dZ[0] = this->weight[1].T() * dZ[1] ^ ReLuPr(Z[0]);
				break;
			case 1:
				dZ[0] = this->weight[1].T() * dZ[1] ^ sigmoidPr(Z[0]);
				break;
			case 2:
				dZ[0] = this->weight[1].T() * dZ[1] ^ leaky_ReLuPr(Z[0], L_RelU_factor);
				break;
			default:
				std::cout<<"invalid switch case\n";
				exit(0);
		}

		dW[0] = 1 / static_cast<float>(m) * dZ[0] * data.T();
		db[0] = 1 / static_cast<float>(m) * dZ[0].sum("rows");

		std::cout<<"alpha is "<<alpha<<"\n";
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
	const int it,
	const std::string& o_activ,
	const std::string& activ,
	const float L_RelU_factor,
	const std::string& learning_schedule,
	float lschedule1,
	float lschedule2,
	const std::string& saveFile)
{
	this->L_ReLu_factor = L_RelU_factor;
	core_train(raw_data, labels, it, o_activ, activ, L_RelU_factor, learning_schedule, lschedule1, lschedule2);
	std::cout << "Saving model. "<< "\n";
	save(saveFile);
}
void nugget::train(const mat<float>& raw_data,
	const mat<float>& labels,
	const int it,
	const std::string& o_activ,
	const std::string& activ,
	const float L_RelU_factor,
	const std::string& learning_schedule,
	float lschedule1,
	float lschedule2)
{
	this->L_ReLu_factor = L_RelU_factor;
	core_train(raw_data, labels, it, o_activ, activ, L_RelU_factor, learning_schedule, lschedule1, lschedule2);
	std::cout << "Saving model. "<< "\n";
	save();
}
void nugget::train(const mat<float>& raw_data,
	const mat<float>& labels,
	const int it,
	const std::string& o_activ,
	const std::string& activ,
	const float L_RelU_factor,
	const std::string& learning_schedule,
	float lschedule1,
	const std::string& saveFile)
{
	this->L_ReLu_factor = L_RelU_factor;
	if(learning_schedule != "fixed") {
		std::cout <<
				"training function argument for learning schedule indicates 'exponential' but only one schedule value passed instead of two.\n";
		exit(0);
	}
	core_train(raw_data, labels, it, o_activ, activ, L_RelU_factor, learning_schedule, lschedule1, 0);
	std::cout << "Saving model. "<< "\n";
	save(saveFile);
}
void nugget::train(const mat<float>& raw_data,
	const mat<float>& labels,
	const int it,
	const std::string& o_activ,
	const std::string& activ,
	const float L_RelU_factor,
	const std::string& learning_schedule,
	float lschedule1)
{
	this->L_ReLu_factor = L_RelU_factor;
	if(learning_schedule != "fixed") {
		std::cout <<
				"training function argument for learning schedule indicates 'exponential' but only one schedule value passed instead of two.\n";
		exit(0);
	}
	core_train(raw_data, labels, it, o_activ, activ, L_RelU_factor, learning_schedule, lschedule1, 0);
	std::cout << "Saving model. "<< "\n";
	save();
}
void nugget::train(const mat<float>& raw_data,
	const mat<float>& labels,
	const int it,
	const std::string& o_activ,
	const std::string& activ,
	const std::string& learning_schedule,
	float lschedule1,
	float lschedule2,
	const std::string& saveFile)
{
	if(activ == "leaky ReLu") {
		std::cout <<
				"training function argument for activation function indicates 'leaky ReLu' but no activation function parameter passed (expected one).\n";
		exit(0);
	}
	core_train(raw_data, labels, it, o_activ, activ, 0, learning_schedule, lschedule1, lschedule2);
	std::cout << "Saving model. "<< "\n";
	save(saveFile);
}
void nugget::train(const mat<float>& raw_data,
	const mat<float>& labels,
	const int it,
	const std::string& o_activ,
	const std::string& activ,
	const std::string& learning_schedule,
	float lschedule1,
	float lschedule2)
{
	if(activ == "leaky ReLu") {
		std::cout <<
				"training function argument for activation function indicates 'leaky ReLu' but no activation function parameter passed (expected one).\n";
		exit(0);
	}
	core_train(raw_data, labels, it, o_activ, activ, 0, learning_schedule, lschedule1, lschedule2);
	std::cout << "Saving model. "<< "\n";
	save();
}
void nugget::train(const mat<float>& raw_data,
	const mat<float>& labels,
	const int it,
	const std::string& o_activ,
	const std::string& activ,
	const std::string& learning_schedule,
	float lschedule1,
	const std::string& saveFile)
{
	if(activ == "leaky ReLu") {
		std::cout <<
				"training function argument for activation function indicates 'leaky ReLu' but no activation function parameter passed (expected one).\n";
		exit(0);
	}
	if(learning_schedule != "fixed") {
		std::cout <<
				"training function argument for learning schedule indicates 'exponential' but only one schedule value passed instead of two.\n";
		exit(0);
	}
	core_train(raw_data, labels, it, o_activ, activ, 0, learning_schedule, lschedule1, 0);
	std::cout << "Saving model. "<< "\n";
	save(saveFile);
}
void nugget::train(const mat<float>& raw_data,
	const mat<float>& labels,
	const int it,
	const std::string& o_activ,
	const std::string& activ,
	const std::string& learning_schedule,
	float lschedule1)
{
	if(activ == "leaky ReLu") {
		std::cout <<
				"training function argument for activation function indicates 'leaky ReLu' but no activation function parameter passed (expected one).\n";
		exit(0);
	}
	if(learning_schedule != "fixed") {
		std::cout <<
				"training function argument for learning schedule indicates 'exponential' but only one schedule value passed instead of two.\n";
		exit(0);
	}
	core_train(raw_data, labels, it, o_activ, activ, 0, learning_schedule, lschedule1, 0);
	std::cout << "Saving model. "<< "\n";
	save();
}
void nugget::run(const mat<float>& raw_data, const mat<float>& labels) {
	// running the model is just the forward propagation step with a new dataset.
	// This function takes labels and performs an accuracy calculation at the end.
	const float L_RelU_factor = this->L_ReLu_factor;
	// normalize and process input data

	mat<float> data = normalize(raw_data, Ninput);

	// 1-hot labels

	mat<float> lab = OneHT(labels, Noutput);

	std::vector<mat<float>> A, Z;
	A.resize(layer.size() + 1);
	Z.resize(layer.size() + 1);

	// forward propagation
	Z[0] = this->weight[0] * data + this->bias[0];

	switch (this->activF) {
		case 0:
			A[0] = ReLu(Z[0]);
		break;
		case 1:
			A[0] = sigmoid(Z[0]);
		break;
		case 2:
			A[0] = leaky_ReLu(Z[0], L_RelU_factor);
		break;
		default:
			std::cout<<"invalid switch case\n";
		exit(0);
	}

	for (int i = 1; i < this->layer.size(); i++) {
		Z[i] = this->weight[i] * A[i - 1] + this->bias[i];

		switch (this->activF) {
			case 0:
				A[i] = ReLu(Z[i]);
			break;
			case 1:
				A[i] = sigmoid(Z[i]);
			break;
			case 2:
				A[i] = leaky_ReLu(Z[i], L_RelU_factor);
			break;
			default:
				std::cout<<"invalid switch case\n";
			exit(0);
		}
	}

	Z[layer.size()] = this->weight[layer.size()] * A[layer.size() - 1] + this->bias[layer.size()];
	A[layer.size()] = softmax(Z[layer.size()]); // output layer

	accuracy(A[layer.size()], lab);

}

void nugget::run(const mat<float>& raw_data) {
	// running the model is just the forward propagation step with a new dataset.
	// This function takes labels and performs an accuracy calculation at the end.
	const float L_RelU_factor = this->L_ReLu_factor;
	// normalize and process input data

	mat<float> data = normalize(raw_data, Ninput);

	std::vector<mat<float>> A, Z;
	A.resize(layer.size() + 1);
	Z.resize(layer.size() + 1);

	// forward propagation
	Z[0] = this->weight[0] * data + this->bias[0];

	switch (this->activF) {
		case 0:
			A[0] = ReLu(Z[0]);
		break;
		case 1:
			A[0] = sigmoid(Z[0]);
		break;
		case 2:
			A[0] = leaky_ReLu(Z[0], L_RelU_factor);
		break;
		default:
			std::cout<<"invalid switch case\n";
		exit(0);
	}

	for (int i = 1; i < this->layer.size(); i++) {
		Z[i] = this->weight[i] * A[i - 1] + this->bias[i];

		switch (this->activF) {
			case 0:
				A[i] = ReLu(Z[i]);
			break;
			case 1:
				A[i] = sigmoid(Z[i]);
			break;
			case 2:
				A[i] = leaky_ReLu(Z[i], L_RelU_factor);
			break;
			default:
				std::cout<<"invalid switch case\n";
			exit(0);
		}
	}

	Z[layer.size()] = this->weight[layer.size()] * A[layer.size() - 1] + this->bias[layer.size()];
	A[layer.size()] = Z[layer.size()]; // output layer

}
