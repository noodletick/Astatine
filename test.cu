// nugget.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "NUGGET.cuh"
#include <fstream>
#include <vector>
#include <string>

int main()
{
//  ---------- Reading training data --------------
    std::ifstream Data, Labels;

    std::vector<float> labels, temp_dat;
    std::vector<std::vector<float>> data;
    float temp1;
    //unsigned int count;
    float temp;

    Labels.open("labels.dat");

    while (true) {

        Labels >> temp;
        temp1 = temp;
        labels.push_back(temp1);
        if (Labels.eof()) { break; }
    }

    Labels.close();

    Data.open("data.dat");

    temp_dat.resize(784);

    while (true) {
        for (int i = 0; i < 784; i++) {
            Data >> temp;
            temp1 = temp;
            temp_dat[i] = temp1;

        }

        data.push_back(temp_dat);

        if (Data.eof()) { break; }
    }
    Data.close();
    labels.pop_back();
    data.pop_back();

    std::cout << "There are " << labels.size() << " labels and " << data.size() << " image arrays of size " << data[0].size() << "\n\n";

    //  ---------- Reading test data --------------

    std::cout << "Reading test data.\n\n";

    Labels.open("test_labels.dat");

    std::vector<float> Tlabels;
    std::vector<std::vector<float>> Tdata;

    while (true) {

        Labels >> temp;
        temp1 = temp;
        Tlabels.push_back(temp1);
        if (Labels.eof()) { break; }
    }

    Labels.close();

    Data.open("test_data.dat");

    temp_dat.resize(784);

    while (true) {
        for (int i = 0; i < 784; i++) {
            Data >> temp;
            temp1 = temp;
            temp_dat[i] = temp1;

        }

        Tdata.push_back(temp_dat);

        if (Data.eof()) { break; }
    }
    Data.close();
    Tlabels.pop_back();
    Tdata.pop_back();

    std::cout << "In test data, there are " << Tlabels.size() << " labels and " << Tdata.size() << " image arrays of size " << Tdata[0].size() << "\n\n";

    //  ---------- Initializing neural network --------------

    std::vector<int> hidden_layers = {80, 80, 80 };

    std::cout << "Initializing neural net.\n\n";
    nugget test_nug(784, 10, hidden_layers, "uniform");

    //  ---------- Training neural network --------------

    std::cout << "Training.\n\n";
    //test_nug.train(data, labels, 20, "ReLu", "softmax", 0.02);
    test_nug.train(data, labels, 100, "softmax", "leaky ReLu", 0.01,"exponential", 0.035, 2, "TestSave.txt");
    //sigmoid

    //  ---------- Running test data --------------
    std::cout << "Testing on new data.\n\n";
    test_nug.run(Tdata, Tlabels);

    //  ---------- initializing new nugget and reading in save file --------------
    std::cout << "Initializing new nugget and reading in save file.\n\n";
    nugget newnug("TestSave.txt");

    //  ---------- Running test data on newnug--------------

    std::cout << "Testing on new data.\n\n";
    newnug.run(Tdata, Tlabels);
}
