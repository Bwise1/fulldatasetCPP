#include "../includes/data_reader.hpp"
#include "../includes/network.hpp"
#include "../includes/utils.hpp"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Read data files
    DataReader::Dataset *dataset = DataReader::readDataFiles();
    if (dataset == nullptr)
    {
        std::cerr << "Failed to read data files." << std::endl;
        return 1;
    }

    // Access the data in the dataset struct
    float *trainInputData = dataset->trainInputData;
    int *trainTargetData = dataset->trainTargetData;

    // Print a sample from the train input data
    std::cout << "Train Input Data (First Sample):" << std::endl;
    for (int j = 0; j < 784; j++)
    {
        std::cout << trainInputData[j] << " ";
    }
    std::cout << std::endl;

    // Print a sample from the train target data
    std::cout << "Train Target Data (First 10 Classes):" << std::endl;
    for (int j = 0; j < 10; j++)
    {
        std::cout << trainTargetData[j] << " ";
    }
    std::cout << std::endl;

    // Initialize network parameters
    int num_inputs = 784;
    int num_hidden = 128;
    int num_outputs = 10;

    // Allocate and initialize weights and biases
    float *wih = initialize_weights(num_inputs * num_hidden);
    float *who = initialize_weights(num_hidden * num_outputs);
    float *bih = initialize_biases(num_hidden);
    float *bho = initialize_biases(num_outputs);

    // Initialize network
    NeuralNetwork::Network network;
    NeuralNetwork::init_network(&network, num_inputs, num_hidden, num_outputs, dataset, wih, who, bih, bho);

    // Create a deep copy for GPU network
    // NeuralNetwork::Network network_gpu = network.clone();

    // // Compare network parameters
    // if (NeuralNetwork::compare_network(&network, &network_gpu))
    // {
    //     std::cout << "\nSame network parameters\n";
    // }
    // else
    // {
    //     std::cout << "\nNetwork parameters differ\n";
    // }

    std::cout << "\n\nTraining Network GPU\n\n";
    NeuralNetwork::train_network(&network, dataset, 20, 0.001f);

    // Clean up allocated memory
    NeuralNetwork::free_network(&network);
    // NeuralNetwork::free_network(&network_gpu);
    delete[] wih;
    delete[] who;
    delete[] bih;
    delete[] bho;
    delete dataset; // Assuming Dataset was allocated with new

    return 0;
}
