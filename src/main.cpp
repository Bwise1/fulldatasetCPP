#include "../includes/data_reader.hpp"
#include "../includes/network.hpp"
#include <iostream>

int main() {
    std::srand(static_cast<unsigned int>(time(nullptr)));

    // Read data files
    DataReader::Dataset* dataset = DataReader::readDataFiles();
    if (dataset == nullptr) {
        std::cerr << "Failed to read data files." << std::endl;
        return 1;
    }

    // Access the data in the dataset struct
    float** trainInputData = dataset->trainInputData;
    int** trainTargetData = dataset->trainTargetData;

    // Print a sample from the train input data
    std::cout << "Train Input Data:" << std::endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << trainInputData[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Print a sample from the train target data
    std::cout << "Train Target Data:" << std::endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << trainTargetData[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Initialize network
    int num_inputs = 784;
    int num_hidden = 100;
    int num_outputs = 10;
    NeuralNetwork::Network network;
    NeuralNetwork::init_network(&network, num_inputs, num_hidden, num_outputs, dataset);

    NeuralNetwork::Network network_gpu = network;
   if (NeuralNetwork::compare_network(&network,&network_gpu)){
        std::cout<<"\nSame network parameters\n";
   }
   std::cout<<"\n\nTraining Network\n\n";

    //train
    // NeuralNetwork::train_network(&network, dataset, 20, 0.001);

    // std::cout<<"\n\nTesting Network\n\n";

    // NeuralNetwork::test_network(&network, dataset);

    std::cout<<"\n\nTraining Network GPU\n\n";
    NeuralNetwork::train_network_gpu(&network, dataset, 20, 0.001);
    return 0;
}
