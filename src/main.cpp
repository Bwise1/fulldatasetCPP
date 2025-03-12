#include "../includes/data_reader.hpp"
#include "../includes/network.hpp"
#include <iostream>

int main()
{
    std::srand(static_cast<unsigned int>(time(nullptr)));

    // Read data files
    DataReader::Dataset *dataset = DataReader::readDataFiles();
    if (dataset == nullptr)
    {
        std::cerr << "Failed to read data files." << std::endl;
        return 1;
    }

    // Access the data in the dataset struct
    // float **trainInputData = dataset->trainInputData;
    // int **trainTargetData = dataset->trainTargetData;

    // Print a sample from the train input data
    // std::cout << "Train Input Data:" << std::endl;
    // for (int i = 0; i < dataset->trainSize; i++) {
    //     for (int j = 0; j < 784; j++) {
    //         std::cout << trainInputData[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Print a sample from the train target data
    // std::cout << "Train Target Data:" << std::endl;
    // for (int i = 0; i < 10; i++)
    // {
    //     for (int j = 0; j < 10; j++)
    //     {
    //         std::cout << trainTargetData[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Initialize network
    int num_inputs = 784;
    int num_hidden = 256;
    int num_outputs = 10;

    NeuralNetwork::Network network;
    NeuralNetwork::init_network(&network, num_inputs, num_hidden, num_outputs, dataset);
    // NeuralNetwork::DeviceNetwork d_network;
    // NeuralNetwork::Network network_gpu = network;

    NeuralNetwork::Network network_gpu;
    NeuralNetwork::copy_network(&network_gpu, &network);
    // if (NeuralNetwork::compare_network(&network, &network_gpu))
    // {
    //     std::cout << "\nSame network parameters\n";
    // }
    std::cout << "\n\nTraining Network\n\n";

    // train

    TrainingMetricsVector cpu_metrics = NeuralNetwork::train_network(&network, dataset, 1, 0.001);
    for (const auto &metric : cpu_metrics)
    {
        std::cout << "Epoch " << metric.epoch
                  << " - Acc: " << metric.accuracy << "%"
                  << " - Time: " << metric.total_epoch_time << "ms\n";
    }

    std::cout << "\n\nTesting Network\n\n";
    NeuralNetwork::test_network(&network, dataset);

    std::cout << "\n\nTraining Network GPU\n\n";
    TrainingMetricsVector metrics = NeuralNetwork::train_network_gpu(&network_gpu, dataset, 1, 0.001);
    for (const auto &m : metrics)
    {
        std::cout << "Epoch " << m.epoch
                  << " - Acc: " << m.accuracy << "%"
                  << " - Kernel: " << m.kernel_time << "ms"
                  << " - Data Copy: " << m.data_copy_time << "ms"
                  << " - Total: " << m.total_epoch_time << "ms\n";
    }

    std::cout << "\n\nTesting Network GPU\n\n";
    NeuralNetwork::test_network_gpu(&network_gpu, dataset);
    return 0;
}
