#include "../includes/data_reader.hpp"
#include "../includes/network.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>

// Function to write CPU metrics to a CSV file for specific epochs
void writeCPUMetricsToFile(const TrainingMetricsVector &metrics, const std::string &filename, const std::vector<int> &epoch_points)
{
    std::ofstream outputFile(filename);
    if (outputFile.is_open())
    {
        outputFile << "Epoch,Accuracy(%),Total Time(ms)\n";
        for (const auto &metric : metrics)
        {
            // Only write metrics for the specified epoch points
            if (std::find(epoch_points.begin(), epoch_points.end(), metric.epoch) != epoch_points.end())
            {
                outputFile << metric.epoch << ","
                           << std::fixed << std::setprecision(2) << metric.accuracy << ","
                           << metric.total_epoch_time << "\n";
            }
        }
        outputFile.close();
        std::cout << "CPU metrics written to " << filename << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

// Function to write GPU metrics to a CSV file for specific epochs
void writeGPUMetricsToFile(const TrainingMetricsVector &metrics, const std::string &filename, const std::vector<int> &epoch_points)
{
    std::ofstream outputFile(filename);
    if (outputFile.is_open())
    {
        outputFile << "Epoch,Accuracy(%),Kernel Time(ms),Data Copy Time(ms),Total Time(ms)\n";
        for (const auto &metric : metrics)
        {
            // Only write metrics for the specified epoch points
            if (std::find(epoch_points.begin(), epoch_points.end(), metric.epoch) != epoch_points.end())
            {
                outputFile << metric.epoch << ","
                           << std::fixed << std::setprecision(2) << metric.accuracy << ","
                           << metric.kernel_time << ","
                           << metric.data_copy_time << ","
                           << metric.total_epoch_time << "\n";
            }
        }
        outputFile.close();
        std::cout << "GPU metrics written to " << filename << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

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

    // Initialize network
    int num_inputs = 784;
    int num_hidden = 533;
    int num_outputs = 10;
    float learning_rate = 0.001f;
    int max_epochs = 100; // Train up to the maximum number of epochs

    // Define the epoch points you want to analyze
    std::vector<int> epoch_points = {10, 20, 50, 100};

    NeuralNetwork::Network network;
    NeuralNetwork::init_network(&network, num_inputs, num_hidden, num_outputs, dataset);

    NeuralNetwork::Network network_gpu;
    NeuralNetwork::copy_network(&network_gpu, &network);

    std::cout << "\n\nTraining Network (CPU)\n\n";
    // Train CPU up to max_epochs
    TrainingMetricsVector cpu_metrics = NeuralNetwork::train_network(&network, dataset, max_epochs, learning_rate);
    for (const auto &metric : cpu_metrics)
    {
        std::cout << "Epoch " << metric.epoch
                  << " - Acc: " << std::fixed << std::setprecision(2) << metric.accuracy << "%"
                  << " - Time: " << metric.total_epoch_time << "ms\n";
    }

    std::cout << "\n\nTesting Network (CPU)\n\n";
    NeuralNetwork::test_network(&network, dataset);

    std::cout << "\n\nTraining Network (GPU)\n\n";
    // Train GPU up to max_epochs
    TrainingMetricsVector gpu_metrics = NeuralNetwork::train_network_gpu(&network_gpu, dataset, max_epochs, learning_rate);
    for (const auto &m : gpu_metrics)
    {
        std::cout << "Epoch " << m.epoch
                  << " - Acc: " << std::fixed << std::setprecision(2) << m.accuracy << "%"
                  << " - Kernel: " << m.kernel_time << "ms"
                  << " - Data Copy: " << m.data_copy_time << "ms"
                  << " - Total: " << m.total_epoch_time << "ms\n";
    }

    std::cout << "\n\nTesting Network (GPU)\n\n";
    NeuralNetwork::test_network_gpu(&network_gpu, dataset);

    // Write metrics for specific epoch points to files
    writeCPUMetricsToFile(cpu_metrics, "cpu_metrics.csv", epoch_points);
    writeGPUMetricsToFile(gpu_metrics, "gpu_metrics.csv", epoch_points);

    return 0;
}
