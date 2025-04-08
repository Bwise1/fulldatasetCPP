#include "../includes/data_reader.hpp"
#include "../includes/network.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>

// Function to write CPU metrics to a CSV file
void writeCPUMetricsToFile(const TrainingMetricsVector &metrics, const std::string &filename)
{
    std::ofstream outputFile(filename);
    if (outputFile.is_open())
    {
        outputFile << "Epoch,Accuracy(%),Total Time(ms)\n";
        for (const auto &metric : metrics)
        {
            outputFile << metric.epoch << ","
                       << std::fixed << std::setprecision(2) << metric.accuracy << ","
                       << metric.total_epoch_time << "\n";
        }
        outputFile.close();
        std::cout << "CPU metrics written to " << filename << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

// Function to write GPU metrics to a CSV file
void writeGPUMetricsToFile(const TrainingMetricsVector &metrics, const std::string &filename)
{
    std::ofstream outputFile(filename);
    if (outputFile.is_open())
    {
        outputFile << "Epoch,Accuracy(%),Kernel Time(ms),Data Copy Time(ms),Total Time(ms)\n";
        for (const auto &metric : metrics)
        {
            outputFile << metric.epoch << ","
                       << std::fixed << std::setprecision(2) << metric.accuracy << ","
                       << metric.kernel_time << ","
                       << metric.data_copy_time << ","
                       << metric.total_epoch_time << "\n";
        }
        outputFile.close();
        std::cout << "GPU metrics written to " << filename << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

// Function to calculate and display total training time from metrics
float calculateTotalTime(const TrainingMetricsVector &metrics)
{
    float total_time = 0.0f;
    for (const auto &metric : metrics)
    {
        total_time += metric.total_epoch_time;
    }
    return total_time;
}

// Function to train network for a specific number of epochs and display results
void trainForEpochs(int num_epochs, DataReader::Dataset *dataset, int num_inputs, int num_hidden, int num_outputs, float learning_rate)
{
    // Initialize new networks for this training session
    NeuralNetwork::Network cpu_network;
    NeuralNetwork::init_network(&cpu_network, num_inputs, num_hidden, num_outputs, dataset);

    NeuralNetwork::Network gpu_network;
    NeuralNetwork::copy_network(&gpu_network, &cpu_network);

    // CPU Training
    std::cout << "\n=== Training for " << num_epochs << " Epochs (CPU) ===\n";
    TrainingMetricsVector cpu_metrics = NeuralNetwork::train_network(&cpu_network, dataset, num_epochs, learning_rate);

    float cpu_total_time = calculateTotalTime(cpu_metrics);
    std::cout << "Total Training Time (CPU) for " << num_epochs << " epochs: " << cpu_total_time << " ms ("
              << std::fixed << std::setprecision(2) << cpu_total_time / 1000.0 << " seconds)\n";

    for (const auto &metric : cpu_metrics)
    {
        std::cout << "Epoch " << metric.epoch
                  << " - Acc: " << std::fixed << std::setprecision(2) << metric.accuracy << "%"
                  << " - Time: " << metric.total_epoch_time << "ms\n";
    }

    // GPU Training
    std::cout << "\n=== Training for " << num_epochs << " Epochs (GPU) ===\n";
    TrainingMetricsVector gpu_metrics = NeuralNetwork::train_network_gpu(&gpu_network, dataset, num_epochs, learning_rate);

    float gpu_total_time = calculateTotalTime(gpu_metrics);
    std::cout << "Total Training Time (GPU) for " << num_epochs << " epochs: " << gpu_total_time << " ms ("
              << std::fixed << std::setprecision(2) << gpu_total_time / 1000.0 << " seconds)\n";

    for (const auto &m : gpu_metrics)
    {
        std::cout << "Epoch " << m.epoch
                  << " - Acc: " << std::fixed << std::setprecision(2) << m.accuracy << "%"
                  << " - Kernel: " << m.kernel_time << "ms"
                  << " - Data Copy: " << m.data_copy_time << "ms"
                  << " - Total: " << m.total_epoch_time << "ms\n";
    }

    // Write metrics to files
    std::string cpu_filename = "cpu_metrics_" + std::to_string(num_epochs) + "_epochs.csv";
    std::string gpu_filename = "gpu_metrics_" + std::to_string(num_epochs) + "_epochs.csv";
    writeCPUMetricsToFile(cpu_metrics, cpu_filename);
    writeGPUMetricsToFile(gpu_metrics, gpu_filename);

    // Test networks after training
    std::cout << "\n=== Testing Network after " << num_epochs << " Epochs (CPU) ===\n";
    NeuralNetwork::test_network(&cpu_network, dataset);

    std::cout << "\n=== Testing Network after " << num_epochs << " Epochs (GPU) ===\n";
    NeuralNetwork::test_network_gpu(&gpu_network, dataset);
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

    // Network parameters
    int num_inputs = 784;
    int num_hidden = 533;
    int num_outputs = 10;
    float learning_rate = 0.001f;

    // Define the epoch points you want to analyze
    std::vector<int> epoch_points = {10, 20, 50, 100};

    // Train for each specified number of epochs
    for (int epochs : epoch_points)
    {
        trainForEpochs(epochs, dataset, num_inputs, num_hidden, num_outputs, learning_rate);
    }

    return 0;
}
