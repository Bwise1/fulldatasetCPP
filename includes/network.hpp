#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "data_reader.hpp"

namespace NeuralNetwork
{

    struct Network
    {
        int num_inputs;
        int num_hidden;
        int num_outputs;
        int train_dataset_size;
        int test_dataset_size;
        float **wih; // input to hidden layer weights
        float **who; // hidden to output layer weights
        float *bih;  // bias for hidden layer
        float *bho;  // bias for output layer
    };

    struct DeviceNetwork
    {
        int num_inputs;
        int num_hidden;
        int num_outputs;
        int train_dataset_size;
        float *d_wih;      // num_inputs * num_hidden
        float *d_who;      // num_hidden * num_outputs
        float *d_bih;      // num_hidden
        float *d_bho;      // num_outputs
        float *d_who_grad; // New gradient buffers
        float *d_bho_grad;
        float *d_wih_grad;
        float *d_bih_grad;
    };

    struct ConfusionMatrix
    {
        int truePositive;
        int falsePositive;
        int trueNegative;
        int falseNegative;
    };

    void softmax(float *arr, int size);
    void feedforward(Network *net, float *input, float *hidden_outputs, float *output_outputs);
    void backpropagate(Network *net, float *input, int *target, float *hidden_outputs, float *output_outputs, float learning_rate);
    int get_predicted_class(float *output_outputs, int num_outputs);
    int get_true_class(int *targets, int num_targets);

    void init_network(Network *net, int num_inputs, int num_hidden, int num_outputs, DataReader::Dataset *data);
    void train_network(Network *net, DataReader::Dataset *data, int num_epochs, float learning_rate);
    void test_network(Network *net, DataReader::Dataset *data);
    void free_network(Network *net);
    ConfusionMatrix calculateConfusionMatrix(int y_true[], int y_pred[], int numInstances);

    // gpu functions
    // void NeuralNetwork::init_network_gpu(Network *net, DeviceNetwork *d_net);
    void init_network_gpu(Network *net, DeviceNetwork *d_net);
    void free_network_gpu(DeviceNetwork *d_net);
    void train_network_gpu(Network *net, DataReader::Dataset *data, int num_epochs, float learning_rate);
    void test_network_gpu(Network *net, DataReader::Dataset *data);
    void test();

    // compare 2 network
    bool compare_network(Network *network1, Network *network2);

} // namespace NeuralNetwork

#endif /* NETWORK_HPP */
