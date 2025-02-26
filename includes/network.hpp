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
        float *wih;
        float *who;
        float *bih;
        float *bho;

        // Clone function for deep copying
        // Network clone() const
        // {
        //     Network copy;
        //     copy.num_inputs = num_inputs;
        //     copy.num_hidden = num_hidden;
        //     copy.num_outputs = num_outputs;
        //     copy.train_dataset_size = train_dataset_size;
        //     copy.test_dataset_size = test_dataset_size;

        //     int wih_size = num_inputs * num_hidden;
        //     int who_size = num_hidden * num_outputs;
        //     int bih_size = num_hidden;
        //     int bho_size = num_outputs;

        //     copy.wih = new float[wih_size];
        //     copy.who = new float[who_size];
        //     copy.bih = new float[bih_size];
        //     copy.bho = new float[bho_size];

        //     std::memcpy(copy.wih, wih, wih_size * sizeof(float));
        //     std::memcpy(copy.who, who, who_size * sizeof(float));
        //     std::memcpy(copy.bih, bih, bih_size * sizeof(float));
        //     std::memcpy(copy.bho, bho, bho_size * sizeof(float));

        //     return copy;
        // }
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

    void init_network(Network *net, int num_inputs, int num_hidden, int num_outputs, DataReader::Dataset *data, float *wih, float *who, float *bih, float *bho);
    void train_network(Network *net, DataReader::Dataset *data, int num_epochs, float learning_rate);
    void test_network(Network *net, DataReader::Dataset *data);
    void free_network(Network *net);
    ConfusionMatrix calculateConfusionMatrix(int y_true[], int y_pred[], int numInstances);

    // GPU functions
    void train_network_gpu(Network *net, DataReader::Dataset *data, int num_epochs, float learning_rate);
    void test_network_gpu(Network *net, DataReader::Dataset *data);
    void test();

    // Compare two networks
    bool compare_network(Network *network1, Network *network2);

} // namespace NeuralNetwork

#endif /* NETWORK_HPP */
