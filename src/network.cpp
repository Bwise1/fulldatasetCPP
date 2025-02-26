#include <cmath> // Include cmath for math functions
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "../includes/network.hpp"
#include "../includes/utils.hpp"

/* Softmax function */
void NeuralNetwork::softmax(float *arr, int size)
{
    float max_val = arr[0];
    for (int i = 1; i < size; i++)
    {
        if (arr[i] > max_val)
        {
            max_val = arr[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++)
    {
        sum_exp += std::exp(arr[i] - max_val);
    }

    for (int i = 0; i < size; i++)
    {
        arr[i] = std::exp(arr[i] - max_val) / sum_exp;
    }
}

void NeuralNetwork::feedforward(NeuralNetwork::Network *net, float *input, float *hidden_outputs, float *output_outputs)
{
    int hid, out;
    float sum;

    /* Calculate input to hidden layer */
    for (hid = 0; hid < net->num_hidden; hid++)
    {
        sum = 0.0f;
        for (int inp = 0; inp < net->num_inputs; inp++)
        {
            sum += input[inp] * net->wih[inp * net->num_hidden + hid];
        }

        /* Add in Bias */
        sum += net->bih[hid];
        hidden_outputs[hid] = relu(sum);
    }

    /* Calculate the hidden to output layer */
    for (out = 0; out < net->num_outputs; out++)
    {
        sum = 0.0f;
        for (hid = 0; hid < net->num_hidden; hid++)
        {
            sum += hidden_outputs[hid] * net->who[hid * net->num_outputs + out];
        }
        /* Add in Bias */
        sum += net->bho[out];
        output_outputs[out] = sum;
    }
    softmax(output_outputs, net->num_outputs);
}

void NeuralNetwork::backpropagate(NeuralNetwork::Network *net, float *input, int *target, float *hidden_outputs, float *output_outputs, float learning_rate)
{
    // Calculate loss gradient
    float *loss_gradients = new float[net->num_outputs];
    for (int i = 0; i < net->num_outputs; i++)
    {
        loss_gradients[i] = output_outputs[i] - target[i]; // Corrected the sign
    }

    // Backpropagate through the output layer
    for (int out = 0; out < net->num_outputs; out++)
    {
        // Calculate gradients for output layer weights and biases
        for (int hid = 0; hid < net->num_hidden; hid++)
        {
            net->who[hid * net->num_outputs + out] -= learning_rate * loss_gradients[out] * hidden_outputs[hid];
        }
        net->bho[out] -= learning_rate * loss_gradients[out];
    }

    // Backpropagate through the hidden layer
    float *hidden_gradients = new float[net->num_hidden];
    for (int hid = 0; hid < net->num_hidden; hid++)
    {
        float sum = 0.0f;
        for (int out = 0; out < net->num_outputs; out++)
        {
            sum += loss_gradients[out] * net->who[hid * net->num_outputs + out];
        }
        hidden_gradients[hid] = sum * relu_derivative(hidden_outputs[hid]); // Using ReLU derivative
    }

    // Update hidden layer weights and biases
    for (int hid = 0; hid < net->num_hidden; hid++)
    {
        for (int inp = 0; inp < net->num_inputs; inp++)
        {
            net->wih[inp * net->num_hidden + hid] -= learning_rate * hidden_gradients[hid] * input[inp];
        }
        net->bih[hid] -= learning_rate * hidden_gradients[hid];
    }

    delete[] loss_gradients;
    delete[] hidden_gradients;
}

int NeuralNetwork::get_predicted_class(float *output_outputs, int num_outputs)
{
    int predicted_class = 0;
    float max_output = output_outputs[0];

    for (int i = 1; i < num_outputs; i++)
    {
        if (output_outputs[i] > max_output)
        {
            max_output = output_outputs[i];
            predicted_class = i;
        }
    }

    return predicted_class;
}

int NeuralNetwork::get_true_class(int *target, int num_outputs)
{
    for (int i = 0; i < num_outputs; i++)
    {
        if (target[i] == 1)
        {
            return i;
        }
    }
    // If no class with 1 is found, return -1 (error or not found)
    return -1;
}

void NeuralNetwork::init_network(Network *net, int num_inputs, int num_hidden, int num_outputs, DataReader::Dataset *data, float *wih, float *who, float *bih, float *bho)
{
    // Set the network architecture
    std::cout << std::endl
              << "Initializing network" << std::endl;
    net->num_inputs = num_inputs;
    net->num_hidden = num_hidden;
    net->num_outputs = num_outputs;
    net->train_dataset_size = data->trainSize;
    net->test_dataset_size = data->testSize;

    std::cout << "\n\nSize of train dataset: " << net->train_dataset_size << std::endl;
    std::cout << "Size of test data: " << net->test_dataset_size << std::endl;

    // Allocate memory and copy weights and biases
    int wih_size = num_inputs * num_hidden;
    net->wih = new float[wih_size];
    std::memcpy(net->wih, wih, wih_size * sizeof(float));

    int who_size = num_hidden * num_outputs;
    net->who = new float[who_size];
    std::memcpy(net->who, who, who_size * sizeof(float));

    std::memcpy(net->bih, bih, num_hidden * sizeof(float));
    std::memcpy(net->bho, bho, num_outputs * sizeof(float));

    std::cout << "Finished initializing network" << std::endl;
}

void NeuralNetwork::train_network(Network *net, DataReader::Dataset *data, int num_epochs, float learning_rate)
{
    const int BATCH_SIZE = 128;
    const int num_batches = (net->train_dataset_size + BATCH_SIZE - 1) / BATCH_SIZE;

    // Allocate memory for batch processing
    float *batch_hidden_outputs = new float[BATCH_SIZE * net->num_hidden];
    float *batch_final_outputs = new float[BATCH_SIZE * net->num_outputs];

    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
        int correct_predictions = 0;

        for (int batch = 0; batch < num_batches; batch++)
        {
            int batch_start = batch * BATCH_SIZE;
            int current_batch_size = std::min(BATCH_SIZE, net->train_dataset_size - batch_start);

            // Process each sample in the batch
            for (int i = 0; i < current_batch_size; i++)
            {
                int sample_idx = batch_start + i;
                float *input = &data->trainInputData[sample_idx * net->num_inputs];
                int *target = &data->trainTargetData[sample_idx * net->num_outputs];

                // Feedforward
                float *hidden_outputs = &batch_hidden_outputs[i * net->num_hidden];
                float *output_outputs = &batch_final_outputs[i * net->num_outputs];

                // Calculate input to hidden layer
                for (int hid = 0; hid < net->num_hidden; hid++)
                {
                    float sum = 0.0f;
                    for (int inp = 0; inp < net->num_inputs; inp++)
                    {
                        sum += input[inp] * net->wih[inp * net->num_hidden + hid];
                    }
                    sum += net->bih[hid];
                    hidden_outputs[hid] = relu(sum);
                }

                // Calculate hidden to output layer
                for (int out = 0; out < net->num_outputs; out++)
                {
                    float sum = 0.0f;
                    for (int hid = 0; hid < net->num_hidden; hid++)
                    {
                        sum += hidden_outputs[hid] * net->who[hid * net->num_outputs + out];
                    }
                    sum += net->bho[out];
                    output_outputs[out] = sum;
                }
                softmax(output_outputs, net->num_outputs);

                // Backpropagation
                backpropagate(net, input, target, hidden_outputs, output_outputs, learning_rate);

                // Track accuracy
                int predicted_class = get_predicted_class(output_outputs, net->num_outputs);
                int true_class = get_true_class(target, net->num_outputs);
                if (predicted_class == true_class)
                {
                    correct_predictions++;
                }
            }
        }

        float accuracy = (static_cast<float>(correct_predictions) / net->train_dataset_size) * 100.0f;
        std::cout << "Epoch " << epoch + 1 << " Training Accuracy: " << accuracy << "%" << std::endl;
    }

    delete[] batch_hidden_outputs;
    delete[] batch_final_outputs;
}

void NeuralNetwork::test_network(Network *net, DataReader::Dataset *data)
{
    int correct_predictions = 0;
    int total_samples = net->test_dataset_size;

    for (int i = 0; i < total_samples; i++)
    {
        float *input = &data->testInputData[i * net->num_inputs];
        int *target = &data->testOutputData[i * net->num_outputs];

        // Feedforward pass
        float hidden_outputs[net->num_hidden];
        float output_outputs[net->num_outputs];
        feedforward(net, input, hidden_outputs, output_outputs);

        int predicted_class = get_predicted_class(output_outputs, net->num_outputs);
        int true_class = get_true_class(target, net->num_outputs);

        if (predicted_class == true_class)
        {
            correct_predictions++;
        }
    }

    // Calculate accuracy
    float accuracy = ((float)correct_predictions / net->test_dataset_size) * 100.0;

    // Print the results
    std::cout << "Test accuracy: " << accuracy << "%" << std::endl;
}

NeuralNetwork::ConfusionMatrix NeuralNetwork::calculateConfusionMatrix(int y_true[], int y_pred[], int numInstances)
{
    ConfusionMatrix cm = {0, 0, 0, 0};

    for (int i = 0; i < numInstances; i++)
    {
        if (y_true[i] == 1 && y_pred[i] == 1)
        {
            cm.truePositive++;
        }
        else if (y_true[i] == 0 && y_pred[i] == 1)
        {
            cm.falsePositive++;
        }
        else if (y_true[i] == 0 && y_pred[i] == 0)
        {
            cm.trueNegative++;
        }
        else if (y_true[i] == 1 && y_pred[i] == 0)
        {
            cm.falseNegative++;
        }
    }

    return cm;
}

bool NeuralNetwork::compare_network(Network *network1, Network *network2)
{
    // Compare weights (wih)
    for (int i = 0; i < network1->num_inputs * network1->num_hidden; i++)
    {
        if (std::abs(network1->wih[i] - network2->wih[i]) > 1e-6)
            return false;
    }

    // Compare weights (who)
    for (int i = 0; i < network1->num_hidden * network1->num_outputs; i++)
    {
        if (std::abs(network1->who[i] - network2->who[i]) > 1e-6)
            return false;
    }

    // Compare biases (bih)
    for (int i = 0; i < network1->num_hidden; i++)
    {
        if (std::abs(network1->bih[i] - network2->bih[i]) > 1e-6)
            return false;
    }

    // Compare biases (bho)
    for (int i = 0; i < network1->num_outputs; i++)
    {
        if (std::abs(network1->bho[i] - network2->bho[i]) > 1e-6)
            return false;
    }

    return true;
}
