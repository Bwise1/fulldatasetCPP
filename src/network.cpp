#include <cmath> // Include cmath for math functions
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "../includes/network.hpp"
#include "../includes/utils.hpp"



/* Softmax function */
void NeuralNetwork::softmax(float *arr, int size) {
    float max_val = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_exp += std::exp(arr[i] - max_val);
    }

    for (int i = 0; i < size; i++) {
        arr[i] = std::exp(arr[i] - max_val) / sum_exp;
    }
}

void NeuralNetwork::feedforward(NeuralNetwork::Network *net, float *input, float *hidden_outputs, float *output_outputs) {
    int inp, hid, out;
    float sum;

    /* Calculate input to hidden layer */
    for (hid = 0 ; hid < net->num_hidden ; hid++) {
        sum = 0.0f;
        for (inp = 0 ; inp < net->num_inputs ; inp++) {
            sum += input[inp] * net->wih[inp][hid];
        }

        /* Add in Bias */
        sum += net->bih[hid];
        hidden_outputs[hid] = relu(sum);
    }

    /* Calculate the hidden to output layer */
    for (out = 0 ; out < net->num_outputs ; out++) {
        sum = 0.0f;
        for (hid = 0 ; hid < net->num_hidden ; hid++) {
            sum += hidden_outputs[hid] * net->who[hid][out];
        }
        /* Add in Bias */
        sum += net->bho[out];
        output_outputs[out] = sum;
    }
    softmax(output_outputs, net->num_outputs);
}

void NeuralNetwork::backpropagate(NeuralNetwork::Network *net, float *input, int *target, float *hidden_outputs, float *output_outputs, float learning_rate) {
    // Calculate loss gradient
    float loss_gradients[net->num_outputs];
    for (int i = 0; i < net->num_outputs; i++) {
        loss_gradients[i] = output_outputs[i] - target[i]; // Corrected the sign
    }

    // Backpropagate through the output layer
    for (int out = 0; out < net->num_outputs; out++) {
        // Calculate gradients for output layer weights and biases
        for (int hid = 0; hid < net->num_hidden; hid++) {
            net->who[hid][out] -= learning_rate * loss_gradients[out] * hidden_outputs[hid];
        }
        net->bho[out] -= learning_rate * loss_gradients[out];
    }

    // Backpropagate through the hidden layer
    float hidden_gradients[net->num_hidden];
    for (int hid = 0; hid < net->num_hidden; hid++) {
        float sum = 0.0f;
        for (int out = 0; out < net->num_outputs; out++) {
            sum += loss_gradients[out] * net->who[hid][out];
        }
        hidden_gradients[hid] = sum * relu_derivative(hidden_outputs[hid]); // Using ReLU derivative
    }

    // Update hidden layer weights and biases
    for (int hid = 0; hid < net->num_hidden; hid++) {
        for (int inp = 0; inp < net->num_inputs; inp++) {
            net->wih[inp][hid] -= learning_rate * hidden_gradients[hid] * input[inp];
        }
        net->bih[hid] -= learning_rate * hidden_gradients[hid];
    }
}

int NeuralNetwork::get_predicted_class(float *output_outputs, int num_outputs) {
    int predicted_class = 0;
    float max_output = output_outputs[0];

    for (int i = 0; i < num_outputs; i++) {
        if (output_outputs[i] > max_output) {
            max_output = output_outputs[i];
            predicted_class = i;
        }
    }

    return predicted_class;
}

int NeuralNetwork::get_true_class(int *target, int num_outputs) {
    for (int i = 0; i < num_outputs; i++) {
        if (target[i] == 1) {
            return i;
        }
    }
    // If no class with 1 is found, return -1 (error or not found)
    return -1;
}

void NeuralNetwork::init_network(Network *net, int num_inputs, int num_hidden, int num_outputs, DataReader::Dataset *data) {
    // Set the network architecture
    std::cout<<std::endl<<"Initializing network"<<std::endl;
    net->num_inputs = num_inputs;
    net->num_hidden = num_hidden;
    net->num_outputs = num_outputs;
    net->train_dataset_size = data->trainSize;
    net->test_dataset_size = data->testSize;

    std::cout<<"\n\nSize of train dataset"<<net->train_dataset_size<<std::endl;
    std::cout<<"\n\nSize of iut data"<<net->test_dataset_size<<std::endl;
    // Allocate memory for the input-to-hidden layer weights
    net->wih = new float*[num_inputs];
    for (int i = 0; i < num_inputs; i++) {
        net->wih[i] = new float[num_hidden];
        for (int j = 0; j < num_hidden; j++) {
            // Initialize weights with random values
            net->wih[i][j] = randWeight();
        }
    }

    // Allocate memory for the hidden-to-output layer weights
    net->who = new float*[num_hidden];
    for (int i = 0; i < num_hidden; i++) {
        net->who[i] = new float[num_outputs];
        for (int j = 0; j < num_outputs; j++) {
            // Initialize weights with random values
            net->who[i][j] = randWeight();
        }
    }

    // Allocate memory for the hidden layer biases
    net->bih = new float[num_hidden];
    for (int i = 0; i < num_hidden; i++) {
        // Initialize biases with random values
        net->bih[i] = randWeight();
    }

    // Allocate memory for the output layer biases
    net->bho = new float[num_outputs];
    for (int i = 0; i < num_outputs; i++) {
        // Initialize biases with random values
        net->bho[i] = randWeight();
    }
     std::cout<<"finished Initializing network"<<std::endl;
}

void NeuralNetwork::train_network(Network *net, DataReader::Dataset *data, int num_epochs, float learning_rate) {
    // Iterate over the dataset for the specified number of epochs
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Perform training for each data sample
        std::cout << "\nEpoch " <<epoch <<std::endl;

        int correct_predictions = 0;

        for (int i = 0; i < net->train_dataset_size; i++) {
            // Retrieve the input and target data for the current sample
            float *input = data->trainInputData[i];
            int *target = data->trainTargetData[i];

            // Feedforward pass
            float hidden_outputs[net->num_hidden];
            float output_outputs[net->num_outputs];
            feedforward(net, input, hidden_outputs, output_outputs);

            // Backpropagate
            backpropagate(net, input, target, hidden_outputs, output_outputs, learning_rate);

            // Calculate training accuracy
            int predicted_class = get_predicted_class(output_outputs, net->num_outputs);
            int true_class = get_true_class(target, net->num_outputs);

            if (predicted_class == true_class) {
                correct_predictions++;
            }
        }

        float accuracy = ((float)correct_predictions / net->train_dataset_size) * 100.0;
        std::cout<<"Training Accuracy in Epoch "<<epoch<<": "<<accuracy<<"%"<<std::endl;
    }
}

void NeuralNetwork::test_network(Network *net, DataReader::Dataset *data) {
    int correct_predictions = 0;
    int total_samples = net->test_dataset_size;

    for (int i = 0; i < total_samples; i++) {
        float *input = data->testInputData[i];
        int *target = data->testOutputData[i];

        // Feedforward pass
        float hidden_outputs[net->num_hidden];
        float output_outputs[net->num_outputs];
        feedforward(net, input, hidden_outputs, output_outputs);

        int predicted_class = get_predicted_class(output_outputs, net->num_outputs);
        int true_class = get_true_class(target, net->num_outputs);

        if (predicted_class == true_class) {
            correct_predictions++;
        }
    }

    // Calculate accuracy
    float accuracy = ((float)correct_predictions / net->test_dataset_size) * 100.0;

    // Print the results
    std::cout<<"Test accuracy: " <<accuracy<<"%"<<std::endl;
}

NeuralNetwork::ConfusionMatrix NeuralNetwork::calculateConfusionMatrix(int y_true[], int y_pred[], int numInstances) {
    ConfusionMatrix cm = {0, 0, 0, 0};

    for (int i = 0; i < numInstances; i++) {
        if (y_true[i] == 1 && y_pred[i] == 1) {
            cm.truePositive++;
        } else if (y_true[i] == 0 && y_pred[i] == 1) {
            cm.falsePositive++;
        } else if (y_true[i] == 0 && y_pred[i] == 0) {
            cm.trueNegative++;
        } else if (y_true[i] == 1 && y_pred[i] == 0) {
            cm.falseNegative++;
        }
    }

    return cm;
}

 bool NeuralNetwork::compare_network(Network* network1, Network* network2){
    return true;
 }
