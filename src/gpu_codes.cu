#include "../includes/utils.hpp"
#include <stdio.h>
#include "../includes/network.hpp"
#include "data_reader.hpp"
#include <cuda_runtime.h>


__device__ float relu_gpu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

__device__ void softmax_gpu(float *arr, int size) {
    float max_val = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_exp += expf(arr[i] - max_val);
    }

    for (int i = 0; i < size; i++) {
        arr[i] = expf(arr[i] - max_val) / sum_exp;
    }
}

 __global__ void feedforward_gpu(NeuralNetwork::Network* net, float* d_input, float* d_hidden_outputs, float* d_output_outputs) {
    int hid = blockIdx.x * blockDim.x + threadIdx.x;
    int out = blockIdx.y * blockDim.y + threadIdx.y;

    if (hid < net->num_hidden) {
        float sum = 0.0f;
        for (int inp = 0; inp < net->num_inputs; inp++) {
            sum += d_input[inp] * net->wih[inp][hid];
        }

        // Add in Bias
        sum += net->bih[hid];
        d_hidden_outputs[hid] = relu_gpu(sum);
    }

    __syncthreads();

    if (out < net->num_outputs) {
        float sum = 0.0f;
        for (int h = 0; h < net->num_hidden; h++) {
            sum += d_hidden_outputs[h] * net->who[h][out];
        }

        // Add in Bias
        sum += net->bho[out];
        d_output_outputs[out] = sum;
    }

    __syncthreads();

    if (out < net->num_outputs) {
        softmax_gpu(d_output_outputs, net->num_outputs);
    }
}

void NeuralNetwork::train_network_gpu(Network *net, DataReader::Dataset *data, int num_epochs, float learning_rate) {
    // Allocate GPU memory for the necessary variables
    float *d_input, *d_hidden_outputs, *d_output_outputs;
    int *d_target;

    cudaMalloc((void**)&d_input, sizeof(float) * net->train_dataset_size * net->num_inputs);
    cudaMalloc((void**)&d_hidden_outputs, sizeof(float) * net->train_dataset_size * net->num_hidden);
    cudaMalloc((void**)&d_output_outputs, sizeof(float) * net->train_dataset_size * net->num_outputs);
    cudaMalloc((void**)&d_target, sizeof(int) * net->train_dataset_size * net->num_outputs);

    // Define block size and calculate grid size based on the dataset size
    int blockSize = 256; // Adjust this based on your GPU's capabilities and workload
    int gridSize = (net->train_dataset_size + blockSize - 1) / blockSize;


    // Iterate over the specified number of epochs
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        int correct_predictions = 0;
        // float *hidden_outputs;
        // float *output_outputs;
         // Transfer data from CPU to GPU for each epoch
        cudaMemcpy(d_input, data->trainInputData, sizeof(float) * net->train_dataset_size * net->num_inputs, cudaMemcpyHostToDevice);
        cudaMemcpy(d_target, data->trainTargetData, sizeof(int) * net->train_dataset_size * net->num_outputs, cudaMemcpyHostToDevice);

        // Launch CUDA kernels for each data sample
        for (int i = 0; i < net->train_dataset_size; i++) {
            // Retrieve the input and target data for the current sample
            float *input = data->trainInputData[i];
            int *target = data->trainTargetData[i];

            // CUDA kernels for feedforward and backpropagation
            feedforward_gpu<<<gridSize, blockSize>>>(net, input, hidden_outputs, output_outputs);

            // Synchronize the GPU to ensure all kernel executions are completed
            cudaDeviceSynchronize();

            // Calculate training accuracy
            // Increment correct_predictions if predicted_class == true_class
        }

        // Copy computed results back to CPU memory

        // Training accuracy calculation using CPU
        // printf("Training Accuracy in Epoch %d: %.2%%n", epoch, accuracy);
    }

    // Free GPU memory
}


