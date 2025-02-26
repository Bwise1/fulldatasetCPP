#include "../includes/utils.hpp"
#include <stdio.h>
#include "../includes/network.hpp"
#include "data_reader.hpp"
#include <cuda_runtime.h>
#include <algorithm> // For std::min

__device__ float relu_gpu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

__device__ float relu_derivative_gpu(float x) {
    return (x > 0.0f) ? 1.0f : 0.0f;
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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void feedforward_gpu(NeuralNetwork::Network* net, float* d_input, float* d_hidden_outputs, float* d_output_outputs) {
    int hid = threadIdx.x;
    int out = threadIdx.y;

    if (hid < net->num_hidden) {
        float sum = 0.0f;
        for (int inp = 0; inp < net->num_inputs; inp++) {
            sum += d_input[inp] * net->wih[inp * net->num_hidden + hid];
        }

        // Add in Bias
        sum += net->bih[hid];
        d_hidden_outputs[hid] = relu_gpu(sum);
    }

    __syncthreads();

    if (out < net->num_outputs) {
        float sum = 0.0f;
        for (int h = 0; h < net->num_hidden; h++) {
            sum += d_hidden_outputs[h] * net->who[h * net->num_outputs + out];
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

__global__ void backpropagate_gpu(NeuralNetwork::Network* net, float *d_input, int *target, float *d_hidden_outputs, float *d_output_outputs, float learning_rate) {
    int hid = threadIdx.x;
    int out = threadIdx.y;

    if (out < net->num_outputs) {
        // Calculate loss gradient
        float loss_gradient = d_output_outputs[out] - target[out];

        // Update output layer weights and biases
        if (hid < net->num_hidden) {
            atomicAdd(&net->who[hid * net->num_outputs + out], -learning_rate * loss_gradient * d_hidden_outputs[hid]);
        }

        __syncthreads();

        // Update output layer biases
        atomicAdd(&net->bho[out], -learning_rate * loss_gradient);
    }

    __syncthreads();

    if (hid < net->num_hidden) {
        // Initialize hidden gradient
        float hidden_gradient = 0.0f;

        for (int o = 0; o < net->num_outputs; o++) {
            // Accumulate contributions from the output layer neurons
            hidden_gradient += (d_output_outputs[o] - target[o]) * net->who[hid * net->num_outputs + o];
        }

        // Apply the ReLU derivative
        hidden_gradient *= relu_derivative_gpu(d_hidden_outputs[hid]);

        // Update hidden layer weights and biases
        for (int inp = 0; inp < net->num_inputs; inp++) {
            atomicAdd(&net->wih[inp * net->num_hidden + hid], -learning_rate * hidden_gradient * d_input[inp]);
        }

        // Update hidden layer biases
        atomicAdd(&net->bih[hid], -learning_rate * hidden_gradient);
    }
}

void NeuralNetwork::train_network_gpu(Network *net, DataReader::Dataset *data, int num_epochs, float learning_rate) {
    // Allocate GPU memory for the necessary variables
    float *d_input, *d_hidden_outputs, *d_output_outputs, *d_output_data;
    int *d_target;

    gpuErrchk(cudaMalloc((void**)&d_input, sizeof(float) * net->num_inputs));
    gpuErrchk(cudaMalloc((void**)&d_hidden_outputs, sizeof(float) * net->num_hidden));
    gpuErrchk(cudaMalloc((void**)&d_output_outputs, sizeof(float) * net->num_outputs));
    gpuErrchk(cudaMalloc((void**)&d_target, sizeof(int) * net->num_outputs));
    // cudaMalloc((void**)&d_output_data, sizeof(float) * net->num_outputs);

    // Define block size and grid size
    dim3 block_dim(net->num_hidden, net->num_outputs);
    dim3 grid_dim(1, 1);

    // Iterate over the specified number of epochs
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("Epoch %d\n", epoch);
        int correct_predictions = 0;

        for (int sample = 0; sample < net->train_dataset_size; sample++) {
            // Prepare single sample data
            float* sample_input = new float[net->num_inputs];
            int* sample_target = new int[net->num_outputs];

            // Copy data from dataset to sample_input and sample_target
            std::copy(&data->trainInputData[sample * net->num_inputs],
                      &data->trainInputData[(sample + 1) * net->num_inputs],
                      sample_input);
            std::copy(&data->trainTargetData[sample * net->num_outputs],
                      &data->trainTargetData[(sample + 1) * net->num_outputs],
                      sample_target);

            // Copy data to GPU
            gpuErrchk(cudaMemcpy(d_input, sample_input, sizeof(float) * net->num_inputs, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_target, sample_target, sizeof(int) * net->num_outputs, cudaMemcpyHostToDevice));

            // Launch feedforward kernel
            feedforward_gpu<<<grid_dim, block_dim>>>(net, d_input, d_hidden_outputs, d_output_outputs);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());

            // Launch backpropagation kernel
            backpropagate_gpu<<<grid_dim, block_dim>>>(net, d_input, d_target, d_hidden_outputs, d_output_outputs, learning_rate);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());

            // Clean up
            delete[] sample_input;
            delete[] sample_target;
        }

        // Optionally, evaluate accuracy after each epoch
        // ...
    }

    // Free GPU memory
    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_hidden_outputs));
    gpuErrchk(cudaFree(d_output_outputs));
    gpuErrchk(cudaFree(d_target));
    // gpuErrchk(cudaFree(d_output_data));
}
