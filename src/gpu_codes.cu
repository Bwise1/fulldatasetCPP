#include "../includes/utils.hpp"
#include <stdio.h>
#include "../includes/network.hpp"
#include "data_reader.hpp"
#include <cuda_runtime.h>

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

 __global__ void feedforward_gpu(NeuralNetwork::Network* net, float** d_input, float* d_hidden_outputs, float* d_output_outputs) {
    int hid = blockIdx.x * blockDim.x + threadIdx.x;
    int out = blockIdx.y * blockDim.y + threadIdx.y;

    if (hid < net->num_hidden) {
        float sum = 0.0f;
        for (int inp = 0; inp < net->num_inputs; inp++) {
            sum += d_input[inp][hid] * net->wih[inp][hid];
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
//  __global__ void feedforward_gpu(NeuralNetwork::Network* net, float** d_input, float* d_hidden_outputs, float* d_output_outputs) {
//     int hid = blockIdx.x * blockDim.x + threadIdx.x;
//     int out = blockIdx.y * blockDim.y + threadIdx.y;

//     if (hid < net->num_hidden) {
//         float sum = 0.0f;
//         for (int inp = 0; inp < net->num_inputs; inp++) {
//             sum += d_input[inp][hid] * net->wih[inp][hid];

//         }
//         //  printf("%f ",sum);

//         // Add in Bias
//         sum += net->bih[hid];
//         d_hidden_outputs[hid] = relu_gpu(sum);
//     }

//     __syncthreads();

//     if (out < net->num_outputs) {
//         float sum = 0.0f;
//         for (int h = 0; h < net->num_hidden; h++) {
//             sum += d_hidden_outputs[h] * net->who[h][out];
//         }

//         // Add in Bias
//         sum += net->bho[out];
//         d_output_outputs[out] = sum;
//     }

//     __syncthreads();

//     if (out < net->num_outputs) {
//         softmax_gpu(d_output_outputs, net->num_outputs);
//     }
// }

// __global__ void backpropagate_gpu(NeuralNetwork::Network* net, float* d_input, int* target, float* d_hidden_outputs, float* d_output_outputs, float* d_hidden_error, float* d_output_error, float learning_rate) {
//     int hid = blockIdx.x * blockDim.x + threadIdx.x;
//     int out = blockIdx.y * blockDim.y + threadIdx.y;

//     if (out < net->num_outputs) {
//         // Calculate loss gradient for the output layer
//         float loss_gradient = d_output_outputs[out] - target[out];

//         if (hid < net->num_hidden) {
//             // Update output layer weights and biases
//             atomicAdd(&net->who[hid][out], -learning_rate * loss_gradient * d_hidden_outputs[hid]);
//         }

//         __syncthreads();

//         if (hid < net->num_hidden) {
//             // Update output layer biases
//             atomicAdd(&net->bho[out], -learning_rate * loss_gradient);
//         }
//     }

//     __syncthreads();

//     if (hid < net->num_hidden) {
//         float hidden_gradient = 0.0f;
//         for (int o = 0; o < net->num_outputs; o++) {
//             hidden_gradient += (out < net->num_outputs) ? (d_output_error[o] * net->who[hid][o]) : 0.0f;
//         }
//         hidden_gradient *= relu_derivative_gpu(d_hidden_outputs[hid]);

//         // Update hidden layer weights and biases
//         for (int inp = 0; inp < net->num_inputs; inp++) {
//             atomicAdd(&net->wih[inp][hid], -learning_rate * hidden_gradient * d_input[inp]);
//         }

//         __syncthreads();

//         // Update hidden layer biases
//         atomicAdd(&net->bih[hid], -learning_rate * hidden_gradient);
//     }
// }
// __global__ void backpropagate_gpu(NeuralNetwork::Network* net, float *d_input, int *target, float *d_hidden_outputs, float *d_output_outputs, float learning_rate) {
//     int hid = blockIdx.x * blockDim.x + threadIdx.x;
//     int out = blockIdx.y * blockDim.y + threadIdx.y;

//     if (out < net->num_outputs) {
//         // Calculate loss gradient
//         float loss_gradient = d_output_outputs[out] - target[out];

//         // Update output layer weights and biases
//         if (hid < net->num_hidden) {
//             for (int inp = 0; inp < net->num_inputs; inp++) {
//                 atomicAdd(&net->who[inp][out], -learning_rate * loss_gradient * d_hidden_outputs[hid]);
//             }
//         }

//         __syncthreads();

//         // Update output layer biases
//         atomicAdd(&net->bho[out], -learning_rate * loss_gradient);
//     }

//     __syncthreads();

//     if (hid < net->num_hidden) {
//         // Calculate hidden layer gradients
//         float hidden_gradient = 0.0f;
//         for (int o = 0; o < net->num_outputs; o++) {
//             hidden_gradient += (out < net->num_outputs) ? (loss_gradient * net->who[hid][out]) : 0.0;
//         }
//         hidden_gradient *= relu_derivative_gpu(d_hidden_outputs[hid]); // Use GPU ReLU derivative

//         // Update hidden layer weights and biases
//         for (int inp = 0; inp < net->num_inputs; inp++) {
//             atomicAdd(&net->wih[inp][hid], -learning_rate * hidden_gradient * d_input[inp]);
//         }

//         __syncthreads();

//         // Update hidden layer biases
//         atomicAdd(&net->bih[hid], -learning_rate * hidden_gradient);
//     }
// }

__global__ void backpropagate_gpu(NeuralNetwork::Network* net, float *d_input, int *target, float *d_hidden_outputs, float *d_output_outputs, float learning_rate) {
    int hid = blockIdx.x * blockDim.x + threadIdx.x;
    int out = blockIdx.y * blockDim.y + threadIdx.y;

    if (out < net->num_outputs) {
        // Calculate loss gradient
        float loss_gradient = d_output_outputs[out] - target[out];

        // Update output layer weights and biases
        if (hid < net->num_hidden) {
            for (int inp = 0; inp < net->num_inputs; inp++) {
                atomicAdd(&net->who[inp][out], -learning_rate * loss_gradient * d_hidden_outputs[hid]);
            }
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
            hidden_gradient += (d_output_outputs[o] - target[o]) * net->who[hid][o];
        }

        // Apply the ReLU derivative
        hidden_gradient *= relu_derivative_gpu(d_hidden_outputs[hid]);

        // Update hidden layer weights and biases
        for (int inp = 0; inp < net->num_inputs; inp++) {
            atomicAdd(&net->wih[inp][hid], -learning_rate * hidden_gradient * d_input[inp]);
        }

        // Update hidden layer biases
        atomicAdd(&net->bih[hid], -learning_rate * hidden_gradient);
    }
}


void NeuralNetwork::train_network_gpu(Network *net, DataReader::Dataset *data, int num_epochs, float learning_rate) {
    // Allocate GPU memory for the necessary variables
    float *d_input, *d_hidden_outputs, *d_output_outputs, *d_output_data;
    int *d_target;

    gpuErrchk(cudaMalloc((void**)&d_input, sizeof(float) * net->train_dataset_size * net->num_inputs));
    gpuErrchk(cudaMalloc((void**)&d_hidden_outputs, sizeof(float) * net->train_dataset_size * net->num_hidden));
    gpuErrchk(cudaMalloc((void**)&d_output_outputs, sizeof(float) * net->train_dataset_size * net->num_outputs));
    gpuErrchk(cudaMalloc((void**)&d_target, sizeof(int) * net->train_dataset_size * net->num_outputs));
    // cudaMalloc((void**)&d_output_data, sizeof(float) * net->train_dataset_size * net->num_outputs);

    // Define block size and calculate grid size based on the dataset size
    int blockSize = 256; // Adjust this based on your GPU's capabilities and workload
    int gridSize = (net->train_dataset_size + blockSize - 1) / blockSize;

    float** trainInputData = data->trainInputData;
    gpuErrchk(cudaMemcpy(d_input, trainInputData, sizeof(float) * net->train_dataset_size * net->num_inputs, cudaMemcpyDeviceToHost));


    // Iterate over the specified number of epochs
    for (int epoch = 0; epoch < 1; epoch++) {
         printf("Epoch %d",epoch);
        int correct_predictions = 0;

        // CUDA kernels for feedforward and backpropagation
        feedforward_gpu<<<gridSize, blockSize>>>(net, trainInputData, d_hidden_outputs, d_output_outputs);
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(cudaError));
            // Handle or exit the program if there's an error
        }
        // Synchronize the GPU to ensure all kernel executions are completed
        cudaDeviceSynchronize();
        // backpropagate_gpu<<<gridSize, blockSize>>>(net, d_input, d_target, d_hidden_outputs, d_output_outputs, learning_rate);


        // Copy computed results back to CPU memory
        gpuErrchk(cudaMemcpy(d_output_data, d_output_outputs, sizeof(float) * net->train_dataset_size * net->num_outputs, cudaMemcpyDeviceToHost));
        // free(d_hidden_outputs);
        printf("%f ", d_output_data[0]);
        // // Training accuracy calculation using CPU
        // for (int i = 0; i < net->train_dataset_size; i++) {
        //     float *output = &d_output_data[i * net->num_outputs];
        //     int predicted_class = get_predicted_class(output, net->num_outputs);

        //     int true_class = get_true_class(data->trainTargetData[i], net->num_outputs);
        //     if (predicted_class == true_class) {
        //         correct_predictions++;
        //     }
        // }
        // double accuracy = ((double) correct_predictions / net->train_dataset_size) * 100.0;
        // printf("Training Accuracy in Epoch %d: %.2f%%n", epoch, accuracy);
    }

    // Free GPU memory
}


