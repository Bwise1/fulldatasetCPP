#include "../includes/utils.hpp"
#include <stdio.h>
#include "../includes/network.hpp"
#include "data_reader.hpp"
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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

__global__ void softmax_kernel(float* arr, int size) {
    softmax_gpu(arr, size);
}
// feedforward_gpu kernel
__global__ void feedforward_gpu(
    const float* d_input,        // [num_inputs]
    const float* d_wih,          // [num_inputs][num_hidden]
    const float* d_who,          // [num_hidden][num_outputs]
    const float* d_bih,          // [num_hidden]
    const float* d_bho,          // [num_outputs]
    float* d_hidden_outputs,     // [num_hidden]
    float* d_output_outputs,     // [num_outputs]
    int num_inputs,
    int num_hidden,
    int num_outputs
) {
    const int hid = blockIdx.x * blockDim.x + threadIdx.x;
    const int out = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate hidden layer outputs
    if (hid < num_hidden) {
        float sum = 0.0f;
        for (int inp = 0; inp < num_inputs; inp++) {
            sum += d_input[inp] * d_wih[inp * num_hidden + hid];
        }
        sum += d_bih[hid];
        d_hidden_outputs[hid] = relu_gpu(sum);
    }

    __syncthreads();

    // Calculate output layer outputs
    if (out < num_outputs) {
        float sum = 0.0f;
        for (int h = 0; h < num_hidden; h++) {
            sum += d_hidden_outputs[h] * d_who[h * num_outputs + out];
        }
        sum += d_bho[out];
        d_output_outputs[out] = sum;
    }

    __syncthreads();

    // // Apply softmax
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     softmax_gpu(d_output_outputs, num_outputs);
    // }
}

// New kernels for backpropagation split into output and hidden layers
__global__ void backprop_output_gpu(
    const float* d_input,
    const int* target,
    const float* d_hidden_outputs,
    const float* d_output_outputs,
    float* d_who,
    float* d_bho,
    int num_inputs,
    int num_hidden,
    int num_outputs,
    float learning_rate
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= num_outputs) return;

    // Convert target to one-hot encoding
    float target_val = 0.0f;
    if (target[out_idx] == 1) target_val = 1.0f;

    const float output_grad = d_output_outputs[out_idx] - target_val;

    // Update output bias
    d_bho[out_idx] -= learning_rate * output_grad;

    // Update hidden-to-output weights
    for (int h = 0; h < num_hidden; h++) {
        d_who[h * num_outputs + out_idx] -= learning_rate * output_grad * d_hidden_outputs[h];
    }
}

__global__ void backprop_hidden_gpu(
    const float* d_input,
    const int* target,
    const float* d_hidden_outputs,
    const float* d_output_outputs,
    float* d_wih,
    float* d_bih,
    const float* d_who,
    int num_inputs,
    int num_hidden,
    int num_outputs,
    float learning_rate
) {
    int hid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hid_idx >= num_hidden) return;

    // Calculate hidden gradient
    float hidden_grad = 0.0f;
    for (int o = 0; o < num_outputs; o++) {
        float target_val = (target[o] == 1) ? 1.0f : 0.0f;
        hidden_grad += (d_output_outputs[o] - target_val) * d_who[hid_idx * num_outputs + o];
    }
    hidden_grad *= relu_derivative_gpu(d_hidden_outputs[hid_idx]);

    // Update hidden bias
    d_bih[hid_idx] -= learning_rate * hidden_grad;

    // Update input-to-hidden weights
    for (int i = 0; i < num_inputs; i++) {
        d_wih[i * num_hidden + hid_idx] -= learning_rate * hidden_grad * d_input[i];
    }
}
void NeuralNetwork::init_network_gpu(Network *net, DeviceNetwork *d_net) {
    printf("Initializing GPU network...\n");

    // Validate input pointers
    if (!net || !d_net) {
        fprintf(stderr, "Error: Null pointer passed (net=%p, d_net=%p)\n", net, d_net);
        exit(EXIT_FAILURE);
    }

    // Get network dimensions
    const int num_inputs = net->num_inputs;
    const int num_hidden = net->num_hidden;
    const int num_outputs = net->num_outputs;

    // Validate dimensions
    if (num_inputs <= 0 || num_hidden <= 0 || num_outputs <= 0) {
        fprintf(stderr, "Invalid network dimensions: inputs=%d, hidden=%d, outputs=%d\n",
                num_inputs, num_hidden, num_outputs);
        exit(EXIT_FAILURE);
    }

    // Verify host data integrity
    if (!net->wih || !net->who || !net->bih || !net->bho) {
        fprintf(stderr, "Host network contains uninitialized pointers\n");
        exit(EXIT_FAILURE);
    }

    // Initialize CUDA context
    CHECK_CUDA_ERROR(cudaFree(0));

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_net->d_wih, num_inputs * num_hidden * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_net->d_who, num_hidden * num_outputs * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_net->d_bih, num_hidden * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_net->d_bho, num_outputs * sizeof(float)));

    printf("GPU memory allocated successfully.\n");

    // Copy input-to-hidden weights
    float *h_wih_flat = new float[num_inputs * num_hidden];
    for (int i = 0; i < num_inputs; ++i) {
        if (!net->wih[i]) {
            fprintf(stderr, "Host weight matrix wih[%d] is null!\n", i);
            delete[] h_wih_flat;
            exit(EXIT_FAILURE);
        }
        memcpy(h_wih_flat + i * num_hidden, net->wih[i], num_hidden * sizeof(float));
    }
    CHECK_CUDA_ERROR(cudaMemcpy(
        d_net->d_wih,
        h_wih_flat,
        num_inputs * num_hidden * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    delete[] h_wih_flat;

    // Copy hidden-to-output weights
    float *h_who_flat = new float[num_hidden * num_outputs];
    for (int i = 0; i < num_hidden; ++i) {
        memcpy(h_who_flat + i * num_outputs, net->who[i], num_outputs * sizeof(float));
    }
    CHECK_CUDA_ERROR(cudaMemcpy(
        d_net->d_who,
        h_who_flat,
        num_hidden * num_outputs * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    delete[] h_who_flat;

    // Copy hidden layer biases
    CHECK_CUDA_ERROR(cudaMemcpy(
        d_net->d_bih,
        net->bih,
        num_hidden * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // Copy output layer biases
    CHECK_CUDA_ERROR(cudaMemcpy(
        d_net->d_bho,
        net->bho,
        num_outputs * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    printf("Network parameters successfully copied to GPU.\n");
}

void NeuralNetwork::free_network_gpu(DeviceNetwork *d_net) {
    CHECK_CUDA_ERROR(cudaFree(d_net->d_wih));
    CHECK_CUDA_ERROR(cudaFree(d_net->d_who));
    CHECK_CUDA_ERROR(cudaFree(d_net->d_bih));
    CHECK_CUDA_ERROR(cudaFree(d_net->d_bho));
}


// Modified training loop section
void NeuralNetwork::train_network_gpu(Network* net, DataReader::Dataset* data,
                                    int num_epochs, float learning_rate) {
    DeviceNetwork d_net;
    init_network_gpu(net, &d_net);
    printf("First 10 weights from input to hidden layer:\n");
    for (int i = 0; i < 10 && i < (net->num_inputs * net->num_hidden); i++) {
        printf("%.6f ", net->wih[0][i]);
    }
    printf("\n");

    float *d_input, *d_hidden, *d_output;
    int *d_target;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, net->num_inputs * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hidden, net->num_hidden * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, net->num_outputs * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_target, net->num_outputs * sizeof(int)));



    const int blockSize = 256;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        int correct = 0;

        for (int i = 0; i < net->train_dataset_size; i++) {
            // Copy input and target to device
            CHECK_CUDA_ERROR(cudaMemcpy(d_input, data->trainInputData[i],
                                      net->num_inputs * sizeof(float),
                                      cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_target, data->trainTargetData[i],
                                      net->num_outputs * sizeof(int),
                                      cudaMemcpyHostToDevice));

            // Feedforward
            dim3 block_feed(16, 16);
            int grid_hidden_x = (net->num_hidden + block_feed.x - 1) / block_feed.x;
            int grid_output_y = (net->num_outputs + block_feed.y - 1) / block_feed.y;
            dim3 grid_feed(grid_hidden_x, grid_output_y);

            feedforward_gpu<<<grid_feed, block_feed>>>(
                d_input,
                d_net.d_wih,
                d_net.d_who,
                d_net.d_bih,
                d_net.d_bho,
                d_hidden,
                d_output,
                net->num_inputs,
                net->num_hidden,
                net->num_outputs
            );
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            // Apply softmax
            softmax_kernel<<<1, 1>>>(d_output, net->num_outputs);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            // Backpropagation for output layer
            int gridSizeOutput = (net->num_outputs + blockSize - 1) / blockSize;
            backprop_output_gpu<<<gridSizeOutput, blockSize>>>(
                d_input,
                d_target,
                d_hidden,
                d_output,
                d_net.d_who,
                d_net.d_bho,
                net->num_inputs,
                net->num_hidden,
                net->num_outputs,
                learning_rate
            );
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            // Backpropagation for hidden layer
            int gridSizeHidden = (net->num_hidden + blockSize - 1) / blockSize;
            backprop_hidden_gpu<<<gridSizeHidden, blockSize>>>(
                d_input,
                d_target,
                d_hidden,
                d_output,
                d_net.d_wih,
                d_net.d_bih,
                d_net.d_who,
                net->num_inputs,
                net->num_hidden,
                net->num_outputs,
                learning_rate
            );
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            // Accuracy calculation (removed problematic cudaMemcpy)
            float* h_output = new float[net->num_outputs];
            int* h_target = new int[net->num_outputs];
            CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output,
                                      net->num_outputs * sizeof(float),
                                      cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(h_target, d_target,
                                      net->num_outputs * sizeof(int),
                                      cudaMemcpyDeviceToHost));

            int pred_class = 0;
            float max_prob = h_output[0];
            for (int c = 1; c < net->num_outputs; c++) {
                if (h_output[c] > max_prob) {
                    max_prob = h_output[c];
                    pred_class = c;
                }
            }

            if (h_target[pred_class] == 1) correct++;

            delete[] h_output;
            delete[] h_target;
        }

        printf("Epoch %d - Accuracy: %.2f%%\n",
              epoch, (float)correct / net->train_dataset_size * 100.0f);
    }

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_hidden));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_target));
    free_network_gpu(&d_net);
}

