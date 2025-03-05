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

void NeuralNetwork::train_network_gpu(Network* net, DataReader::Dataset* data, int num_epochs, float learning_rate) {
    // Initialize DeviceNetwork
   // cudaFree(0);
    DeviceNetwork d_net;
   // gpuErrchk(cudaMalloc(&d_net, sizeof(DeviceNetwork)));
     printf("in train\n");
    init_network_gpu(net, &d_net);
 printf("in train\n");
}
