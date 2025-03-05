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
