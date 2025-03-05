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

// Modified kernel signature
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

    // Apply softmax
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        softmax_gpu(d_output_outputs, num_outputs);
    }
}

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

// void NeuralNetwork::train_network_gpu(Network* net, DataReader::Dataset* data, int num_epochs, float learning_rate) {
//     // Initialize DeviceNetwork
//    // cudaFree(0);
//     DeviceNetwork d_net;
//    // gpuErrchk(cudaMalloc(&d_net, sizeof(DeviceNetwork)));
//      printf("in train\n");
//     init_network_gpu(net, &d_net);
//     printf("in train\n");

//     // Allocate and copy training data to device
//     float *d_train_inputs, *d_train_targets;
//     size_t input_size = net->num_inputs * net->train_dataset_size * sizeof(float);
//     size_t target_size = net->num_outputs * net->train_dataset_size * sizeof(float);

//     // Convert targets to float and flatten
//     float* h_targets_flat = new float[net->num_outputs * net->train_dataset_size];
//     for(int i = 0; i < net->train_dataset_size; i++) {
//         for(int j = 0; j < net->num_outputs; j++) {
//             h_targets_flat[i * net->num_outputs + j] = static_cast<float>(data->trainTargetData[i][j]);
//         }
//     }

//     CHECK_CUDA_ERROR(cudaMalloc(&d_train_inputs, input_size));
//     CHECK_CUDA_ERROR(cudaMalloc(&d_train_targets, target_size));

//     // Flatten and copy inputs
//     float* h_inputs_flat = new float[net->num_inputs * net->train_dataset_size];
//     for(int i = 0; i < net->train_dataset_size; i++) {
//         memcpy(h_inputs_flat + i * net->num_inputs,
//             data->trainInputData[i],
//             net->num_inputs * sizeof(float));
//     }

//     CHECK_CUDA_ERROR(cudaMemcpy(d_train_inputs, h_inputs_flat, input_size, cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_train_targets, h_targets_flat, target_size, cudaMemcpyHostToDevice));
//     printf("Training Data copied to GPU\n");

//     for(int epoch = 0; epoch < num_epochs; epoch++) {
//         int correct = 0;
//         float total_loss = 0.0f;

//         for(int i = 0; i < net->train_dataset_size; i++) {
//             // Get current sample pointers
//             float* d_input = d_train_inputs + i * net->num_inputs;
//             float* d_target = d_train_targets + i * net->num_outputs;

//             // Allocate temporary device memory
//             float *d_hidden, *d_output;
//             CHECK_CUDA_ERROR(cudaMalloc(&d_hidden, net->num_hidden * sizeof(float)));
//             CHECK_CUDA_ERROR(cudaMalloc(&d_output, net->num_outputs * sizeof(float)));

//             // Launch feedforward kernel
//             dim3 block(16, 16);
//             dim3 grid_hidden((net->num_hidden + block.x - 1) / block.x, 1);
//             dim3 grid_output((net->num_outputs + block.y - 1) / block.y, 1);

//             printf("Launching feedforward kernel\n");
//             feedforward_gpu<<<grid_hidden, block>>>(net, &d_input, d_hidden, d_output);
//             CHECK_CUDA_ERROR(cudaGetLastError());
//             CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//             printf("finished feedforward kernel\n");

//             // Launch backpropagation kernel
//             backpropagate_gpu<<<grid_hidden, block>>>(net, d_input, reinterpret_cast<int*>(d_target),
//                                                     d_hidden, d_output, learning_rate);
//             CHECK_CUDA_ERROR(cudaGetLastError());
//             CHECK_CUDA_ERROR(cudaDeviceSynchronize());

//             // Calculate accuracy
//             float* h_output = new float[net->num_outputs];
//             float* h_target = new float[net->num_outputs];
//             CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, net->num_outputs * sizeof(float), cudaMemcpyDeviceToHost));
//             CHECK_CUDA_ERROR(cudaMemcpy(h_target, d_target, net->num_outputs * sizeof(float), cudaMemcpyDeviceToHost));

//             // Find predicted class
//             int pred_class = 0;
//             float max_val = h_output[0];
//             for(int c = 1; c < net->num_outputs; c++) {
//                 if(h_output[c] > max_val) {
//                     max_val = h_output[c];
//                     pred_class = c;
//                 }
//             }

//             // Check correctness
//             if(h_target[pred_class] == 1.0f) correct++;

//             // Cleanup
//             delete[] h_output;
//             delete[] h_target;
//             CHECK_CUDA_ERROR(cudaFree(d_hidden));
//             CHECK_CUDA_ERROR(cudaFree(d_output));
//         }

//         // Print epoch statistics
//         float accuracy = (float)correct / net->train_dataset_size * 100.0f;
//         printf("Epoch %d - Accuracy: %.2f%%\n", epoch, accuracy);
//     }
// }

void NeuralNetwork::train_network_gpu(Network* net, DataReader::Dataset* data,
                                    int num_epochs, float learning_rate) {
    // Initialize device network
    DeviceNetwork d_net;
    init_network_gpu(net, &d_net);

    // Allocate device memory for inputs/targets
    float *d_inputs, *d_targets;
    const size_t input_size = net->num_inputs * sizeof(float);
    const size_t target_size = net->num_outputs * sizeof(int);

    CHECK_CUDA_ERROR(cudaMalloc(&d_inputs, input_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_targets, target_size));

    // Training loop
    for(int epoch = 0; epoch < num_epochs; epoch++) {
        int correct = 0;
        printf("Epoch %d\n", epoch);
        for(int i = 0; i < net->train_dataset_size; i++) {
            // Copy current sample to device
            CHECK_CUDA_ERROR(cudaMemcpy(d_inputs, data->trainInputData[i],
                                      input_size, cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_targets, data->trainTargetData[i],
                                      target_size, cudaMemcpyHostToDevice));

            // Allocate temporary device buffers
            float *d_hidden, *d_output;
            CHECK_CUDA_ERROR(cudaMalloc(&d_hidden, net->num_hidden * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_output, net->num_outputs * sizeof(float)));

            // Configure kernel launch
            dim3 block(256);
            dim3 grid_hidden((net->num_hidden + block.x - 1) / block.x);
            dim3 grid_output((net->num_outputs + block.x - 1) / block.x);

            // Launch feedforward kernel
            feedforward_gpu<<<grid_hidden, block>>>(
                d_inputs,
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
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            // Launch backpropagation kernel
            // backpropagate_gpu<<<grid_hidden, block>>>(
            //     d_inputs,
            //     reinterpret_cast<int*>(d_targets),
            //     d_hidden,
            //     d_output,
            //     d_net.d_wih,
            //     d_net.d_who,
            //     d_net.d_bih,
            //     d_net.d_bho,
            //     net->num_inputs,
            //     net->num_hidden,
            //     net->num_outputs,
            //     learning_rate
            // );
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            // Cleanup
            CHECK_CUDA_ERROR(cudaFree(d_hidden));
            CHECK_CUDA_ERROR(cudaFree(d_output));
        }
    }

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_inputs));
    CHECK_CUDA_ERROR(cudaFree(d_targets));
    free_network_gpu(&d_net);
}
