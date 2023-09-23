#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdlib> // For rand() and RAND_MAX
#include <cmath>   // For exp() and other math functions

/* Inline functions */

// Generate a random weight between -0.5 and 0.5
inline float randWeight() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
}

// Generate a random float between 0 and 1
inline float getSRand() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

// Generate a random integer between 0 and x (exclusive)
inline int getRand(int x) {
    return static_cast<int>((static_cast<float>(x) * rand()) / (RAND_MAX + 1.0f));
}

// Calculate the square of a value
inline float sqr(float x) {
    return x * x;
}

// Sigmoid activation function
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Derivative of sigmoid function
inline float sigmoid_derivative(float val) {
    return sigmoid(val) * (1.0f - sigmoid(val));
}

// ReLU activation
inline float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

// Derivative of ReLU
inline float relu_derivative(float val) {
    return (val > 0.0f) ? 1.0f : 0.0f;
}

#endif // UTILS_HPP
