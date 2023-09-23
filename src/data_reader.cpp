#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cerrno>
#include "../includes/data_reader.hpp"

#define MAX_LINE_LENGTH 3054
#define NUM_SAMPLES 42000
#define INPUT_DIMENSION 784
#define TARGET_DIMENSION 10

namespace DataReader {

void shuffle(float** X, int** Y, int num_samples) {
    std::srand(time(nullptr));
    for (int i = num_samples - 1; i >= 0; i--) {
        int j = std::rand() % (i + 1);

        // Swap X[i] and X[j]
        float* tempX = X[i];
        X[i] = X[j];
        X[j] = tempX;

        // Swap Y[i] and Y[j]
        int* tempY = Y[i];
        Y[i] = Y[j];
        Y[j] = tempY;
    }
}

void splitData(float percent, float** X, int** Y, Dataset* dataset) {
    int trainSize = static_cast<int>(percent * NUM_SAMPLES);
    int testSize = NUM_SAMPLES - trainSize;

    dataset->trainSize = trainSize;
    dataset->testSize = testSize;

    // Allocate memory for train and test arrays
    dataset->trainInputData = new float*[ dataset->trainSize];
    dataset->trainTargetData = new int*[ dataset->trainSize];
    dataset->testInputData = new float*[ dataset->testSize];
    dataset->testOutputData = new int*[ dataset->testSize];

    // Populate train arrays
    for (int i = 0; i <  dataset->trainSize; i++) {
        dataset->trainInputData[i] = X[i];
        dataset->trainTargetData[i] = Y[i];
    }

    // Populate test arrays
    for (int i = 0; i <  dataset->testSize; i++) {
        dataset->testInputData[i] = X[i +  dataset->testSize];
        dataset->testOutputData[i] = Y[i +  dataset->testSize];
    }
}

Dataset* readDataFiles() {

    std::ifstream trainFile("./data/train.txt");
    char line[MAX_LINE_LENGTH];

    if (!trainFile) {
        std::cerr << "Error opening the train file." << std::endl;
        return nullptr;
    }

    float** x_train = new float*[NUM_SAMPLES];
    for (int i = 0; i < NUM_SAMPLES; i++) {
        x_train[i] = new float[INPUT_DIMENSION];
    }

    int** y_train = new int*[NUM_SAMPLES];
    for (int i = 0; i < NUM_SAMPLES; i++) {
        y_train[i] = new int[TARGET_DIMENSION];
    }

    int sample_count = 0;

    while (trainFile.getline(line, sizeof(line)) && sample_count < NUM_SAMPLES) {
        char* token = strtok(line, "\t");

        unsigned digit = atoi(token);
        for (unsigned i = 0; i < TARGET_DIMENSION; ++i) {
            y_train[sample_count][i] = (i == digit) ? 1 : 0;
        }

        unsigned feature_idx = 0;
        while ((token = strtok(NULL, "\t")) != NULL && feature_idx < INPUT_DIMENSION) {
            x_train[sample_count][feature_idx] = atof(token);
            feature_idx++;
        }

        sample_count++;
    }

    trainFile.close();

    // Normalize feature values to [0, 1]
    for (int i = 0; i < sample_count; ++i) {
        for (int j = 0; j < INPUT_DIMENSION; ++j) {
            x_train[i][j] /= 255.0;
        }
    }

    Dataset* dataset = new Dataset;
    dataset->datasetSize = sample_count;
     std::cout<<"\n\nSample count: "<< dataset->datasetSize<<std::endl;
    shuffle(x_train, y_train, NUM_SAMPLES);
    splitData(0.8f, x_train, y_train, dataset);

    return dataset;
}

} // namespace DataReader
