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

namespace DataReader
{

    void shuffle(float *X, int *Y, int num_samples, int input_dim, int target_dim)
    {
        std::srand(time(nullptr));
        for (int i = num_samples - 1; i > 0; i--)
        {
            int j = std::rand() % (i + 1);

            // Swap X[i] and X[j]
            for (int k = 0; k < input_dim; ++k)
            {
                std::swap(X[i * input_dim + k], X[j * input_dim + k]);
            }

            // Swap Y[i] and Y[j]
            for (int k = 0; k < target_dim; ++k)
            {
                std::swap(Y[i * target_dim + k], Y[j * target_dim + k]);
            }
        }
    }

    void splitData(float percent, float *X, int *Y, Dataset *dataset, int input_dim, int target_dim)
    {
        int trainSize = static_cast<int>(percent * NUM_SAMPLES);
        int testSize = NUM_SAMPLES - trainSize;

        dataset->trainSize = trainSize;
        dataset->testSize = testSize;

        // Allocate memory for train and test arrays
        dataset->trainInputData = new float[trainSize * input_dim];
        dataset->trainTargetData = new int[trainSize * target_dim];
        dataset->testInputData = new float[testSize * input_dim];
        dataset->testOutputData = new int[testSize * target_dim];

        // Populate train arrays
        std::memcpy(dataset->trainInputData, X, trainSize * input_dim * sizeof(float));
        std::memcpy(dataset->trainTargetData, Y, trainSize * target_dim * sizeof(int));

        // Populate test arrays
        std::memcpy(dataset->testInputData, X + trainSize * input_dim, testSize * input_dim * sizeof(float));
        std::memcpy(dataset->testOutputData, Y + trainSize * target_dim, testSize * target_dim * sizeof(int));
    }

    Dataset *readDataFiles()
    {
        std::ifstream trainFile("./data/train.txt");
        char line[MAX_LINE_LENGTH];

        if (!trainFile)
        {
            std::cerr << "Error opening the train file." << std::endl;
            return nullptr;
        }

        float *x_train;
        int *y_train;

        x_train = new float[NUM_SAMPLES * INPUT_DIMENSION];
        y_train = new int[NUM_SAMPLES * TARGET_DIMENSION];

        int sample_count = 0;

        while (trainFile.getline(line, sizeof(line)) && sample_count < NUM_SAMPLES)
        {
            char *token = strtok(line, "\t");

            unsigned digit = atoi(token);
            for (unsigned i = 0; i < TARGET_DIMENSION; ++i)
            {
                y_train[sample_count * TARGET_DIMENSION + i] = (i == digit) ? 1 : 0;
            }

            unsigned feature_idx = 0;
            while ((token = strtok(NULL, "\t")) != NULL && feature_idx < INPUT_DIMENSION)
            {
                x_train[sample_count * INPUT_DIMENSION + feature_idx] = atof(token);
                feature_idx++;
            }

            sample_count++;
        }

        trainFile.close();

        // Normalize feature values to [0, 1]
        for (int i = 0; i < sample_count * INPUT_DIMENSION; ++i)
        {
            x_train[i] /= 255.0f;
        }

        // Shuffle data
        shuffle(x_train, y_train, NUM_SAMPLES, INPUT_DIMENSION, TARGET_DIMENSION);

        Dataset *dataset = new Dataset;
        dataset->datasetSize = sample_count;
        std::cout << "\n\nSample count: " << dataset->datasetSize << std::endl;

        // Split data
        splitData(0.8f, x_train, y_train, dataset, INPUT_DIMENSION, TARGET_DIMENSION);

        return dataset;
    }

} // namespace DataReader
