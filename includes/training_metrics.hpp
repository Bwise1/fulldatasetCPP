// TrainingMetrics.h
#ifndef TRAINING_METRICS_H
#define TRAINING_METRICS_H

#include <vector>

struct TrainingMetrics
{
    int epoch;
    float accuracy;         // Accuracy percentage
    float kernel_time;      // Kernel execution time (ms)
    float data_copy_time;   // Data copy time (ms)
    float total_epoch_time; // Total epoch time (ms)
};

using TrainingMetricsVector = std::vector<TrainingMetrics>;

#endif
