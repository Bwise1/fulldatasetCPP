#ifndef DATA_READER_HPP
#define DATA_READER_HPP

namespace DataReader {

    struct Dataset {
        float** trainInputData;
        int** trainTargetData;
        float** testInputData;
        int** testOutputData;
        int datasetSize;
        int trainSize;
        int testSize;
    };

    Dataset* readDataFiles();
    void shuffle(float** X, int** Y, int num_samples);
    void splitData(float percent, float** X, int** Y, Dataset* dataset);

} // namespace DataReader

#endif /* DATA_READER_HPP */
