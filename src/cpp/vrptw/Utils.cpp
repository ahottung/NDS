/*
 * VRPTW Utilities Implementation
 * 
 * Implementation of utility functions for random number generation and array operations.
 */

#include "Utils.h"

// Generate a random integer in the range [min, max] (inclusive)
int getRandomNumber(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}

// Generate a random float in the range [min, max]
float getRandomFraction(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

// Generate a random float between 0.0 and 1.0 using pre-generated values for speed
float getRandomFractionFast() {
    static std::vector<float> randomValues;
    static size_t index = 0;
    static const size_t poolSize = 10000;

    // Initialize the random value pool if empty
    if (randomValues.empty()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        // Pre-generate random values
        randomValues.reserve(poolSize);
        for (size_t i = 0; i < poolSize; ++i) {
            randomValues.push_back(dist(gen));
        }
    }

    // Get the next random value from the pool
    float randomValue = randomValues[index];
    index = (index + 1) % poolSize; // Wrap around when reaching the end

    return randomValue;
}

// Perform argsort on a vector of float values - returns indices sorted by values
std::vector<int> argsort(const std::vector<float>& values) {
    // Create an index vector [0, 1, 2, ..., n-1]
    std::vector<int> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort the indices based on the corresponding values
    std::sort(indices.begin(), indices.end(), [&values](size_t i1, size_t i2) {
        return values[i1] < values[i2];
    });

    return indices;
}
