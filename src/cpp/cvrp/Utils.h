/*
 * CVRP Utilities Header
 * 
 * Declares utility functions for random number generation and array operations
 * used throughout the CVRP implementation.
 */

#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

// Generate a random integer in the range [min, max] (inclusive)
int getRandomNumber(int min, int max);

// Generate a random float in the range [min, max]
float getRandomFraction(float min = 0.0, float max = 1.0);

// Generate a random float between 0.0 and 1.0 using pre-generated values for speed
float getRandomFractionFast();

// Perform argsort on a vector of float values - returns indices sorted by values
std::vector<int> argsort(const std::vector<float>& values);

#endif // UTILS_H