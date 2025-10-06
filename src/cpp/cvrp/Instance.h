#pragma once
#ifndef INSTANCE_H
#define INSTANCE_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <tuple>
#include <unordered_set>
#include <iterator>
#include <numeric>
#include <fstream>
#include <string>

#include "Utils.h"

/**
 * CVRP Instance structure containing problem data
 */
struct Instance {
    // Basic problem parameters
    int numNodes;           // Total number of nodes (customers + depot)
    int numCustomers;       // Number of customers (excluding depot)
    int vehicleCapacity;    // Vehicle capacity constraint
    
    // Problem data
    std::vector<int> demand;                           // Demand for each node
    std::vector<std::vector<float>> distanceMatrix;   // Distance matrix between all nodes
    std::vector<std::vector<float>> nodePositions;    // 2D coordinates of each node
    std::vector<std::vector<int>> adj;                // Adjacency lists sorted by distance

    // Constructor
    Instance(int numCustomers, int vehicleCapacity, 
             const std::vector<int>& demand, 
             const std::vector<std::vector<float>>& nodePositions);
};

#endif // INSTANCE_H