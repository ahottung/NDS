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
#include <sstream>

#include "Utils.h"

/**
 * VRPTW Instance structure containing problem data
 * Vehicle Routing Problem with Time Windows variant
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
    
    // Time window specific data
    std::vector<float> startTW;    // Start time of time window for each node
    std::vector<float> endTW;      // End time of time window for each node
    std::vector<float> TW_Width;  // Width of time window for each node
    std::vector<float> serviceTime; // Service time required at each node

    // Constructor
    Instance(int numCustomers, int vehicleCapacity, 
             const std::vector<int>& demand, 
             const std::vector<float>& startTW,
             const std::vector<float>& endTW, 
             const std::vector<float>& serviceTime, 
             const std::vector<std::vector<float>>& nodePositions);
};

#endif // INSTANCE_H