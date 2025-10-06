#include "Instance.h"

/**
 * CVRP Instance constructor
 */
Instance::Instance(int numCustomers, int vehicleCapacity, 
                   const std::vector<int>& demand, 
                   const std::vector<std::vector<float>>& nodePositions)
    : numCustomers(numCustomers), vehicleCapacity(vehicleCapacity), 
      demand(demand), nodePositions(nodePositions)
{
    // Calculate total number of nodes (customers + depot)
    numNodes = numCustomers + 1;

    // Validate input data
    if (demand.size() != numNodes) {
        throw std::invalid_argument("Size of demand vector does not match number of customers!");
    }

    // Initialize distance matrix
    distanceMatrix.resize(numNodes, std::vector<float>(numNodes, 0.0f));

    // Calculate Euclidean distances between all node pairs
    for (int i = 0; i < numNodes; ++i) {
        for (int j = 0; j < numNodes; ++j) {
            float dx = nodePositions[i][0] - nodePositions[j][0];
            float dy = nodePositions[i][1] - nodePositions[j][1];
            distanceMatrix[i][j] = std::sqrt(dx * dx + dy * dy);
        }
    }

    // Build adjacency lists sorted by distance for each node
    for (int i = 0; i < numNodes; ++i) {
        adj.push_back(argsort(distanceMatrix[i]));
    }
}
