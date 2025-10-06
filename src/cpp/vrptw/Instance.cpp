#include "Instance.h"

/**
 * VRPTW Instance constructor
 */
Instance::Instance(int numCustomers, int vehicleCapacity, 
                   const std::vector<int>& demand, 
                   const std::vector<float>& startTW, 
                   const std::vector<float>& endTW, 
                   const std::vector<float>& serviceTime, 
                   const std::vector<std::vector<float>>& nodePositions)
    : numCustomers(numCustomers), vehicleCapacity(vehicleCapacity), 
      demand(demand), startTW(startTW), endTW(endTW), 
      serviceTime(serviceTime), nodePositions(nodePositions)
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

    // Calculate time window width for each node
    TW_Width.resize(numNodes);
    for (int i = 0; i < numNodes; ++i) {
        TW_Width[i] = endTW[i] - startTW[i];
    }

    // Validate time window constraints
    // Ensure customers can return to depot within depot's time window
    for (int i = 1; i < numNodes; ++i) {
        float returnTime = endTW[i] + distanceMatrix[i][0] + serviceTime[i];
        if (returnTime > endTW[0]) {
            std::cout << "Customer " << i << " cannot return to depot in time (depot closes at " << endTW[0] << ")" << std::endl;
            throw std::invalid_argument("Time window end of customers need to be adjusted so that the depot is reached in time!");
        }
    }
}

