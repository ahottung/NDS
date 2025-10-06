/*
 * PCVRP Tour Implementation
 * 
 * Implementation of tour operations for the Prize-Collecting Vehicle Routing Problem.
 */

#include "Tour.h"

// Update tour cost and demand based on instance data (includes prize collection)
void Tour::updateCostAndDemand(const Instance& instance) {
    demand = 0;
    costs = 0;

    if (nodes.size() > 0) {
        // Cost from depot to first customer, minus prize collected
        costs += instance.distanceMatrix[0][nodes[0]] - instance.prizes[nodes[0]];
        demand += instance.demand[nodes[0]];

        // Cost between consecutive customers, minus prizes collected
        for (int j = 0; j < nodes.size() - 1; ++j) {
            costs += instance.distanceMatrix[nodes[j]][nodes[j + 1]] - instance.prizes[nodes[j + 1]];
            demand += instance.demand[nodes[j + 1]];
        }

        // Cost from last customer back to depot (no prize at depot)
        costs += instance.distanceMatrix[nodes.back()][0];
    }
}

// Remove a contiguous sequence of customers starting from a random position near c_star
void Tour::stringRemoval(std::vector<int>& removed_cust, Tour& newTour, int lt, int c_star) {
    // Find the position of the starting customer
    auto it = std::find(nodes.begin(), nodes.end(), c_star);
    int ctStarIdx = std::distance(nodes.begin(), it);

    // Calculate valid removal range
    int minIdx = std::max(0, ctStarIdx - lt);
    int maxIdx = std::min(ctStarIdx, (int)nodes.size() - lt);

    // Randomly select starting position for removal
    int start = getRandomNumber(minIdx, maxIdx);

    // Extract the removed customers
    removed_cust.assign(nodes.begin() + start, nodes.begin() + start + lt);

    // Create new tour excluding removed customers
    newTour.nodes.clear();
    std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(newTour.nodes),
        [&](int cust) { 
            return std::find(removed_cust.begin(), removed_cust.end(), cust) == removed_cust.end(); 
        });
}

// Remove customers in a non-contiguous pattern, preserving some in the middle
void Tour::splitStringRemoval(std::vector<int>& removed_cust, Tour& newTour, int lt, int c_star, float alpha) {
    // Find the position of the starting customer
    auto it = std::find(nodes.begin(), nodes.end(), c_star);
    int ctStarIdx = std::distance(nodes.begin(), it);

    // Determine how many customers to preserve in the middle
    int m_max = nodes.size() - lt;
    int m = 1;

    while (getRandomFraction() > alpha && m < m_max) {
        m += 1;
    }

    // Calculate total removal length including preserved customers
    int lt_m = lt + m;

    // Calculate valid removal range
    int minIdx = std::max(0, ctStarIdx - lt_m);
    int maxIdx = std::min(ctStarIdx, (int)nodes.size() - lt_m);

    // Randomly select starting position for removal
    int start = getRandomNumber(minIdx, maxIdx);

    // Calculate where to start preserving customers
    int preserve_start = (int)(start + lt / 2.0);

    // Extract removed customers (before and after preserved section)
    removed_cust.assign(nodes.begin() + start, nodes.begin() + preserve_start);
    removed_cust.insert(removed_cust.end(), 
                       nodes.begin() + preserve_start + m, 
                       nodes.begin() + start + lt_m);

    // Create new tour excluding removed customers
    newTour.nodes.clear();
    std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(newTour.nodes),
        [&](int cust) { 
            return std::find(removed_cust.begin(), removed_cust.end(), cust) == removed_cust.end(); 
        });
}