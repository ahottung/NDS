/*
 * PCVRP Tour Header
 * 
 * Defines the Tour structure for the Prize-Collecting Vehicle Routing Problem.
 * A tour represents a route that starts and ends at the depot.
 */

#ifndef TOUR_H
#define TOUR_H

#include <vector>
#include <algorithm>
#include "Instance.h"

/*
 * Tour structure representing a vehicle route for PCVRP
 * A tour is a sequence of customer nodes that starts and ends at the depot (node 0).
 * Each tour has an associated total demand and cost. In PCVRP, cost calculation
 * includes prize collection where prizes are subtracted from travel costs.
 */
struct Tour {
    std::vector<int> nodes;  // Sequence of customer nodes in the tour
    int demand = 0;          // Total demand of customers in this tour
    float costs = 0;         // Total cost of this tour (including prize collection)

    // Update the cost and demand of the tour based on the instance (includes prize collection)
    void updateCostAndDemand(const Instance& instance);

    // Remove a contiguous sequence of customers from the tour
    void stringRemoval(std::vector<int>& removed_cust, Tour& newTour, int lt, int c_star);

    // Remove customers in a non-contiguous pattern, preserving some in the middle
    void splitStringRemoval(std::vector<int>& removed_cust, Tour& newTour, int lt, int c_star, float alpha);
};

#endif // TOUR_H
