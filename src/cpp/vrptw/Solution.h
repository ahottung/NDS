/*
 * VRPTW Solution Header
 * 
 * Defines the Solution and ModifiedSolution structures for the
 * Vehicle Routing Problem with Time Windows.
 */

#ifndef SOLUTION_H
#define SOLUTION_H

#include "Instance.h"
#include "Tour.h"

// Forward declarations
struct Solution;
struct ModifiedSolution;

/*
 * VRPTW Solution structure
 * Represents a complete solution containing tours that visit all customers
 * while respecting capacity constraints and time window constraints.
 */
struct Solution {
    const Instance& instance;           // Reference to the problem instance
    float totalCosts;                   // Total cost of the solution
    std::vector<Tour> tours;            // List of tours in the solution
    std::vector<int> customerToTourMap; // Mapping from customer to tour index

    // Constructors
    Solution(const Instance& instance);
    Solution(const Instance& instance, const std::vector<std::vector<int>>& tours);

    // Solution modification
    void acceptModifiedSolution(ModifiedSolution& modSol);
    void generateCustomerToTourMap();

    // Utility functions
    std::vector<std::vector<int>> getTourList() const;

    // Assignment operator
    Solution& operator=(const Solution& other);
};

/*
 * Modified Solution structure for VRPTW
 * Represents a temporary modification to a solution during local search.
 * Tracks removed tours and new tours with consideration for time window constraints.
 */
struct ModifiedSolution {
    Solution& originalSolution;         // Reference to the original solution
    const Instance& instance;           // Reference to the problem instance
    float totalCosts;                   // Total cost of the modified solution
    std::vector<int> removedToursId;    // Indices of tours to be removed
    std::vector<Tour> newTours;         // New tours to be added

    // Constructor
    ModifiedSolution(Solution& originalSolution);

    // Modification operations
    void removeCustomers(const std::vector<int>& A);
    std::unordered_set<int> destroy(float c_bar, int L_max, float alpha);
    void repair(const std::vector<int>& A, float beta, bool insertInNewToursOnly);

    // Assignment operator
    ModifiedSolution& operator=(const ModifiedSolution& other);
};

#endif // SOLUTION_H