/*
 * VRPTW Operations Header
 * 
 * Declares various operations for the Vehicle Routing Problem with Time Windows,
 * including solution construction, destruction, and repair operations.
 */

#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "Instance.h"
#include "Solution.h"
#include "Utils.h"

// Baseline code: Sort customers by various criteria (random, demand, distance, time windows)
void sort_abs_cust(std::vector<int>& A, const Instance& instance, char order = '\0');

// Baseline code: Generate multiple customer removal sets using heuristic deconstruction
std::vector<std::vector<int>> heuristic_deconstruction_selection(Solution& solution, float c_bar, int m);

// Remove and recreate with all improvements (simulated annealing)
std::tuple<Solution, std::vector<float>> remove_recreate_allImp(Solution solution, std::vector<std::vector<int>>& A, 
                                                               float beta, int n, float T, bool insertInNewToursOnly = true);

// Remove and recreate with single improvement (best solution only)
std::tuple<Solution, std::vector<float>> remove_recreate_singleImp(Solution solution, std::vector<std::vector<int>>& A, 
                                                                  float beta, int n, bool insertInNewToursOnly = true);

// Create a starting solution using random improvements
Solution create_starting_solution(const Instance& instance, int nbImprovement, int nbDestroy);

#endif // OPERATIONS_H