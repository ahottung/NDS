/*
 * PCVRP Operations Implementation
 * 
 * Implementation of various operations for the Prize-Collecting Vehicle Routing Problem,
 * including solution construction, destruction, and repair operations.
 */

#include "Operations.h"

// Sort customers by various criteria (random, demand, distance from depot) as described in the SISRs paper
// This function is only used by the handcrafted baseline heuristic
void sort_abs_cust(std::vector<int>& A, const Instance& instance, char order) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    // If no order specified, randomly choose one
    if (order == '\0') {
        static std::vector<char> options = { 'R', 'D', 'F', 'C' };
        static std::discrete_distribution<> dist({ 4, 4, 2, 1 });
        order = options[dist(gen)];
    }

    // Apply sorting based on the specified order
    if (order == 'R') {
        // Random shuffle
        std::shuffle(A.begin(), A.end(), gen);
    }
    else if (order == 'D') {
        // Sort by demand (descending)
        std::sort(A.begin(), A.end(), [&](int a, int b) {
            return instance.demand[a] > instance.demand[b];
        });
    }
    else if (order == 'F') {
        // Sort by distance from depot (far to near)
        std::sort(A.begin(), A.end(), [&](int a, int b) {
            return instance.distanceMatrix[a][0] > instance.distanceMatrix[b][0];
        });
    }
    else {
        // Sort by distance from depot (near to far)
        std::sort(A.begin(), A.end(), [&](int a, int b) {
            return instance.distanceMatrix[a][0] < instance.distanceMatrix[b][0];
        });
    }
}

// Baseline deconstruction: Generate multiple customer removal sets using heuristic deconstruction
std::vector<std::vector<int>> heuristic_deconstruction_selection(Solution& solution, float c_bar, int m) {
    std::vector<std::vector<int>> A;
    
    // Generate m different removal sets
    for (int i = 0; i < m; i++) {
        ModifiedSolution msol(solution);
        std::unordered_set<int> removedCustomers_set = msol.destroy(c_bar, 10, 0.01);
        std::vector<int> removedCustomers(removedCustomers_set.begin(), removedCustomers_set.end());
        A.push_back(removedCustomers);
    }
    
    return A;
}

// Remove and recreate with all improvements (simulated annealing)
std::tuple<Solution, std::vector<float>> remove_recreate_allImp(
    Solution solution,
    std::vector<std::vector<int>>& A,
    float beta,
    int n,
    float T,
    bool insertInNewToursOnly)
{
    std::vector<float> costs(A.size(), std::numeric_limits<float>::infinity());

    // Process each removal set
    for (size_t i = 0; i < A.size(); ++i) {
        ModifiedSolution baseMsol(solution);
        baseMsol.removeCustomers(A[i]);
        std::vector<int> removedCustomers = A[i];

        float innerBestCost = std::numeric_limits<float>::infinity();
        ModifiedSolution innerBestSol(baseMsol);

        // Try n different repair attempts for this removal set
        for (int j = 0; j < n; ++j) {
            // Use the order of the DNN for the first attempt, then randomize for subsequent attempts
            if (j > 0) {
                sort_abs_cust(removedCustomers, solution.instance, 'R');
            }

            ModifiedSolution msolCopy(baseMsol);
            msolCopy.repair(removedCustomers, beta, insertInNewToursOnly);

            // Update best solution for this removal set
            if (msolCopy.totalCosts < innerBestCost) {
                innerBestCost = msolCopy.totalCosts;
                innerBestSol = msolCopy;
            }
        }

        costs[i] = innerBestCost;

        // Accept local best solution with simulated annealing threshold
        float thresh = solution.totalCosts - T * std::log(getRandomFraction(0, 1));
        if (thresh > innerBestSol.totalCosts) {
            solution.acceptModifiedSolution(innerBestSol);
        }
    }

    return std::make_tuple(solution, costs);
}

// Remove and recreate with single improvement (best solution only)
std::tuple<Solution, std::vector<float>> remove_recreate_singleImp(
    Solution solution,
    std::vector<std::vector<int>>& A,
    float beta,
    int n,
    bool insertInNewToursOnly)
{
    // Track the best solution and its cost across all removal sets
    float outerBestCost = std::numeric_limits<float>::infinity();
    ModifiedSolution outerBestSol(solution);

    std::vector<float> costs(A.size(), std::numeric_limits<float>::infinity());

    // Process each removal set
    for (size_t i = 0; i < A.size(); ++i) {
        ModifiedSolution baseMsol(solution);
        baseMsol.removeCustomers(A[i]);
        std::vector<int> removedCustomers = A[i];

        float innerBestCost = std::numeric_limits<float>::infinity();

        // Try n different repair attempts for this removal set
        for (int j = 0; j < n; ++j) {
            // Randomize customer order for subsequent attempts
            if (j > 0) {
                sort_abs_cust(removedCustomers, solution.instance, 'R');
            }

            ModifiedSolution msolCopy(baseMsol);
            msolCopy.repair(removedCustomers, beta, insertInNewToursOnly);

            // Update local best for this removal set
            if (msolCopy.totalCosts < innerBestCost) {
                innerBestCost = msolCopy.totalCosts;
            }

            // Update global best across all removal sets
            if (msolCopy.totalCosts < outerBestCost) {
                outerBestCost = msolCopy.totalCosts;
                outerBestSol = msolCopy;
            }
        }

        costs[i] = innerBestCost;
    }

    // Accept the best found solution across all removal sets
    solution.acceptModifiedSolution(outerBestSol);

    return std::make_tuple(solution, costs);
}

// Create a starting solution using random improvements
Solution create_starting_solution(const Instance& instance, int nbImprovement, int nbDestroy) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    // Start with a basic solution (one tour per customer)
    Solution sol = Solution(instance);
    std::vector<float> costs;

    // Create list of all customer nodes
    std::vector<int> allNodes(instance.numCustomers);
    std::iota(allNodes.begin(), allNodes.end(), 1);

    // Apply random improvements
    for (int i = 0; i < nbImprovement; ++i) {
        // Randomly shuffle all nodes
        std::shuffle(allNodes.begin(), allNodes.end(), gen);

        // Select first nbDestroy nodes for removal
        std::vector<int> nodesToRemove(allNodes.begin(), allNodes.begin() + nbDestroy);

        // Create removal set and apply improvement
        std::vector<std::vector<int>> A = {nodesToRemove};

        std::tie(sol, costs) = remove_recreate_allImp(sol, A, 0.0, 1, 0, false);
    }

    return sol;
}

