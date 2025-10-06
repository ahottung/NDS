/*
 * PCVRP Solution Implementation
 * 
 * Implementation of Solution and ModifiedSolution for the Prize-Collecting Vehicle Routing Problem.
 */

#include "Solution.h"
#include <unordered_map>

// Constructor: Create initial solution with one tour per customer
Solution::Solution(const Instance& instance) : instance(instance) {
    // Initialize totalCosts to zero
    totalCosts = 0.0;

    // Create a tour for each customer (initial greedy solution)
    for (int idx = 1; idx <= instance.numCustomers; ++idx) {
        Tour t;

        // Start each tour with a single node (customer)
        t.nodes.push_back(idx);

        // Update the cost and demand of the tour based on the instance
        t.updateCostAndDemand(instance);

        // Add the tour to the solution
        tours.push_back(t);

        // Increment the totalCosts with the cost of the new tour
        totalCosts += t.costs;
    }

    // Add total prizes collected (PCVRP specific)
    totalCosts += instance.total_prizes;

    generateCustomerToTourMap();
}

// Constructor: Create solution from given tour structure
Solution::Solution(const Instance& instance, const std::vector<std::vector<int>>& tours) : instance(instance) {
    // Initialize totalCosts to zero
    totalCosts = 0.0;

    // Create tours from the provided tour structure
    for (const auto& nodes : tours) {
        Tour t;

        // Copy the nodes from the input tours to the new tour
        t.nodes = nodes;

        // Update the cost and demand of the tour based on the instance
        t.updateCostAndDemand(instance);

        // Add the tour to the solution
        this->tours.push_back(t);

        // Increment the totalCosts with the cost of the new tour
        totalCosts += t.costs;
    }

    // Add total prizes collected (PCVRP specific)
    totalCosts += instance.total_prizes;

    generateCustomerToTourMap();
}

// Generate mapping from customer to tour index
void Solution::generateCustomerToTourMap() {
    customerToTourMap.resize(instance.numNodes);

    // Initialize all customers as unassigned
    for (int i = 0; i < instance.numCustomers; ++i) {
        customerToTourMap[i] = -1;
    }

    // Map each customer to its tour index
    for (size_t tourIndex = 0; tourIndex < tours.size(); ++tourIndex) {
        const auto& t = tours[tourIndex];
        for (const auto& c : t.nodes) {
            customerToTourMap[c] = tourIndex;
        }
    }
}



// Accept a modified solution and update the current solution
void Solution::acceptModifiedSolution(ModifiedSolution& modSol) {
    totalCosts = modSol.totalCosts;

    // Sort removed tour indices in descending order for safe removal
    std::sort(modSol.removedToursId.rbegin(), modSol.removedToursId.rend());

    // Remove tours by swapping with back and popping (efficient removal)
    for (size_t index : modSol.removedToursId) {
        auto& removedTour = tours[index];
        removedTour = std::move(tours.back());
        tours.pop_back();

        // Update customer-to-tour mapping for moved tour
        for (const int c : removedTour.nodes) {
            customerToTourMap[c] = index; 
        }
    }

    // Capture the starting point of new tours
    auto startNewTours = tours.size();

    // Add new tours to the solution
    tours.insert(tours.end(), modSol.newTours.begin(), modSol.newTours.end());

    // Update customer-to-tour mapping for new tours
    for (size_t tourIndex = startNewTours; tourIndex < tours.size(); ++tourIndex) {
        const auto& t = tours[tourIndex];
        for (const int c : t.nodes) {
            customerToTourMap[c] = tourIndex;
        }
    }

    // Mark non-inserted customers as unassigned (PCVRP specific)
    for (int c : modSol.nonInsertedNodes) {
        customerToTourMap[c] = -1;
    }

}

// Get list of tours as vector of node sequences
std::vector<std::vector<int>> Solution::getTourList() const {
    std::vector<std::vector<int>> nodeList;
    nodeList.reserve(tours.size());  // Reserve space for efficiency

    // Extract node sequences from each tour
    for (const auto& tour : tours) {
        nodeList.push_back(tour.nodes);
    }

    return nodeList;
}


// Assignment operator
Solution& Solution::operator=(const Solution& other) {
    totalCosts = other.totalCosts;
    tours = other.tours;
    customerToTourMap = other.customerToTourMap;   
    return *this;
}





// ModifiedSolution constructor
ModifiedSolution::ModifiedSolution(Solution& originalSolution) : originalSolution(originalSolution), instance(originalSolution.instance) {
    totalCosts = originalSolution.totalCosts;
}

// Remove specified customers from the solution
void ModifiedSolution::removeCustomers(const std::vector<int>& A) {
    std::unordered_map<int, std::vector<int>> tourModifications;

    totalCosts = 0;

    // Group customers to be removed by their current tour
    for (int customer : A) {
        int tourIndex = originalSolution.customerToTourMap[customer];
        if (tourIndex >= 0) {
            tourModifications[tourIndex].push_back(customer);
        }
    }

    // Apply removals and update tours
    for (const auto& entry : tourModifications) {
        int tourIndex = entry.first;
        const std::vector<int>& customersToRemove = entry.second;

        Tour& tour = originalSolution.tours[tourIndex];
        Tour newTour;
        // Reserve space in advance to avoid multiple allocations
        newTour.nodes.reserve(tour.nodes.size() - customersToRemove.size());

        // Remove specified customers from the tour
        for (int node_id : tour.nodes) {
            if (std::find(customersToRemove.begin(), customersToRemove.end(), node_id) == customersToRemove.end()) {
                newTour.nodes.push_back(node_id);
            }
        }

        if (!newTour.nodes.empty()) {
            newTour.updateCostAndDemand(instance);
            totalCosts += newTour.costs;
            newTours.push_back(newTour);
        }
        removedToursId.push_back(tourIndex);
    }

    // Include the costs of unaffected tours
    for (size_t i = 0; i < originalSolution.tours.size(); ++i) {
        if (tourModifications.find(i) == tourModifications.end()) {
            totalCosts += originalSolution.tours[i].costs;
        }
    }

    totalCosts += instance.total_prizes;
}


// Destroy solution using SISRs destroy operation (baseline code)
std::unordered_set<int> ModifiedSolution::destroy(float c_bar, int L_max, float alpha) {
    std::unordered_set<int> A;      // Set of customers removed
    std::unordered_set<Tour*> R;    // Set of tours affected (to be removed)

    // Compute average tour cardinality (number of customers per tour)
    float avg_tour_cardinality = 0;
    for (auto& t : originalSolution.tours) { avg_tour_cardinality += t.nodes.size(); }
    avg_tour_cardinality /= originalSolution.tours.size();

    // Limit for string removal size
    int ls_max = std::min(L_max, static_cast<int>(avg_tour_cardinality));

    // Maximum number of tours to remove from, based on heuristic formula
    float ks_max = (4 * c_bar) / (1 + ls_max) - 1;
    int ks = static_cast<int>(getRandomFraction(1, ks_max + 0.9999));

    // Randomly select a seed customer
    int seed_c = getRandomNumber(1, instance.numCustomers);

    // Iterate over the adjacency list of the seed customer
    for (int c : instance.adj[seed_c]) {

        if (c == 0) {  // Skip depot
            continue;
        }

        // Stop if enough tours/customers have been removed
        if (R.size() >= ks || A.size() > c_bar * 2) {
            break;
        }

        // If customer is not part of any tour, add directly to removal set
        if (originalSolution.customerToTourMap[c] == -1)
        {
            // The customer is not part of a tour
            A.insert(c);
        }

        // Get the tour containing customer c
        Tour* c_tour = &originalSolution.tours[originalSolution.customerToTourMap[c]];

        // If customer and tour haven't been processed yet
        if (A.count(c) == 0 && R.count(c_tour) == 0) {
            int c_star = c;

            int c_tour_card = c_tour->nodes.size();
            int lt_max = std::min(c_tour_card, ls_max);

            // Randomly select string length for removal
            int lt = static_cast<int>(getRandomFraction(1, lt_max + 0.9999));

            std::vector<int> removed_cust;
            Tour newTour;

            // Remove a string or split string from the tour
            if (lt < 2 || lt == lt_max || getRandomFractionFast() < 0.5) {
                c_tour->stringRemoval(removed_cust, newTour, lt, c_star);
            }
            else {
                c_tour->splitStringRemoval(removed_cust, newTour, lt, c_star, alpha);
            }

            // Add removed customers to set A, mark tour as affected
            A.insert(removed_cust.begin(), removed_cust.end());
            R.insert(c_tour);

            // If the tour still has customers, update and keep it
            if (newTour.nodes.size() > 0) {
                newTour.updateCostAndDemand(instance);
                newTours.push_back(newTour);
            }
        }
    }

    // Recalculate total cost: sum costs of unaffected tours and new tours
    totalCosts = 0;
    for (int i = originalSolution.tours.size() - 1; i >= 0; --i) {
        Tour* tour = &originalSolution.tours[i];
        if (R.count(tour) == 0)
        {
            totalCosts += tour->costs;
        }
        else {
            removedToursId.push_back(i);
        }
    }
    for (auto& t : newTours)
    {
        totalCosts += t.costs;
    }
    totalCosts += instance.total_prizes;

    return A;
}

// Repair the solution by reinserting removed customers
void ModifiedSolution::repair(const std::vector<int>& A, float beta, bool insertInNewToursOnly) {
    const int& capacity = instance.vehicleCapacity;

    int best_original_tour_idx = -1;
    int best_new_tour_idx = -1;
    int best_ins_pos = 0;
    int next_node;
    float insertionCosts;
    float bestInsertionCost;
    std::vector<int> oldTourIds;
    nonInsertedNodes.clear();

    // Build list of indices of tours not removed (if allowed)
    if (!insertInNewToursOnly) {
        for (int t_idx = 0; t_idx < originalSolution.tours.size(); ++t_idx) {
            if (std::find(removedToursId.begin(), removedToursId.end(), t_idx) == removedToursId.end()) {
                oldTourIds.push_back(t_idx);
            }
        }
    }

    // Try to reinsert each removed customer
    for (int c : A) {
        bestInsertionCost = std::numeric_limits<float>::infinity();
        const int& c_demand = instance.demand[c];
        best_original_tour_idx = -1;
        best_new_tour_idx = -1;

        // Try to insert into existing (old) tours
        if (!insertInNewToursOnly) {
            for (int t_idx : oldTourIds) {
                const auto& tour = originalSolution.tours[t_idx];

                if (tour.demand + c_demand <= capacity) {

                    int prev_node = 0;
                    for (int new_pos = 0; new_pos < tour.nodes.size(); ++new_pos) {
                        next_node = tour.nodes[new_pos];

                        // Calculate cost of inserting c between prev_node and next_node
                        insertionCosts = instance.distanceMatrix[prev_node][c] + instance.distanceMatrix[c][next_node] - instance.distanceMatrix[prev_node][next_node] - instance.prizes[c];

                        prev_node = next_node;

                        // Accept with probability (1-beta) if cost is best so far
                        if (insertionCosts < bestInsertionCost) {
                            if (getRandomFractionFast() < (1 - beta)) {
                                best_original_tour_idx = t_idx;
                                best_ins_pos = new_pos;
                                bestInsertionCost = insertionCosts;
                            }
                        }
                    }

                    // Try inserting at the end of the tour
                    insertionCosts = instance.distanceMatrix[prev_node][c] + instance.distanceMatrix[c][0] - instance.distanceMatrix[prev_node][0] - instance.prizes[c];

                    if (insertionCosts < bestInsertionCost) {
                        if (getRandomFractionFast() < (1 - beta)) {
                            best_original_tour_idx = t_idx;
                            best_ins_pos = tour.nodes.size();
                            bestInsertionCost = insertionCosts;
                        }
                    }
                }
            }
        }

        // Try to insert into new tours created during destroy
        for (int t_idx = 0; t_idx < newTours.size(); ++t_idx) {
            const auto& tour = newTours[t_idx];

            if (tour.demand + c_demand <= capacity) {

                int prev_node = 0;
                for (int new_pos = 0; new_pos < tour.nodes.size(); ++new_pos) {
                    next_node = tour.nodes[new_pos];

                    // Calculate cost of inserting c between prev_node and next_node
                    insertionCosts = instance.distanceMatrix[prev_node][c] + instance.distanceMatrix[c][next_node] - instance.distanceMatrix[prev_node][next_node] - instance.prizes[c];

                    prev_node = next_node;

                    // Accept with probability (1-beta) if cost is best so far
                    if (insertionCosts < bestInsertionCost) {
                        if (getRandomFractionFast() < (1 - beta)) {
                            best_new_tour_idx = t_idx;
                            best_ins_pos = new_pos;
                            bestInsertionCost = insertionCosts;
                        }
                    }
                }

                // Try inserting at the end of the tour
                insertionCosts = instance.distanceMatrix[prev_node][c] + instance.distanceMatrix[c][0] - instance.distanceMatrix[prev_node][0] - instance.prizes[c];

                if (insertionCosts < bestInsertionCost) {
                    if (getRandomFractionFast() < (1 - beta)) {
                        best_new_tour_idx = t_idx;
                        best_ins_pos = tour.nodes.size();
                        bestInsertionCost = insertionCosts;
                    }
                }
            }
        }

        // Insert customer at best found position if cost is negative (improves solution)
        if (bestInsertionCost < 0) {
            if (best_new_tour_idx != -1) {
                auto& tour = newTours[best_new_tour_idx];
                tour.nodes.insert(tour.nodes.begin() + best_ins_pos, c);
                tour.demand += c_demand;
                tour.costs += bestInsertionCost;
                totalCosts += bestInsertionCost;

            }
            else if (best_original_tour_idx != -1) {
                // If inserted into an old tour, create a new tour object for it
                Tour newTour = originalSolution.tours[best_original_tour_idx];
                removedToursId.push_back(best_original_tour_idx);
                oldTourIds.erase(std::remove(oldTourIds.begin(), oldTourIds.end(), best_original_tour_idx), oldTourIds.end());

                newTour.nodes.insert(newTour.nodes.begin() + best_ins_pos, c);
                newTour.demand += c_demand;
                newTour.costs += bestInsertionCost;

                newTours.push_back(newTour);
                totalCosts += bestInsertionCost;
            }
        }
        // If not inserted, but a single-customer tour is beneficial or by chance, create a new tour
        else if ((instance.distanceMatrix[0][c] + instance.distanceMatrix[c][0] - instance.prizes[c] < 0) ||
            getRandomFractionFast() < 0.5) {
            Tour newTour;
            newTour.nodes = { c };
            newTour.costs = instance.distanceMatrix[0][c] + instance.distanceMatrix[c][0] - instance.prizes[c];
            newTour.demand = c_demand;
            totalCosts += newTour.costs;
            newTours.push_back(newTour);
        }
        // Otherwise, mark as not inserted
        else {
            nonInsertedNodes.push_back(c);
        }
    }
}


ModifiedSolution& ModifiedSolution::operator=(const ModifiedSolution& other) {
    if (this != &other) {
        // Copy values from 'other' to 'this'
        totalCosts = other.totalCosts;
        newTours = other.newTours;
        removedToursId = other.removedToursId;
        nonInsertedNodes = other.nonInsertedNodes;
    }
    return *this;
}
