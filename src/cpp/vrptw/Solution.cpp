/*
 * VRPTW Solution Implementation
 * 
 * Implementation of Solution and ModifiedSolution for the Vehicle Routing Problem with Time Windows.
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

    generateCustomerToTourMap();
}

// Generate mapping from customer to tour index
void Solution::generateCustomerToTourMap() {
    customerToTourMap.resize(instance.numNodes);

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

    // Populate the map with customers to be removed per tour
    for (int customer : A) {
        int tourIndex = originalSolution.customerToTourMap[customer];
        tourModifications[tourIndex].push_back(customer);
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
}


// Destroy solution using SISRs destroy operation (baseline code)
std::unordered_set<int> ModifiedSolution::destroy(float c_bar, int L_max, float alpha) {
    std::unordered_set<int> A; // Set of customers removed
    std::unordered_set<Tour*> R; // Set of tours affected by removal

    // Calculate average tour cardinality (number of customers per tour)
    float avg_tour_cardinality = 0;
    for (auto& t : originalSolution.tours) { 
        avg_tour_cardinality += t.nodes.size(); 
    }
    avg_tour_cardinality /= originalSolution.tours.size();

    // Calculate maximum string length for removal
    int ls_max = std::min(L_max, static_cast<int>(avg_tour_cardinality));

    // Calculate number of strings to remove (ks)
    float ks_max = (4 * c_bar) / (1 + ls_max) - 1;
    int ks = static_cast<int>(getRandomFraction(1, ks_max + 0.9999));

    // Select random seed customer (not depot)
    int seed_c = getRandomNumber(1, instance.numCustomers);

    // Iterate over the neighbors of the seed customer
    for (int c : instance.adj[seed_c]) {

        if (c == 0) {  // Skip depot
            continue;
        }

        // Stop if enough strings have been removed
        if (R.size() >= ks) {
            break;
        }

        Tour* c_tour = &originalSolution.tours[originalSolution.customerToTourMap[c]];

        // Only consider customers not already removed and tours not already affected
        if (A.count(c) == 0 && R.count(c_tour) == 0) {
            int c_star = c;

            int c_tour_card = c_tour->nodes.size();
            int lt_max = std::min(c_tour_card, ls_max);

            // Randomly select string length to remove
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

            // Add removed customers to set A and mark tour as affected
            A.insert(removed_cust.begin(), removed_cust.end());
            R.insert(c_tour);

            // If the new tour is not empty, update its cost and add to newTours
            if (newTour.nodes.size() > 0) {
                newTour.updateCostAndDemand(instance);
                newTours.push_back(newTour);
            }
        }
    }

    // Recalculate total cost and update removedToursId
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
    // Add costs of new tours
    for (auto& t : newTours)
    {
        totalCosts += t.costs;
    }

    return A;
}


// Repair solution by inserting removed customers
void ModifiedSolution::repair(const std::vector<int>& A, float beta, bool insertInNewToursOnly) {
    const int& capacity = instance.vehicleCapacity;

    // Variables for best insertion tracking
    int best_original_tour_idx = -1;
    int best_new_tour_idx = -1;
    int best_ins_pos = 0;
    int next_node;
    float insertionCosts;
    float bestInsertionCost;
    std::vector<int> oldTourIds;

    // Collect indices of original tours that can accept new customers
    if (!insertInNewToursOnly) {
        for (int t_idx = 0; t_idx < originalSolution.tours.size(); ++t_idx) {
            if (std::find(removedToursId.begin(), removedToursId.end(), t_idx) == removedToursId.end()) {
                oldTourIds.push_back(t_idx);
            }
        }
    }

    // Lambda to check time window feasibility for an insertion
    auto checkTimeWindows = [&](const auto& tour, int& c, int new_pos, float& prev_node_time, int& prev_node, bool& isTourInserationFeasible) -> bool {
        float arrival_time = prev_node_time + instance.distanceMatrix[prev_node][c];

        if (arrival_time > instance.endTW[c]) {
            isTourInserationFeasible = false;
            return false;
        }

        arrival_time = std::max(arrival_time, instance.startTW[c]) + instance.serviceTime[c];
        int prev_cust = c;
        for (int j = new_pos; j < tour.nodes.size(); ++j) {
            int cust = tour.nodes[j];
            arrival_time += instance.distanceMatrix[prev_cust][cust];

            if (arrival_time > instance.endTW[cust]) {
                return false;
            }
            if (arrival_time < instance.startTW[cust]) {
                break;
            }
            arrival_time += instance.serviceTime[cust];

            prev_cust = cust;
        }
        return true;
    };

    // Try to reinsert each removed customer
    for (int c : A) {
        bestInsertionCost = std::numeric_limits<float>::infinity();
        const int& c_demand = instance.demand[c];
        best_original_tour_idx = -1;
        best_new_tour_idx = -1;

        // Try to insert into original tours (if allowed)
        if (!insertInNewToursOnly) {
            for (int t_idx : oldTourIds) {
                const auto& tour = originalSolution.tours[t_idx];

                // Check capacity constraint
                if (tour.demand + c_demand <= capacity) {

                    int prev_node = 0;
                    float prev_node_time = 0;
                    bool isTourInserationFeasible = true;
                    // Try all possible insertion positions
                    for (int new_pos = 0; new_pos < tour.nodes.size(); ++new_pos) {
                        next_node = tour.nodes[new_pos];

                        insertionCosts = instance.distanceMatrix[prev_node][c] + instance.distanceMatrix[c][next_node] - instance.distanceMatrix[prev_node][next_node];

                        // Check if this insertion is best so far and feasible
                        if (insertionCosts < bestInsertionCost && checkTimeWindows(tour, c, new_pos, prev_node_time, prev_node, isTourInserationFeasible)) {
                            best_original_tour_idx = t_idx;
                            best_ins_pos = new_pos;
                            bestInsertionCost = insertionCosts;
                        }

                        if (!isTourInserationFeasible) { break; }
                      
                        prev_node_time = std::max(prev_node_time + instance.distanceMatrix[prev_node][next_node], instance.startTW[next_node]) + instance.serviceTime[next_node];
                        prev_node = next_node;
                    }

                    // Try insertion at the end of the tour
                    if (isTourInserationFeasible) {
                        insertionCosts = instance.distanceMatrix[prev_node][c] + instance.distanceMatrix[c][0] - instance.distanceMatrix[prev_node][0];

                        if (insertionCosts < bestInsertionCost  && checkTimeWindows(tour, c, tour.nodes.size(), prev_node_time, prev_node, isTourInserationFeasible)) {
                            // With probability (1-beta), accept this insertion at the end
                            if (getRandomFractionFast() < (1 - beta)) {
                                best_original_tour_idx = t_idx;
                                best_ins_pos = tour.nodes.size();
                                bestInsertionCost = insertionCosts;
                            }
                        }
                    }

                }
            }
        }

        // Try to insert into new tours
        for (int t_idx = 0; t_idx < newTours.size(); ++t_idx) {
            const auto& tour = newTours[t_idx];

            // Check capacity constraint
            if (tour.demand + c_demand <= capacity) {

                int prev_node = 0;
                float prev_node_time = 0;
                bool isTourInserationFeasible = true;
                // Try all possible insertion positions
                for (int new_pos = 0; new_pos < tour.nodes.size(); ++new_pos) {
                    next_node = tour.nodes[new_pos];

                    insertionCosts = instance.distanceMatrix[prev_node][c] + instance.distanceMatrix[c][next_node] - instance.distanceMatrix[prev_node][next_node];                

                    // Check if this insertion is best so far and feasible
                    if (insertionCosts < bestInsertionCost && checkTimeWindows(tour, c, new_pos, prev_node_time, prev_node, isTourInserationFeasible)) {
                        best_new_tour_idx = t_idx;
                        best_ins_pos = new_pos;
                        bestInsertionCost = insertionCosts;
                    }

                    if (!isTourInserationFeasible) { break; }
                 
                    prev_node_time = std::max(prev_node_time + instance.distanceMatrix[prev_node][next_node], instance.startTW[next_node]) + instance.serviceTime[next_node];
                    prev_node = next_node;
                }

                // Try insertion at the end of the tour
                if (isTourInserationFeasible) {

                    insertionCosts = instance.distanceMatrix[prev_node][c] + instance.distanceMatrix[c][0] - instance.distanceMatrix[prev_node][0];

                    if (insertionCosts < bestInsertionCost && checkTimeWindows(tour, c, tour.nodes.size(), prev_node_time, prev_node, isTourInserationFeasible)) {
                        best_new_tour_idx = t_idx;
                        best_ins_pos = tour.nodes.size();
                        bestInsertionCost = insertionCosts;
                    }
                }

            }
        }

        // Insert customer into the best found position (new tour, original tour, or create a new tour)
        if (best_new_tour_idx != -1) {
            auto& tour = newTours[best_new_tour_idx];
            tour.nodes.insert(tour.nodes.begin() + best_ins_pos, c);
            tour.demand += c_demand;
            tour.costs += bestInsertionCost;
            totalCosts += bestInsertionCost;

        }
        else if (best_original_tour_idx != -1) {
            // Remove the original tour and create a new one with the customer inserted
            Tour newTour = originalSolution.tours[best_original_tour_idx];
            removedToursId.push_back(best_original_tour_idx);
            oldTourIds.erase(std::remove(oldTourIds.begin(), oldTourIds.end(), best_original_tour_idx), oldTourIds.end());

            newTour.nodes.insert(newTour.nodes.begin() + best_ins_pos, c);
            newTour.demand += c_demand;
            newTour.costs += bestInsertionCost;

            newTours.push_back(newTour);
            totalCosts += bestInsertionCost;

        }
        else {
            // If no feasible insertion, create a new tour for this customer
            Tour newTour;
            newTour.nodes = { c };
            newTour.costs = instance.distanceMatrix[0][c] + instance.distanceMatrix[c][0];
            newTour.demand = c_demand;
            totalCosts += newTour.costs;
            newTours.push_back(newTour);
        }
    }
}



// Assignment operator for ModifiedSolution
ModifiedSolution& ModifiedSolution::operator=(const ModifiedSolution& other) {
    if (this != &other) {
        // Copy values from 'other' to 'this'
        totalCosts = other.totalCosts;
        newTours = other.newTours;
        removedToursId = other.removedToursId;
    }
    return *this;
}
