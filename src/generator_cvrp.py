import torch
import math, os, random
import torch
import numpy as np

UNIFORM_CAPACITY = {
    20: 30,
    50: 40,
    100: 50,
    200: 70,
    300: 80,
    500: 100,
    800: 130,
    1000: 200,
    2000: 300,
}
# 500 node capacity flowing (Drakulic et al., 2023; Luo et al., 2023).
# 1000 and 2000 node capacity following GLOP


def distance(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


"""
Instance generation code from Queiroga, Eduardo, et al. (2022). 10,000 optimal CVRP solutions for 
testing machine learning based heuristics.

Depot positioning
    1 = Random				
    2 = Centered	# x			
    3 = Cornered				

Customer positioning
    1 = Random	# (x)			
    2 = Clustered	# x			
    3 = Random-clustered		

Demand distribution	
    1 = Unitary		
    2 = Small, large var	# x	
    3 = Small, small var		
    4 = Large, large var		
    5 = Large, small var		
    6 = Large, depending on quadrant	# x
    7 = Few large, many small

Average route size
    1 = Very short
    2 = Short
    3 = Medium # x
    4 = Long
    5 = Very long
    6 = Ultra long

Output: instance file XML<n>_<depotPos><custPos><demandType><avgRouteSize>_<instanceID>.vrp

For more details about the generation process read:
    Uchoa et al (2017). New benchmark instances for the Capacitated Vehicle Routing Problem. European Journal of Operational Research
    Queiroga, Eduardo, et al. (2022). 10,000 optimal CVRP solutions for testing machine learning based heuristics.
"""


def generate_X_instance(n, rootPos, custPos, demandType, avgRouteSize, random):
    # constants
    maxCoord = 1000
    decay = 40

    if demandType > 7:
        print("Demant type out of range!")
        exit(0)

    nSeeds = random.randint(2, 6)

    In = {1: (3, 5), 2: (5, 8), 3: (8, 12), 4: (12, 16), 5: (16, 25), 6: (25, 50)}
    if avgRouteSize > 6:
        print("Average route size out of range!")
        exit(0)
    r = random.uniform(In[avgRouteSize][0], In[avgRouteSize][1])

    S = set()  # set of coordinates for the customers

    # Root positioning
    if rootPos == 1:
        x_ = random.randint(0, maxCoord)
        y_ = random.randint(0, maxCoord)
    elif rootPos == 2:
        x_ = y_ = int(maxCoord / 2.0)
    elif rootPos == 3:
        x_ = y_ = 0
    else:
        print("Depot Positioning out of range!")
        exit(0)
    depot = (x_, y_)

    # Customer positioning
    if custPos == 3:
        nRandCust = int(n / 2.0)
    elif custPos == 2:
        nRandCust = 0
    elif custPos == 1:
        nRandCust = n
        nSeeds = 0
    else:
        print("Costumer Positioning out of range!")
        exit(0)

    nClustCust = n - nRandCust

    # Generating random customers
    for i in range(1, nRandCust + 1):
        x_ = random.randint(0, maxCoord)
        y_ = random.randint(0, maxCoord)
        while (x_, y_) in S or (x_, y_) == depot:
            x_ = random.randint(0, maxCoord)
            y_ = random.randint(0, maxCoord)
        S.add((x_, y_))

    nS = nRandCust

    seeds = []
    # Generation of the clustered customers
    if nClustCust > 0:
        if nClustCust < nSeeds:
            print("Too many seeds!")
            exit(0)

        # Generate the seeds
        for i in range(nSeeds):
            x_ = random.randint(0, maxCoord)
            y_ = random.randint(0, maxCoord)
            while (x_, y_) in S or (x_, y_) == depot:
                x_ = random.randint(0, maxCoord)
                y_ = random.randint(0, maxCoord)
            S.add((x_, y_))
            seeds.append((x_, y_))
        nS = nS + nSeeds

        # Determine the seed with maximum sum of weights (w.r.t. all seeds)
        maxWeight = 0.0
        for i, j in seeds:
            w_ij = 0.0
            for i_, j_ in seeds:
                w_ij += 2 ** (-distance((i, j), (i_, j_)) / decay)
            if w_ij > maxWeight:
                maxWeight = w_ij

        norm_factor = 1.0 / maxWeight

        # Generate the remaining customers using Accept-reject method
        while nS < n:
            x_ = random.randint(0, maxCoord)
            y_ = random.randint(0, maxCoord)
            while (x_, y_) in S or (x_, y_) == depot:
                x_ = random.randint(0, maxCoord)
                y_ = random.randint(0, maxCoord)

            weight = 0.0
            for i_, j_ in seeds:
                weight += 2 ** (-distance((x_, y_), (i_, j_)) / decay)
            weight *= norm_factor
            rand = random.uniform(0, 1)

            if rand <= weight:  # Will we accept the customer?
                S.add((x_, y_))
                nS = nS + 1

    V = [depot] + list(S)  # set of vertices (from now on, the ids are defined)

    # Demands
    demandMinValues = [1, 1, 5, 1, 50, 1, 51, 50, 1]
    demandMaxValues = [1, 10, 10, 100, 100, 50, 100, 100, 10]
    demandMin = demandMinValues[demandType - 1]
    demandMax = demandMaxValues[demandType - 1]
    demandMinEvenQuadrant = 51
    demandMaxEvenQuadrant = 100
    demandMinLarge = 50
    demandMaxLarge = 100
    largePerRoute = 1.5
    demandMinSmall = 1
    demandMaxSmall = 10

    D = []  # demands
    sumDemands = 0
    maxDemand = 0

    for i in range(2, n + 2):
        j = int((demandMax - demandMin + 1) * random.uniform(0, 1) + demandMin)
        if demandType == 6:
            if (V[i - 1][0] < maxCoord / 2.0 and V[i - 1][1] < maxCoord / 2.0) or (
                V[i - 1][0] >= maxCoord / 2.0 and V[i - 1][1] >= maxCoord / 2.0
            ):
                j = int(
                    (demandMaxEvenQuadrant - demandMinEvenQuadrant + 1)
                    * random.uniform(0, 1)
                    + demandMinEvenQuadrant
                )
        if demandType == 7:
            if i < (n / r) * largePerRoute:
                j = int(
                    (demandMaxLarge - demandMinLarge + 1) * random.uniform(0, 1)
                    + demandMinLarge
                )
            else:
                j = int(
                    (demandMaxSmall - demandMinSmall + 1) * random.uniform(0, 1)
                    + demandMinSmall
                )
        D.append(j)
        if j > maxDemand:
            maxDemand = j
        sumDemands = sumDemands + j

    # Generate capacity
    capacity = -1
    if sumDemands == n:
        capacity = math.floor(r)
    else:
        capacity = max(maxDemand, math.ceil(r * sumDemands / n))

    k = math.ceil(sumDemands / float(capacity))

    if demandType != 6:
        random.shuffle(D)
    D = [0] + D

    return [V, D, capacity]


class InstanceGenCVRP:
    """
    Instance generator for the Capacitated Vehicle Routing Problem (CVRP).
    Supports both uniform random and X-generator-based instance creation.
    """

    def __init__(
        self, problem_size, use_X_generator, rootPos, custPos, demandType, avgRouteSize
    ):
        self.problem_size = problem_size
        self.use_X_generator = use_X_generator
        self.rootPos = rootPos
        self.custPos = custPos
        self.demandType = demandType
        self.avgRouteSize = avgRouteSize

    def get_uniform_problems(self, batch_size):
        """
        Generate a batch of CVRP instances with uniformly random depot and node locations,
        random demands, and fixed capacity.

        Returns:
            depot_xy: (batch_size, 1, 2) tensor of depot coordinates
            node_xy: (batch_size, problem_size, 2) tensor of node coordinates
            node_demand: (batch_size, problem_size) tensor of node demands
            node_capacity: (batch_size, 1) tensor of vehicle capacities
        """
        depot_xy = torch.rand(batch_size, 1, 2)
        node_xy = torch.rand(batch_size, self.problem_size, 2)
        node_demand = torch.randint(
            low=1, high=10, size=(batch_size, self.problem_size), dtype=torch.int
        )
        node_capacity = torch.full((batch_size, 1), UNIFORM_CAPACITY[self.problem_size])
        return depot_xy, node_xy, node_demand, node_capacity

    def get_random_problems(self, batch_size, seed=None):
        """
        Generate a batch of CVRP instances, either using the uniform generator or the X-generator.

        Args:
            batch_size: Number of instances to generate
            seed: Optional random seed for reproducibility

        Returns:
            depot_xy: (batch_size, 1, 2) tensor of depot coordinates
            node_xy: (batch_size, problem_size, 2) tensor of node coordinates
            node_demand: (batch_size, problem_size) tensor of node demands
            capacity: (batch_size, 1) tensor of vehicle capacities
        """
        if not self.use_X_generator:
            return self.get_uniform_problems(batch_size)

        if seed is not None:
            random.seed(seed)

        # Preallocate numpy arrays for batch data
        depot_xy_np = np.zeros((batch_size, 1, 2))
        node_xy_np = np.zeros((batch_size, self.problem_size, 2))
        demand_np = np.zeros((batch_size, self.problem_size), dtype=int)
        capacity_list = []

        for i in range(batch_size):
            inst = generate_X_instance(
                self.problem_size,
                self.rootPos,
                self.custPos,
                self.demandType,
                self.avgRouteSize,
                random,
            )
            coords = np.array(inst[0])
            depot_xy_np[i] = coords[0][np.newaxis]
            node_xy_np[i] = coords[1:]
            demand_np[i] = inst[1][1:]
            capacity_list.append(inst[2])

        depot_xy = torch.tensor(depot_xy_np, dtype=torch.float32) / 1000.0
        node_xy = torch.tensor(node_xy_np, dtype=torch.float32) / 1000.0
        node_demand = torch.tensor(demand_np, dtype=torch.int32)
        capacity = torch.tensor(capacity_list, dtype=torch.int32).reshape(batch_size, 1)

        return depot_xy, node_xy, node_demand, capacity
