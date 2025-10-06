import torch
import random
import torch
import numpy as np
from .generator_cvrp import generate_X_instance

GRID_SIZE = 1000


def _get_prizes(node_demand, prize_min, prize_max, prize_alpha):
    min_demand = node_demand.min(dim=1, keepdim=True)[0]
    max_demand = node_demand.max(dim=1, keepdim=True)[0]
    demand_factor = (node_demand - min_demand) / (max_demand - min_demand)

    prize_deterministic = prize_min + demand_factor * (prize_max - prize_min)
    prize_randomness = prize_min + torch.rand_like(node_demand.float()) * (
        prize_max - prize_min
    )
    prize = prize_alpha * prize_deterministic + (1 - prize_alpha) * prize_randomness

    return prize


class InstanceGenPCVRP:
    """
    Instance generator for the Prize-Collecting Capacitated Vehicle Routing Problem (PCVRP).
    Only supports X-generator-based instance creation.
    """

    def __init__(
        self,
        problem_size,
        use_X_generator,
        rootPos,
        custPos,
        demandType,
        avgRouteSize,
        prizes_min,
        prizes_max,
        prizes_alpha,
    ):
        self.problem_size = problem_size
        self.use_X_generator = use_X_generator
        self.rootPos = rootPos
        self.custPos = custPos
        self.demandType = demandType
        self.avgRouteSize = avgRouteSize
        self.prizes_min = prizes_min
        self.prizes_max = prizes_max
        self.prizes_alpha = prizes_alpha

        if not self.use_X_generator:
            raise ValueError("Uniform generation not supported for PCVRP")

    def get_random_problems(self, batch_size, seed=None):
        """
        Generate a batch of PCVRP instances using the X-generator.

        Args:
            batch_size (int): Number of instances to generate.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            depot_xy (Tensor): (batch_size, 1, 2) depot coordinates.
            node_xy (Tensor): (batch_size, problem_size, 2) node coordinates.
            node_demand (Tensor): (batch_size, problem_size) node demands.
            capacity (Tensor): (batch_size, 1) vehicle capacities.
            node_prizes (Tensor): (batch_size, problem_size) node prizes.
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Preallocate arrays for batch data
        depot_xy_np = np.zeros((batch_size, 1, 2))
        node_xy_np = np.zeros((batch_size, self.problem_size, 2))
        demand_np = np.zeros((batch_size, self.problem_size), dtype=int)
        capacity_list = []

        for i in range(batch_size):
            # Generate a single instance using the X-generator
            instance = generate_X_instance(
                self.problem_size,
                self.rootPos,
                self.custPos,
                self.demandType,
                self.avgRouteSize,
                random,
            )
            coords = np.array(instance[0])
            depot_xy_np[i] = coords[0][np.newaxis]
            node_xy_np[i] = coords[1:]
            demand_np[i] = instance[1][1:]
            capacity_list.append(instance[2])

        # Convert numpy arrays to torch tensors and normalize coordinates
        depot_xy = torch.tensor(depot_xy_np, dtype=torch.float32) / GRID_SIZE
        node_xy = torch.tensor(node_xy_np, dtype=torch.float32) / GRID_SIZE
        node_demand = torch.tensor(demand_np, dtype=torch.int32)
        capacity = torch.tensor(capacity_list, dtype=torch.int32).reshape(batch_size, 1)

        # Compute node prizes based on demands
        node_prizes = _get_prizes(
            node_demand,
            self.prizes_min,
            self.prizes_max,
            self.prizes_alpha,
        )

        return depot_xy, node_xy, node_demand, capacity, node_prizes
