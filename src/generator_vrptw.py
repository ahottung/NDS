import torch
import random
import torch
import numpy as np
from .generator_cvrp import generate_X_instance

GRID_SIZE = 1000


def _get_time_windows(
    depot_xy, node_xy, service_window, time_window_size, service_duration
):
    batch_size = node_xy.shape[0]
    problem_size = node_xy.shape[1]

    # Distance from the nodes to the depot
    traveling_time = (
        torch.linalg.vector_norm((depot_xy - node_xy).float(), dim=-1) * GRID_SIZE
    )

    # TW start needs to be feasibly reachable directly from depot
    tw_start_min = torch.ceil(traveling_time) + 1

    # TW end needs to be early enough to perform service and return to depot until end of service window
    tw_end_max = service_window - torch.ceil(traveling_time + service_duration) - 1

    # Sample time windows center
    tw_center = tw_start_min + torch.round(
        (tw_end_max - tw_start_min) * torch.rand(batch_size, problem_size)
    )

    # Define time window start and end
    tw_start = tw_center - time_window_size // 2
    tw_end = tw_center + time_window_size // 2

    tw_start = torch.clamp(tw_start, min=tw_start_min)
    tw_end = torch.clamp(tw_end, max=tw_end_max)

    node_tw = torch.stack([tw_start, tw_end], dim=-1)
    depot_tw = torch.Tensor([[0, service_window]]).repeat(batch_size, 1)[:, None]

    # Rescale
    depot_tw /= GRID_SIZE
    node_tw /= GRID_SIZE
    service_duration /= GRID_SIZE

    # Expand service duration
    node_sd = torch.full((batch_size, problem_size), service_duration)

    return depot_tw, node_tw, node_sd


class InstanceGenVRPTW:
    """
    Instance generator for the VRPTW (Vehicle Routing Problem with Time Windows).
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
        service_window,
        time_window_size,
        service_duration,
    ):
        self.problem_size = problem_size
        self.use_X_generator = use_X_generator
        self.rootPos = rootPos
        self.custPos = custPos
        self.demandType = demandType
        self.avgRouteSize = avgRouteSize
        self.service_window = service_window
        self.time_window_size = time_window_size
        self.service_duration = service_duration

        if not self.use_X_generator:
            raise ValueError("Uniform generation not supported for VRPTW")

    def get_random_problems(self, batch_size, seed=None):
        """
        Generate a batch of VRPTW instances using the X-generator.

        Args:
            batch_size (int): Number of instances to generate.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            depot_xy (Tensor): (batch_size, 1, 2) depot coordinates.
            node_xy (Tensor): (batch_size, problem_size, 2) node coordinates.
            node_demand (Tensor): (batch_size, problem_size) node demands.
            capacity (Tensor): (batch_size, 1) vehicle capacities.
            depot_tw (Tensor): (batch_size, 1, 2) depot time windows.
            node_tw (Tensor): (batch_size, problem_size, 2) node time windows.
            node_sd (Tensor): (batch_size, problem_size) node service durations.
        """
        if seed is not None:
            random.seed(seed)
            torch.random.manual_seed(seed)

        # Preallocate arrays for batch data
        depot_xy_np = np.zeros((batch_size, 1, 2))
        node_xy_np = np.zeros((batch_size, self.problem_size, 2))
        demand_np = np.zeros((batch_size, self.problem_size), dtype=int)
        capacity_list = []

        for i in range(batch_size):
            # Generate a single instance using the X-generator
            inst = generate_X_instance(
                self.problem_size,
                self.rootPos,
                self.custPos,
                self.demandType,
                self.avgRouteSize,
                random,
            )
            coords = np.array(inst[0])
            depot_xy_np[i] = coords[0, np.newaxis]
            node_xy_np[i] = coords[1:]
            demand_np[i] = inst[1][1:]
            capacity_list.append(inst[2])

        # Convert numpy arrays to torch tensors and rescale coordinates
        depot_xy = torch.tensor(depot_xy_np, dtype=torch.float32) / GRID_SIZE
        node_xy = torch.tensor(node_xy_np, dtype=torch.float32) / GRID_SIZE
        node_demand = torch.tensor(demand_np, dtype=torch.int32)
        capacity = torch.tensor(capacity_list, dtype=torch.int32).reshape(batch_size, 1)

        # Generate time windows and service durations
        depot_tw, node_tw, node_sd = _get_time_windows(
            depot_xy,
            node_xy,
            self.service_window,
            self.time_window_size,
            self.service_duration,
        )

        return depot_xy, node_xy, node_demand, capacity, depot_tw, node_tw, node_sd
