"""PCVRP (Prize Collecting Vehicle Routing Problem) instance generation and management."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from .problem import Problem
from .generator_pcvrp import InstanceGenPCVRP


@dataclass
class ProblemData:
    """Container for PCVRP problem instance data."""

    problem_name: str = "pcvrp"
    problem_size: int = None
    capacity: np.ndarray = None
    depot_node_xy: np.ndarray = None
    depot_node_demand: np.ndarray = None
    depot_node_prizes: np.ndarray = None


@dataclass
class ProblemFeatures:
    """Container for PCVRP problem features (normalized for neural network)."""

    depot_xy: torch.Tensor = None  # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None  # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None  # shape: (batch, problem) - normalized
    node_prizes: torch.Tensor = None  # shape: (batch, problem)


class ProblemPCVRP(Problem):
    """PCVRP problem instance generator and manager."""

    def __init__(self, problem_size: int, generator_params: dict = None):
        """
        Initialize PCVRP problem generator.

        Args:
            problem_size: Number of customers
            generator_params: Optional parameters for instance generation
        """
        super().__init__(problem_size, generator_params)
        self.name = "pcvrp"

        # Additional dataset field for prizes
        self.dataset_node_prizes = None

        if generator_params is not None:
            self.instanceGen = InstanceGenPCVRP(self.problem_size, **generator_params)

    def load_problem_dataset_pkl(
        self, filename: str, num_problems: int = None, index_begin: int = 0
    ) -> None:
        """Load PCVRP dataset from pickle file (not implemented)."""
        raise NotImplementedError("Loading PCVRP from pickle is not implemented")

    def load_problem_dataset_pt(self, filename: str, device: torch.device) -> None:
        """
        Load PCVRP dataset from PyTorch file.

        Args:
            filename: Path to .pt file
            device: Device to load tensors on
        """
        self.use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device, weights_only=False)
        self.dataset_depot_xy = loaded_dict["depot_xy"]
        self.dataset_node_xy = loaded_dict["node_xy"]
        self.dataset_node_demand = loaded_dict["node_demand"]
        self.dataset_capacity = loaded_dict["capacity"]
        self.dataset_node_prizes = loaded_dict["node_prizes"]
        self.saved_index = 0
        self.nb_instances = self.dataset_depot_xy.shape[0]

    def init_problems(
        self, nb_instances: int, aug_factor: int = 1
    ) -> Tuple[int, ProblemData, ProblemFeatures]:
        """
        Initialize PCVRP problem instances.

        Args:
            nb_instances: Number of instances to create
            aug_factor: Data augmentation factor

        Returns:
            Tuple of (batch_size, problem_data, problem_features)
        """
        # Get base problem data
        if not self.use_saved_problems:
            (depot_xy, node_xy, node_demand, capacity, node_prizes) = (
                self.get_random_problems(nb_instances)
            )
        else:
            depot_xy = self.dataset_depot_xy[
                self.saved_index : self.saved_index + nb_instances
            ]
            node_xy = self.dataset_node_xy[
                self.saved_index : self.saved_index + nb_instances
            ]
            node_demand = self.dataset_node_demand[
                self.saved_index : self.saved_index + nb_instances
            ]
            capacity = self.dataset_capacity[
                self.saved_index : self.saved_index + nb_instances
            ]
            node_prizes = self.dataset_node_prizes[
                self.saved_index : self.saved_index + nb_instances
            ]
            self.saved_index += nb_instances

        # Apply augmentation
        batch_size, depot_xy, node_xy, augmented = self._apply_augmentation(
            aug_factor,
            nb_instances,
            depot_xy,
            node_xy,
            node_demand,
            capacity,
            node_prizes,
        )
        node_demand, capacity, node_prizes = augmented

        # Build problem data object
        problem_data = ProblemData()
        problem_data.problem_size = node_demand.shape[1]

        # Concatenate depot and node data
        depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros(
            size=(batch_size, 1), dtype=torch.int, device=node_demand.device
        )
        depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        depot_prizes = torch.zeros(
            size=(batch_size, 1), dtype=torch.int, device=node_prizes.device
        )
        depot_node_prizes = torch.cat((depot_prizes, node_prizes), dim=1)

        # Convert to numpy for C++ interface
        problem_data.depot_node_xy = depot_node_xy.cpu().numpy()
        problem_data.depot_node_demand = depot_node_demand.cpu().numpy()
        problem_data.capacity = capacity.cpu().numpy()
        problem_data.depot_node_prizes = depot_node_prizes.cpu().numpy()

        # Build problem features (normalized for neural network)
        problem_feat = ProblemFeatures()
        problem_feat.depot_xy = depot_xy
        problem_feat.node_xy = node_xy
        problem_feat.node_demand = node_demand / capacity  # Normalize demands
        problem_feat.node_prizes = node_prizes

        return batch_size, problem_data, problem_feat

    def get_random_problems(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Generate random PCVRP instances.

        Args:
            batch_size: Number of instances to generate

        Returns:
            Tuple of (depot_xy, node_xy, node_demand, capacity, node_prizes)
        """
        return self.instanceGen.get_random_problems(batch_size)
