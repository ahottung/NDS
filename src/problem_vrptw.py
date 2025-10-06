"""VRPTW (Vehicle Routing Problem with Time Windows) instance generation and management."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from .problem import Problem
from .generator_vrptw import InstanceGenVRPTW


@dataclass
class ProblemData:
    """Container for VRPTW problem instance data."""

    problem_name: str = "vrptw"
    problem_size: int = None
    capacity: np.ndarray = None
    depot_node_xy: np.ndarray = None
    depot_node_demand: np.ndarray = None
    depot_node_tw: np.ndarray = None  # Time windows
    depot_node_sd: np.ndarray = None  # Service duration


@dataclass
class ProblemFeatures:
    """Container for VRPTW problem features (normalized for neural network)."""

    depot_xy: torch.Tensor = None  # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None  # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None  # shape: (batch, problem) - normalized
    depot_tw: torch.Tensor = None  # shape: (batch, 1, 2)
    node_tw: torch.Tensor = None  # shape: (batch, problem, 2)
    node_service_duration: torch.Tensor = None  # shape: (batch, problem)


class ProblemVRPTW(Problem):
    """VRPTW problem instance generator and manager."""

    def __init__(self, problem_size: int, generator_params: dict = None):
        """
        Initialize VRPTW problem generator.

        Args:
            problem_size: Number of customers
            generator_params: Optional parameters for instance generation
        """
        super().__init__(problem_size, generator_params)
        self.name = "vrptw"

        # Additional dataset fields for time windows
        self.dataset_depot_tw = None
        self.dataset_node_tw = None
        self.dataset_service_duration = None

        if generator_params is not None:
            self.instanceGen = InstanceGenVRPTW(self.problem_size, **generator_params)

    def load_problem_dataset_pkl(
        self, filename: str, num_problems: int = None, index_begin: int = 0
    ) -> None:
        """Load VRPTW dataset from pickle file (not implemented)."""
        raise NotImplementedError("Loading VRPTW from pickle is not implemented")

    def load_problem_dataset_pt(self, filename: str, device: torch.device) -> None:
        """
        Load VRPTW dataset from PyTorch file.

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
        self.dataset_depot_tw = loaded_dict["depot_tw"]
        self.dataset_node_tw = loaded_dict["node_tw"]
        self.dataset_service_duration = loaded_dict["node_sd"]
        self.saved_index = 0
        self.nb_instances = self.dataset_depot_xy.shape[0]

    def init_problems(
        self, nb_instances: int, aug_factor: int = 1
    ) -> Tuple[int, ProblemData, ProblemFeatures]:
        """
        Initialize VRPTW problem instances.

        Args:
            nb_instances: Number of instances to create
            aug_factor: Data augmentation factor

        Returns:
            Tuple of (batch_size, problem_data, problem_features)
        """
        # Get base problem data
        if not self.use_saved_problems:
            (depot_xy, node_xy, node_demand, capacity, depot_tw, node_tw, node_sd) = (
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
            depot_tw = self.dataset_depot_tw[
                self.saved_index : self.saved_index + nb_instances
            ]
            node_tw = self.dataset_node_tw[
                self.saved_index : self.saved_index + nb_instances
            ]
            node_sd = self.dataset_service_duration[
                self.saved_index : self.saved_index + nb_instances
            ]
            self.saved_index += nb_instances

        # Apply augmentation (coordinates + other tensors)
        batch_size, depot_xy, node_xy, augmented = self._apply_augmentation(
            aug_factor,
            nb_instances,
            depot_xy,
            node_xy,
            node_demand,
            capacity,
            depot_tw,
            node_tw,
            node_sd,
        )
        node_demand, capacity, depot_tw, node_tw, node_sd = augmented

        # Build problem data object
        problem_data = ProblemData()
        problem_data.problem_size = node_demand.shape[1]

        # Concatenate depot and node data
        depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros(
            size=(batch_size, 1), dtype=torch.int, device=node_demand.device
        )
        depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        depot_node_tw = torch.cat((depot_tw, node_tw), dim=1)
        depot_sd = torch.zeros(
            size=(batch_size, 1), dtype=torch.int, device=node_sd.device
        )
        depot_node_sd = torch.cat((depot_sd, node_sd), dim=1)

        # Convert to numpy for C++ interface
        problem_data.depot_node_xy = depot_node_xy.cpu().numpy()
        problem_data.depot_node_demand = depot_node_demand.cpu().numpy()
        problem_data.capacity = capacity.cpu().numpy()
        problem_data.depot_node_tw = depot_node_tw.cpu().numpy()
        problem_data.depot_node_sd = depot_node_sd.cpu().numpy()

        # Build problem features (normalized for neural network)
        problem_feat = ProblemFeatures()
        problem_feat.depot_xy = depot_xy
        problem_feat.node_xy = node_xy
        problem_feat.node_demand = node_demand / capacity  # Normalize demands
        problem_feat.depot_tw = depot_tw
        problem_feat.node_tw = node_tw
        problem_feat.node_service_duration = node_sd

        return batch_size, problem_data, problem_feat

    def get_random_problems(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Generate random VRPTW instances.

        Args:
            batch_size: Number of instances to generate

        Returns:
            Tuple of (depot_xy, node_xy, node_demand, capacity, depot_tw, node_tw, node_sd)
        """
        return self.instanceGen.get_random_problems(batch_size)
