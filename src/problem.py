"""Base class for VRP problem variants."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any

import torch


class Problem(ABC):
    """Abstract base class for VRP problem variants."""

    def __init__(self, problem_size: int, generator_params: Optional[dict] = None):
        """
        Initialize problem instance generator.

        Args:
            problem_size: Number of customers
            generator_params: Optional parameters for instance generation
        """
        self.problem_size = problem_size
        self.generator_params = generator_params

        # Dataset management
        self.use_saved_problems = False
        self.saved_index = None
        self.nb_instances = None  # Number of instances in loaded dataset

        # Dataset storage (to be populated by subclasses)
        self.dataset_depot_xy = None
        self.dataset_node_xy = None
        self.dataset_node_demand = None
        self.dataset_capacity = None

    @abstractmethod
    def load_problem_dataset_pkl(
        self, filename: str, num_problems: int = None, index_begin: int = 0
    ) -> None:
        """Load problem dataset from pickle file."""
        pass

    @abstractmethod
    def load_problem_dataset_pt(self, filename: str, device: torch.device) -> None:
        """Load problem dataset from PyTorch file."""
        pass

    @abstractmethod
    def init_problems(
        self, nb_instances: int, aug_factor: int = 1
    ) -> Tuple[int, Any, Any]:
        """
        Initialize problem instances.

        Args:
            nb_instances: Number of instances to create
            aug_factor: Data augmentation factor

        Returns:
            Tuple of (batch_size, problem_data, problem_feat)
        """
        pass

    @abstractmethod
    def get_random_problems(self, batch_size: int):
        """Generate random problem instances."""
        pass

    def get_nb_instances(self) -> Optional[int]:
        """
        Get the number of instances in the loaded dataset.

        Returns:
            Number of instances if dataset is loaded, None otherwise
        """
        return self.nb_instances

    def augment_xy_data_by_8_fold(self, xy_data: torch.Tensor) -> torch.Tensor:
        """
        Apply 8-fold augmentation via rotations and reflections.

        Args:
            xy_data: Coordinate data of shape (batch, N, 2)

        Returns:
            Augmented data of shape (8*batch, N, 2)
        """
        x = xy_data[:, :, [0]]
        y = xy_data[:, :, [1]]

        # Generate 8 transformations: 4 rotations + reflections
        dat1 = torch.cat((x, y), dim=2)  # Original
        dat2 = torch.cat((1 - x, y), dim=2)  # Flip X
        dat3 = torch.cat((x, 1 - y), dim=2)  # Flip Y
        dat4 = torch.cat((1 - x, 1 - y), dim=2)  # Flip both
        dat5 = torch.cat((y, x), dim=2)  # Transpose
        dat6 = torch.cat((1 - y, x), dim=2)  # Transpose + Flip X
        dat7 = torch.cat((y, 1 - x), dim=2)  # Transpose + Flip Y
        dat8 = torch.cat((1 - y, 1 - x), dim=2)  # Transpose + Flip both

        aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
        return aug_xy_data

    def _apply_augmentation(
        self,
        aug_factor: int,
        nb_instances: int,
        depot_xy: torch.Tensor,
        node_xy: torch.Tensor,
        *other_tensors: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply data augmentation to problem instances.

        Args:
            aug_factor: Augmentation factor
            nb_instances: Number of base instances
            depot_xy: Depot coordinates
            node_xy: Node coordinates
            *other_tensors: Additional tensors to repeat (demands, capacities, etc.)

        Returns:
            Tuple of (batch_size, augmented_depot_xy, augmented_node_xy, augmented_other_tensors)
        """
        if aug_factor <= 1:
            return nb_instances, depot_xy, node_xy, other_tensors

        if aug_factor == 8:
            # Apply 8-fold geometric augmentation
            batch_size = nb_instances * 8
            depot_xy = self.augment_xy_data_by_8_fold(depot_xy)
            node_xy = self.augment_xy_data_by_8_fold(node_xy)
            augmented_others = tuple(self._repeat_tensor(t, 8) for t in other_tensors)

        elif aug_factor % 8 == 0:
            # Apply 8-fold geometric augmentation + repetition
            batch_size = nb_instances * aug_factor
            depot_xy = self.augment_xy_data_by_8_fold(depot_xy)
            node_xy = self.augment_xy_data_by_8_fold(node_xy)
            depot_xy = depot_xy.repeat(aug_factor // 8, 1, 1)
            node_xy = node_xy.repeat(aug_factor // 8, 1, 1)
            augmented_others = tuple(
                self._repeat_tensor(t, aug_factor) for t in other_tensors
            )

        else:
            # Simple repetition
            batch_size = nb_instances * aug_factor
            depot_xy = depot_xy.repeat(aug_factor, 1, 1)
            node_xy = node_xy.repeat(aug_factor, 1, 1)
            augmented_others = tuple(
                self._repeat_tensor(t, aug_factor) for t in other_tensors
            )

        return batch_size, depot_xy, node_xy, augmented_others

    @staticmethod
    def _repeat_tensor(tensor: torch.Tensor, repeat_factor: int) -> torch.Tensor:
        """Repeat tensor along first dimension."""
        if tensor.dim() == 2:
            return tensor.repeat(repeat_factor, 1)
        elif tensor.dim() == 3:
            return tensor.repeat(repeat_factor, 1, 1)
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")
