"""Environment for VRP problems."""

from dataclasses import dataclass
from typing import Tuple, Optional, Any

import numpy as np
import torch

from .problem_cvrp import ProblemCVRP
from .problem_vrptw import ProblemVRPTW
from .problem_pcvrp import ProblemPCVRP
from .instance_set import InstanceSet


@dataclass
class ResetState:
    """State returned when environment is reset."""

    problem_feat: Any = None
    tour_index: torch.Tensor = None
    neighbours: torch.Tensor = None


@dataclass
class StepState:
    """State maintained during environment stepping."""

    BATCH_IDX: torch.Tensor = None
    ROLLOUT_IDX: torch.Tensor = None
    selected_count: int = None
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None


class Env:
    """
    Environment for VRP problems with learned destroy operations.

    Manages problem instances, solutions, and the selection of nodes to remove
    from current solutions via neural network policy.
    """

    def __init__(self, num_processes: int, **env_params):
        """
        Initialize VRP environment.

        Args:
            use_multiprocessing: Whether to use parallel processing for instances
            **env_params: Environment configuration parameters
        """
        self.env_params = env_params
        self.problem_size = env_params["problem_size"]
        self.num_nodes_to_remove = env_params["num_nodes_to_remove"]

        # Device and sizing (set during init_instances)
        self.device = None
        self.batch_size = None
        self.rollout_size = None

        # Batch/rollout indexing tensors
        self.BATCH_IDX = None
        self.ROLLOUT_IDX = None

        # Initialize problem generator
        self.problem = self._create_problem(
            env_params["problem"],
            self.problem_size,
            env_params.get("generator_params", None),
        )

        # Initialize instance set manager
        starting_solution_params = env_params.get("starting_solution_params", {})
        self.instanceSet = InstanceSet(
            env_params["problem"],
            num_processes,
            starting_solution_params=starting_solution_params,
        )

        # Problem data and features
        self.problem_data = None
        self.problem_feat = None

        # Dynamic state during episode
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.ninf_mask = None
        self.step_state = StepState()

    def _create_problem(
        self, problem_type: str, problem_size: int, generator_params: Optional[dict]
    ):
        """Create problem instance generator based on problem type."""
        if problem_type == "cvrp":
            return ProblemCVRP(problem_size, generator_params)
        elif problem_type == "vrptw":
            return ProblemVRPTW(problem_size, generator_params)
        elif problem_type == "pcvrp":
            return ProblemPCVRP(problem_size, generator_params)
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

    def load_problem_dataset_pkl(
        self, filename: str, num_problems: int, index_begin: int = 0
    ) -> None:
        """Load problem dataset from pickle file."""
        self.problem.load_problem_dataset_pkl(filename, num_problems, index_begin)

    def load_problem_dataset_pt(self, filename: str, device: torch.device) -> None:
        """Load problem dataset from PyTorch file."""
        self.problem.load_problem_dataset_pt(filename, device)

    def init_instances(
        self,
        nb_instances: int,
        rollout_size: int,
        device: torch.device,
        aug_factor: int = 1,
    ) -> None:
        """
        Initialize problem instances and create starting solutions.

        Args:
            nb_instances: Number of problem instances to create
            rollout_size: Number of parallel rollouts per instance
            device: Device to place tensors on
            aug_factor: Data augmentation factor
        """
        self.rollout_size = rollout_size
        self.device = device

        # Generate problem instances
        self.batch_size, self.problem_data, self.problem_feat = (
            self.problem.init_problems(nb_instances, aug_factor)
        )

        # Create batch and rollout index tensors
        self._init_index_tensors()

        # Create initial solutions
        self.instanceSet.init_instances(self.problem_data)
        self.get_model_input(device)

    def _init_index_tensors(self) -> None:
        """Initialize batch and rollout indexing tensors."""
        self.BATCH_IDX = torch.arange(self.batch_size, device=self.device)[
            :, None
        ].expand(self.batch_size, self.rollout_size)
        self.ROLLOUT_IDX = torch.arange(self.rollout_size, device=self.device)[
            None, :
        ].expand(self.batch_size, self.rollout_size)
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.ROLLOUT_IDX = self.ROLLOUT_IDX

    def reset(self) -> StepState:
        """
        Reset environment for new episode.

        Returns:
            Initial step state
        """
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.rollout_size, 0),
            dtype=torch.long,
            device=self.device,
        )

        # Initialize mask (depot cannot be selected)
        self.ninf_mask = torch.zeros(
            size=(self.batch_size, self.rollout_size, self.problem_size + 1),
            device=self.device,
        )
        self.ninf_mask[:, :, 0] = float("-inf")

        # Update step state
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask

        return self.step_state

    def get_model_input(self, device: torch.device) -> ResetState:
        """
        Extract model input features from current solutions.

        Args:
            device: Device to place tensors on

        Returns:
            Reset state with problem features and solution structure
        """
        # Initialize arrays for tour structure
        neighbours = np.zeros((self.batch_size, self.problem_size, 2), dtype=np.int_)
        tour_index = np.zeros((self.batch_size, self.problem_size), dtype=np.int_) - 1

        tours = self.instanceSet.getTours()

        # Extract tour structure for each instance
        for b_idx in range(self.batch_size):
            tour = tours[b_idx]
            # Add depot at start and end of each route
            tour_with_depot = [[0, *route, 0] for route in tour]

            # Extract neighbour relationships and tour assignments
            for tour_idx, route in enumerate(tour_with_depot):
                for pos in range(1, len(route) - 1):
                    customer = route[pos]
                    tour_index[b_idx, customer - 1] = tour_idx
                    neighbours[b_idx, customer - 1] = [route[pos - 1], route[pos + 1]]

        # Create reset state
        reset_state = ResetState()
        reset_state.problem_feat = self.problem_feat
        reset_state.tour_index = torch.tensor(
            tour_index, dtype=torch.long, device=device
        )
        reset_state.neighbours = torch.tensor(
            neighbours, dtype=torch.long, device=device
        )

        return reset_state

    def step(self, selected: torch.Tensor) -> Tuple[StepState, bool]:
        """
        Execute one step of node selection.

        Args:
            selected: Selected nodes (batch, rollout)

        Returns:
            Tuple of (updated step state, done flag)
        """
        # Update selection state
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_node[:, :, None]), dim=2
        )

        # Mask selected nodes
        self.ninf_mask[self.BATCH_IDX, self.ROLLOUT_IDX, selected] = float("-inf")

        # Update step state
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask

        # Check if done
        done = self.selected_count == self.num_nodes_to_remove

        return self.step_state, done
