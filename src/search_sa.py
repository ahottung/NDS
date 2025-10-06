"""Simulated Annealing search for solving VRP instances one at a time."""

import os
import time
import random
import itertools
import csv
from logging import getLogger
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch

from .env import Env
from .model import Model
from .logging_utils import (
    get_result_folder,
    TimeEstimator,
    AverageMeter,
)
from .seed_sampler import SeedVectorSampler


class Search:
    """Simulated Annealing search that solves one VRP instance at a time."""

    def __init__(
        self,
        env_params: Dict[str, Any],
        tester_params: Dict[str, Any],
    ):
        """Initialize search with configuration parameters."""
        self.env_params = env_params
        self.tester_params = tester_params

        # Setup logging
        self.logger = getLogger(name="tester")
        self.result_folder = get_result_folder()
        self.time_estimator = TimeEstimator()

        # Setup device
        self.device = self._setup_device()

        # Load problem-specific C++ operations
        self.NDSOps = self._load_cpp_operations()

        # Load trained models for learned destroy operations
        self.destroy_operators = self._load_destroy_operators()

        # Setup environment (single instance at a time)
        self.env = Env(False, **self.env_params)

    def _setup_device(self) -> torch.device:
        """Setup and return the compute device (CPU or CUDA)."""
        use_cuda = self.tester_params["use_cuda"]
        if use_cuda:
            cuda_device_num = self.tester_params["cuda_device_num"]
            torch.cuda.set_device(cuda_device_num)
            return torch.device("cuda", cuda_device_num)
        return torch.device("cpu")

    def _load_cpp_operations(self):
        """Load problem-specific C++ operations module."""
        problem_type = self.env_params["problem"]

        if problem_type == "cvrp":
            from .cpp.cvrp import NDSOps
        elif problem_type == "vrptw":
            from .cpp.vrptw import NDSOps
        elif problem_type == "pcvrp":
            from .cpp.pcvrp import NDSOps
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        return NDSOps

    def _load_destroy_operators(self) -> List[Dict[str, Any]]:
        """Load trained neural network models for destroy operations."""
        if self.tester_params["use_baseline_destroy"]:
            return []

        operators = []
        for model_config in self.tester_params["model_load"]:
            checkpoint_path = "{path}/checkpoint-{epoch}.pt".format(**model_config)
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            model_params = checkpoint["model_params"]

            # Create and load model
            model = Model(**model_params).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            # Create seed vector sampler
            seed_sampler = SeedVectorSampler(model_params["z_dim"], self.device)

            # Verify configuration matches
            assert (
                checkpoint["env_params"]["num_nodes_to_remove"]
                == model_config["node_to_remove"]
            ), f"Model trained with different num_nodes_to_remove: {checkpoint['env_params']['num_nodes_to_remove']} vs {model_config['node_to_remove']}"

            operators.append(
                {"model": model, "seed_sampler": seed_sampler, **model_config}
            )

            self.logger.info(f"Loaded deconstruction policy from {checkpoint_path}")

        return operators

    def run(self) -> None:
        """Run search on all test instances and save results."""
        self.time_estimator.reset()

        # Initialize metrics
        metrics = {
            "costs": AverageMeter(),
            "runtime": AverageMeter(),
            "iterations": AverageMeter(),
        }

        # Load test dataset if specified
        self._load_test_dataset()

        # Process test instances
        nb_instances = self.tester_params.get("nb_instances")
        if nb_instances is None:
            total_instances = self.env.problem.nb_instances
        else:
            total_instances = nb_instances
        self.logger.info("=" * 80)
        self.logger.info(f"Starting search on {total_instances} instances")
        self.logger.info("=" * 80)

        for instance_idx in range(total_instances):
            # Solve one instance
            result = self._solve_one_instance(instance_idx)

            # Update metrics
            metrics["costs"].update(result["cost"], 1)
            metrics["runtime"].update(result["runtime"], 1)
            metrics["iterations"].update(result["nb_iterations"], 1)

            # Log progress
            self._log_instance_progress(instance_idx, total_instances, result, metrics)

            # Save results
            self._save_instance_results(instance_idx, result)

        # Log final summary
        self._log_final_summary(metrics)

    def _load_test_dataset(self) -> None:
        """Load test dataset if specified in config."""
        if not self.tester_params["test_data_load"]["enable"]:
            return

        filename = self.tester_params["test_data_load"]["filename"]
        extension = os.path.splitext(filename)[1]

        if extension == ".pkl":
            # For .pkl files, pass None to load all instances
            nb_instances = self.tester_params.get("nb_instances")
            self.env.load_problem_dataset_pkl(filename, nb_instances)
        elif extension == ".pt":
            self.env.load_problem_dataset_pt(filename, self.device)
        else:
            raise ValueError(f"Unsupported dataset format: {extension}")

        self.logger.info(f"Loaded test dataset from {filename}")

    def _solve_one_instance(self, instance_idx: int) -> Dict[str, Any]:
        """
        Solve a single VRP instance using Simulated Annealing with learned destroy operations.

        Returns dictionary with 'cost', 'runtime', 'nb_iterations', 'solution'.
        """
        aug_factor = self.tester_params["aug_factor"]
        max_iterations = self.tester_params["nb_iterations"]
        rollout_size = self.tester_params["rollout_size"]

        # Initialize SA parameters
        sa_config = self._init_simulated_annealing()

        # Initialize instance with augmentation
        start_time = time.time()
        self.env.init_instances(1, rollout_size, self.device, aug_factor)

        # Track best solution
        incumbent_cost = np.inf
        incumbent_solution = None

        # Simulated Annealing loop
        iteration = 0
        while iteration < max_iterations:
            # Perform one SA iteration
            new_solutions = self._perform_sa_iteration(
                aug_factor, rollout_size, sa_config["T"]
            )

            # Update incumbent
            for sol in new_solutions:
                if sol.totalCosts < incumbent_cost:
                    incumbent_cost = sol.totalCosts
                    incumbent_solution = sol

            # Synchronize augmented solutions (only when aug_factor > 1)
            if aug_factor > 1:
                self._synchronize_augmented_solutions(
                    new_solutions, sa_config["T"], sa_config["delta"]
                )

            # Update solutions in environment
            for idx, sol in enumerate(new_solutions):
                self.env.instanceSet.set_solution(idx, sol)

            # Update temperature
            sa_config["T"] = self._update_temperature(
                sa_config, iteration, start_time, max_iterations
            )

            iteration += 1

            # Check runtime limit
            if (
                sa_config["runtime_limited"]
                and (time.time() - start_time) > sa_config["max_runtime"]
            ):
                break

        runtime = time.time() - start_time

        # Verify runtime limit was enforced correctly
        if sa_config["runtime_limited"]:
            assert (
                runtime > sa_config["max_runtime"]
            ), "Runtime limit was set, but search terminated based on iteration count"

        return {
            "cost": incumbent_cost,
            "runtime": runtime,
            "nb_iterations": iteration,
            "solution": incumbent_solution,
        }

    def _init_simulated_annealing(self) -> Dict[str, Any]:
        """Initialize Simulated Annealing parameters."""
        max_runtime = self.tester_params["max_runtime"]
        runtime_limited = max_runtime > 0

        T_0 = self.tester_params["SA_start_T"]
        T_f = self.tester_params["SA_final_T"]

        config = {
            "T": T_0,
            "T_0": T_0,
            "T_f": T_f,
            "delta": self.tester_params["SA_delta"],
            "max_runtime": max_runtime,
            "runtime_limited": runtime_limited,
        }

        # Compute cooling rate if not runtime-limited
        if not runtime_limited:
            nb_iterations = self.tester_params["nb_iterations"]
            config["cooling_rate"] = (T_f / T_0) ** (1 / nb_iterations)

        return config

    def _perform_sa_iteration(
        self, aug_factor: int, rollout_size: int, temperature: float
    ) -> List[Any]:
        """
        Perform one Simulated Annealing iteration: destroy and repair.

        Returns list of new solutions (one per augmentation).
        """
        use_model = len(self.destroy_operators) > 0
        beta = self.env_params["beta"]
        insert_in_new_tours_only = self.env_params["insert_in_new_tours_only"]
        recreate_n = self.env_params["recreate_n"]

        # If using model, we need to perform destroy for all augmentations at once
        if use_model:
            all_selected_nodes = self._select_nodes_with_model(aug_factor, rollout_size)

        new_solutions = []

        # Perform destroy-repair for each augmentation
        for aug_idx in range(aug_factor):
            current_solution = self.env.instanceSet.get_solution(aug_idx)

            if use_model:
                # Use learned destroy operator (already computed)
                selected_nodes = all_selected_nodes[aug_idx]
            else:
                # Use heuristic destroy operator
                selected_nodes = self.NDSOps.heuristic_deconstruction_selection(
                    current_solution, self.env.num_nodes_to_remove, rollout_size
                )

            # Repair solution (with simulated annealing acceptance)
            new_solution, _ = self.NDSOps.remove_recreate_allImp(
                current_solution,
                selected_nodes,
                beta,
                recreate_n,
                temperature,
                insert_in_new_tours_only,
            )

            new_solutions.append(new_solution)

        return new_solutions

    def _select_nodes_with_model(
        self, aug_factor: int, rollout_size: int
    ) -> np.ndarray:
        """
        Use learned neural network model to select nodes for removal.

        Returns array of shape (aug_factor, rollout_size, num_nodes_to_remove).
        """
        # Randomly choose one destroy operator
        operator = random.choice(self.destroy_operators)
        model = operator["model"]
        self.env.num_nodes_to_remove = operator["node_to_remove"]

        # Reset environment and get state
        state = self.env.reset()
        reset_state = self.env.get_model_input(self.device)

        # Sample latent vectors for all augmentations
        z = operator["seed_sampler"].sample(aug_factor, rollout_size)

        # Forward pass through model
        softmax_temp = self.tester_params["softmax_temp"]
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type):
                model.pre_forward(reset_state, z)

            # Rollout to select nodes
            done = False
            while not done:
                with torch.amp.autocast(device_type=self.device.type):
                    selected, _, _ = model(state, softmax_temp)
                state, done = self.env.step(selected)

        # Extract selected nodes for all augmentations
        selected_nodes = self.env.selected_node_list.cpu().numpy()
        return selected_nodes

    def _synchronize_augmented_solutions(
        self, solutions: List[Any], temperature: float, delta: float
    ) -> None:
        """
        Synchronize solutions across augmentations by replacing poor solutions
        with good candidates (within temperature threshold).
        """
        costs = np.array([sol.totalCosts for sol in solutions])
        min_cost = np.min(costs)

        # Find candidate solutions (within threshold)
        threshold = min_cost + temperature * delta
        candidate_indices = np.where(costs < threshold)[0]

        if len(candidate_indices) == 0:
            return

        # Replace solutions that are too expensive
        for idx in range(len(solutions)):
            if costs[idx] > threshold:
                replacement_idx = np.random.choice(candidate_indices)
                solutions[idx] = solutions[replacement_idx]

    def _update_temperature(
        self,
        sa_config: Dict[str, Any],
        iteration: int,
        start_time: float,
        max_iterations: int,
    ) -> float:
        """Update and return new temperature for Simulated Annealing."""
        if sa_config["runtime_limited"]:
            # Runtime-based cooling schedule
            elapsed = time.time() - start_time
            max_runtime = sa_config["max_runtime"]
            progress = min(1.0, elapsed / max_runtime)
            T = sa_config["T_f"] * (sa_config["T_0"] / sa_config["T_f"]) ** (
                1 - progress
            )
        else:
            # Iteration-based geometric cooling
            T = sa_config["T"] * sa_config["cooling_rate"]

        return T

    def _log_instance_progress(
        self,
        instance_idx: int,
        total: int,
        result: Dict[str, Any],
        metrics: Dict[str, AverageMeter],
    ) -> None:
        """Log progress for current instance."""
        elapsed, remaining = self.time_estimator.get_est_string(instance_idx + 1, total)
        self.logger.info(
            f"Instance {instance_idx + 1:3d}/{total:3d}  |  "
            f"Elapsed: {elapsed}  |  Remain: {remaining}  |  "
            f"Cost: {result['cost']:7.2f}  |  Avg: {metrics['costs'].avg:7.3f}  |  "
            f"Iters: {result['nb_iterations']:4.0f}"
        )

    def _log_final_summary(self, metrics: Dict[str, AverageMeter]) -> None:
        """Log final summary statistics."""
        self.logger.info("=" * 80)
        self.logger.info("Search Complete")
        self.logger.info("=" * 80)
        self.logger.info(f"Average Cost:       {metrics['costs'].avg:7.3f}")
        self.logger.info(f"Average Runtime:    {metrics['runtime'].avg:7.2f} seconds")
        self.logger.info(f"Average Iterations: {metrics['iterations'].avg:7.1f}")
        self.logger.info("=" * 80)

    def _save_instance_results(self, instance_idx: int, result: Dict[str, Any]) -> None:
        """Save instance results to CSV files."""
        # Save cost and runtime statistics
        results_path = os.path.join(self.result_folder, "results.csv")
        with open(results_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    instance_idx + 1,
                    result["cost"],
                    result["runtime"],
                    result["nb_iterations"],
                ]
            )

        # Save solution tours
        if result["solution"] is not None:
            solutions_path = os.path.join(self.result_folder, "solutions.csv")
            tours = result["solution"].getTourList()
            # Add depot (node 0) at start and end of each tour
            tours_with_depot = [[0, *tour, 0] for tour in tours]

            with open(solutions_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([instance_idx + 1, tours_with_depot])
