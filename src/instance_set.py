"""Instance management for VRP problems with multiprocessing support."""

import math
import multiprocessing
from typing import List, Tuple, Any, Optional

import cppimport.import_hook
import numpy as np


def _load_cpp_operations(problem: str):
    """Load problem-specific C++ operations module."""
    if problem == "cvrp":
        from .cpp.cvrp import NDSOps
    elif problem == "vrptw":
        from .cpp.vrptw import NDSOps
    elif problem == "pcvrp":
        from .cpp.pcvrp import NDSOps
    else:
        raise ValueError(f"Unsupported problem type: {problem}")

    return NDSOps


def _create_starting_solution(
    NDSOps,
    instance,
    problem_size: int,
    nb_iterations: int = 50,
    nb_nodes_ratio: float = 0.15,
):
    """
    Create initial solution using heuristics.

    Args:
        NDSOps: C++ operations module
        instance: Problem instance
        problem_size: Number of customers
        nb_iterations: Number of improvement iterations
        nb_nodes_ratio: Ratio of nodes to perturb (relative to problem size)

    Returns:
        Initial solution
    """
    nb_nodes = int(problem_size * nb_nodes_ratio)
    return NDSOps.create_starting_solution(instance, nb_iterations, nb_nodes)


def _create_instance(
    NDSOps, problem: str, problem_size: int, i: int, problem_data
) -> Any:
    """Create a single c++ instance objecte based on problem type."""
    if problem == "cvrp":
        return NDSOps.Instance(
            problem_size,
            problem_data.capacity[i],
            problem_data.depot_node_demand[i],
            problem_data.depot_node_xy[i],
        )
    elif problem == "vrptw":
        return NDSOps.Instance(
            problem_size,
            problem_data.capacity[i],
            problem_data.depot_node_demand[i],
            problem_data.depot_node_tw[i, :, 0],
            problem_data.depot_node_tw[i, :, 1],
            problem_data.depot_node_sd[i],
            problem_data.depot_node_xy[i],
        )
    elif problem == "pcvrp":
        return NDSOps.Instance(
            problem_size,
            problem_data.capacity[i],
            problem_data.depot_node_demand[i],
            problem_data.depot_node_xy[i],
            problem_data.depot_node_prizes[i],
        )
    else:
        raise ValueError(f"Unsupported problem type: {problem}")


def _extract_problem_data_slice(problem_data, start_idx: int, end_idx: int) -> List:
    """Extract a slice of problem data for a worker process."""
    problem = problem_data.problem_name

    if problem == "cvrp":
        return [
            problem_data.problem_size,
            problem_data.capacity,
            problem_data.depot_node_demand[start_idx:end_idx],
            problem_data.depot_node_xy[start_idx:end_idx],
        ]
    elif problem == "vrptw":
        return [
            problem_data.problem_size,
            problem_data.capacity,
            problem_data.depot_node_demand[start_idx:end_idx],
            problem_data.depot_node_xy[start_idx:end_idx],
            problem_data.depot_node_tw[start_idx:end_idx],
            problem_data.depot_node_sd[start_idx:end_idx],
        ]
    elif problem == "pcvrp":
        return [
            problem_data.problem_size,
            problem_data.capacity,
            problem_data.depot_node_demand[start_idx:end_idx],
            problem_data.depot_node_xy[start_idx:end_idx],
            problem_data.depot_node_prizes[start_idx:end_idx],
        ]
    else:
        raise ValueError(f"Unsupported problem type: {problem}")


def worker(
    problem: str,
    input_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    starting_solution_params: dict,
) -> None:
    """
    Worker process for parallel instance processing.

    Handles two operations:
    - 'new_instance': Initialize new problem instances
    - 'remove_recreate': Apply destroy-repair operations to solutions

    Args:
        problem: Problem type ('cvrp', 'vrptw', or 'pcvrp')
        input_queue: Queue for receiving commands
        result_queue: Queue for sending results
        starting_solution_params: Parameters for starting solution generation
    """
    NDSOps = _load_cpp_operations(problem)

    instances = []  # Do not delete. Storing the instances keeps the c++ objects alive.
    solutions = []
    solution_costs = []
    tours = []

    try:
        while True:
            mode, data = input_queue.get()

            if mode == "new_instance":
                instances, solutions, solution_costs, tours = _handle_new_instances(
                    NDSOps, problem, data, starting_solution_params
                )
                result_queue.put([solution_costs, tours])

            elif mode == "remove_recreate":
                candidate_costs = _handle_remove_recreate(
                    NDSOps, solutions, solution_costs, tours, data
                )
                result_queue.put([candidate_costs, solution_costs, tours])

    except Exception as error:
        print(f"Worker exception occurred: {error}")


def _handle_new_instances(
    NDSOps, problem: str, data: List, starting_solution_params: dict
) -> Tuple[List, List, List, List]:
    """Initialize new problem instances and create starting solutions."""
    instances = []
    solutions = []
    solution_costs = []
    tours = []

    # Unpack data based on problem type
    if problem == "cvrp":
        problem_size, capacity, depot_node_demand_np, depot_node_xy_np = data
    elif problem == "vrptw":
        (
            problem_size,
            capacity,
            depot_node_demand_np,
            depot_node_xy_np,
            depot_node_tw_np,
            depot_node_sd_np,
        ) = data
    elif problem == "pcvrp":
        (
            problem_size,
            capacity,
            depot_node_demand_np,
            depot_node_xy_np,
            depot_node_prizes_np,
        ) = data
    else:
        raise ValueError(f"Unsupported problem type: {problem}")

    # Create instances and solutions
    for i in range(depot_node_xy_np.shape[0]):
        if problem == "cvrp":
            instance = NDSOps.Instance(
                problem_size, capacity[i], depot_node_demand_np[i], depot_node_xy_np[i]
            )
        elif problem == "vrptw":
            instance = NDSOps.Instance(
                problem_size,
                capacity[i],
                depot_node_demand_np[i],
                depot_node_tw_np[i, :, 0],
                depot_node_tw_np[i, :, 1],
                depot_node_sd_np[i],
                depot_node_xy_np[i],
            )
        elif problem == "pcvrp":
            instance = NDSOps.Instance(
                problem_size,
                capacity[i],
                depot_node_demand_np[i],
                depot_node_xy_np[i],
                depot_node_prizes_np[i],
            )

        solution = _create_starting_solution(
            NDSOps, instance, problem_size, **starting_solution_params
        )

        instances.append(instance)
        solutions.append(solution)
        solution_costs.append(solution.totalCosts)
        tours.append(solution.getTourList())

    return instances, solutions, solution_costs, tours


def _handle_remove_recreate(
    NDSOps, solutions: List, solution_costs: List, tours: List, data: Tuple
) -> List:
    """Apply remove-repair operations to solutions."""
    selected_nodes, recreate_n, T, beta, insert_in_new_tours_only, search_mode = data

    candidate_costs = []
    assert (
        len(solutions) == selected_nodes.shape[0]
    ), "Number of solutions must match number of selected node arrays"

    for i in range(len(solutions)):
        if search_mode == "allImp":
            best_soln, _ = NDSOps.remove_recreate_allImp(
                solutions[i],
                selected_nodes[i],
                beta,
                recreate_n,
                T,
                insert_in_new_tours_only,
            )
        elif search_mode == "singleImp":
            best_soln, c_costs = NDSOps.remove_recreate_singleImp(
                solutions[i],
                selected_nodes[i],
                beta,
                recreate_n,
                insert_in_new_tours_only,
            )
            candidate_costs.append(c_costs)
        else:
            raise ValueError(f"Unsupported search mode: {search_mode}")

        solutions[i] = best_soln
        solution_costs[i] = best_soln.totalCosts
        tours[i] = best_soln.getTourList()

    return candidate_costs


class InstanceSet:
    """
    Manages a set of VRP c++ instances objects and their solutions.

    Supports both single-process and multi-process execution modes for
    parallel instance processing and solution operations.
    """

    def __init__(
        self,
        problem: str,
        num_processes: int,
        starting_solution_params: Optional[dict] = None,
    ):
        """
        Initialize instance set manager.

        Args:
            problem: Problem type ('cvrp', 'vrptw', or 'pcvrp')
            use_multiprocessing: Whether to use parallel processing
            num_processes: Number of worker processes (if multiprocessing enabled)
            starting_solution_params: Parameters for starting solution generation
        """
        self.problem = problem
        self.batch_size = None
        self.tours = []
        self.costs = []

        # Starting solution parameters with defaults
        self.starting_solution_params = starting_solution_params or {}

        # Single-process mode state
        self._instances = []
        self._solutions = []

        # Multi-process mode setup
        self.use_multiprocessing = num_processes > 1
        self.processes = []

        if self.use_multiprocessing:
            self.num_processes = num_processes
            self._init_worker_processes()
        else:
            self.NDSOps = _load_cpp_operations(problem)

    def _init_worker_processes(self) -> None:
        """Initialize worker processes for parallel execution."""
        for _ in range(self.num_processes):
            input_queue = multiprocessing.Queue()
            output_queue = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=worker,
                args=(
                    self.problem,
                    input_queue,
                    output_queue,
                    self.starting_solution_params,
                ),
            )
            p.start()
            self.processes.append([p, input_queue, output_queue])

    def __del__(self):
        """Clean up worker processes on deletion."""
        if self.use_multiprocessing:
            for p, _, _ in self.processes:
                p.terminate()

    def init_instances(self, problem_data) -> None:
        """
        Initialize problem instances from data.

        Args:
            problem_data: Object containing problem instance data
        """
        if self.use_multiprocessing:
            self._init_instances_mp(problem_data)
        else:
            self._init_instances_sp(problem_data)

    def remove_recreate(
        self,
        selected_nodes: np.ndarray,
        recreate_n: int,
        mode: str,
        T: float = 0,
        beta: float = 0.0,
        insert_in_new_tours_only: bool = True,
    ) -> List:
        """
        Apply remove-repair operations to solutions.

        Args:
            selected_nodes: Nodes to remove from solutions
            recreate_n: Number of reinsertion attempts
            mode: Search mode ('allImp' or 'singleImp')
            T: Temperature for simulated annealing
            beta: Regret parameter for insertion
            insert_in_new_tours_only: Whether to only insert into new tours

        Returns:
            List of candidate costs for each solution
        """
        if self.use_multiprocessing:
            return self._remove_recreate_mp(
                selected_nodes, recreate_n, mode, T, beta, insert_in_new_tours_only
            )
        else:
            return self._remove_recreate_sp(
                selected_nodes, recreate_n, mode, T, beta, insert_in_new_tours_only
            )

    def _init_instances_mp(self, problem_data) -> None:
        """Initialize instances using multiprocessing."""
        self.batch_size = problem_data.depot_node_xy.shape[0]
        instances_per_process = math.ceil(self.batch_size / self.num_processes)

        # Distribute work to processes
        for idx, (p, p_in, p_out) in enumerate(self.processes):
            start_idx = idx * instances_per_process
            end_idx = start_idx + instances_per_process
            p_data = _extract_problem_data_slice(problem_data, start_idx, end_idx)
            p_in.put(["new_instance", p_data])

        # Collect results
        self.tours = []
        self.costs = []
        for p, p_in, p_out in self.processes:
            costs, tours = p_out.get()
            self.costs.extend(costs)
            self.tours.extend(tours)

    def _remove_recreate_mp(
        self,
        selected_nodes: np.ndarray,
        recreate_n: int,
        mode: str,
        T: float,
        beta: float,
        insert_in_new_tours_only: bool,
    ) -> List:
        """Apply remove-recreate operations using multiprocessing."""
        instances_per_process = math.ceil(self.batch_size / self.num_processes)

        # Distribute work to processes
        for idx, (p, p_in, p_out) in enumerate(self.processes):
            start_idx = idx * instances_per_process
            end_idx = start_idx + instances_per_process
            p_data = [
                selected_nodes[start_idx:end_idx],
                recreate_n,
                T,
                beta,
                insert_in_new_tours_only,
                mode,
            ]
            p_in.put(["remove_recreate", p_data])

        # Collect results
        self.tours = []
        self.costs = []
        candidate_costs_set = []
        for p, p_in, p_out in self.processes:
            candidate_costs, costs, tours = p_out.get()
            self.costs.extend(costs)
            self.tours.extend(tours)
            candidate_costs_set.extend(candidate_costs)

        return candidate_costs_set

    def _init_instances_sp(self, problem_data) -> None:
        """Initialize instances in single-process mode."""
        self.batch_size = problem_data.depot_node_xy.shape[0]
        problem_size = problem_data.problem_size

        self._instances = []
        self._solutions = []
        self.costs = []
        self.tours = []

        for i in range(self.batch_size):
            instance = _create_instance(
                self.NDSOps, self.problem, problem_size, i, problem_data
            )
            solution = _create_starting_solution(
                self.NDSOps, instance, problem_size, **self.starting_solution_params
            )

            self._instances.append(instance)
            self._solutions.append(solution)
            self.costs.append(solution.totalCosts)
            self.tours.append(solution.getTourList())

    def _remove_recreate_sp(
        self,
        selected_nodes: np.ndarray,
        recreate_n: int,
        mode: str,
        T: float,
        beta: float,
        insert_in_new_tours_only: bool,
    ) -> List:
        """Apply remove-recreate operations in single-process mode."""
        candidate_costs_set = []

        for i in range(self.batch_size):
            if mode == "allImp":
                best_soln, _ = self.NDSOps.remove_recreate_allImp(
                    self._solutions[i],
                    selected_nodes[i],
                    beta,
                    recreate_n,
                    T,
                    insert_in_new_tours_only,
                )
            elif mode == "singleImp":
                best_soln, candidate_costs = self.NDSOps.remove_recreate_singleImp(
                    self._solutions[i],
                    selected_nodes[i],
                    beta,
                    recreate_n,
                    insert_in_new_tours_only,
                )
                candidate_costs_set.append(candidate_costs)
            else:
                raise ValueError(f"Unsupported search mode: {mode}")

            self._solutions[i] = best_soln
            self.costs[i] = best_soln.totalCosts
            self.tours[i] = best_soln.getTourList()

        return candidate_costs_set

    def getTours(self) -> List:
        """Get tours for all solutions."""
        return self.tours

    def get_solution(self, idx: int):
        """Get solution at specified index (single-process mode only)."""
        return self._solutions[idx]

    def set_solution(self, idx: int, sol) -> None:
        """
        Update solution at specified index (single-process mode only).

        Args:
            idx: Solution index
            sol: New solution object
        """
        self._solutions[idx] = sol
        self.costs[idx] = sol.totalCosts
        self.tours[idx] = sol.getTourList()
