"""Logging utilities for training and evaluation."""

import os
import sys
import time
from typing import Dict, Any, Optional, Tuple
import logging


# Global result folder path
_result_folder: str = "./results/{desc}"


def get_result_folder() -> str:
    """Get the current result folder path."""
    return _result_folder


def set_result_folder(folder: str) -> None:
    """Set the result folder path."""
    global _result_folder
    _result_folder = folder


def create_logger(logger_params: Optional[Dict[str, Any]] = None) -> None:
    """
    Create and configure logging system.

    Args:
        logger_params: Dictionary containing logging configuration
    """
    if logger_params is None:
        logger_params = {}

    # Determine log file path
    filepath = logger_params.get("filepath", get_result_folder())
    desc = logger_params.get("desc", "")
    filepath = filepath.format(desc=desc)

    # Set global result folder
    set_result_folder(filepath)

    # Determine log filename
    filename = logger_params.get("filename", "log.txt")
    log_file_path = os.path.join(filepath, filename)

    # Create directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


class AverageMeter:
    """Tracks running average of values."""

    def __init__(self) -> None:
        """Initialize meter."""
        self.reset()

    def reset(self) -> None:
        """Reset meter to initial state."""
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update meter with new value.

        Args:
            val: Value to add
            n: Number of samples (weight)
        """
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        """Get current average."""
        return self.sum / self.count if self.count > 0 else 0.0


class TimeEstimator:
    """Estimates elapsed and remaining time for long-running processes."""

    def __init__(self) -> None:
        """Initialize time estimator."""
        self.logger = logging.getLogger("TimeEstimator")
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count: int = 1) -> None:
        """
        Reset timer to start counting from given count.

        Args:
            count: Starting count (default: 1)
        """
        self.start_time = time.time()
        self.count_zero = count - 1

    def get_est(self, count: int, total: int) -> Tuple[float, float]:
        """
        Get elapsed and remaining time estimates in hours.

        Args:
            count: Current progress count
            total: Total count to complete

        Returns:
            Tuple of (elapsed_hours, remaining_hours)
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Calculate remaining time based on current progress rate
        progress_count = count - self.count_zero
        if progress_count <= 0:
            return 0.0, float("inf")

        remaining_count = total - count
        remaining_time = elapsed_time * remaining_count / progress_count

        # Convert to hours
        elapsed_hours = elapsed_time / 3600.0
        remaining_hours = remaining_time / 3600.0

        return elapsed_hours, remaining_hours

    def get_est_string(self, count: int, total: int) -> Tuple[str, str]:
        """
        Get formatted time estimates as strings.

        Args:
            count: Current progress count
            total: Total count to complete

        Returns:
            Tuple of (elapsed_str, remaining_str)
        """
        elapsed_hours, remaining_hours = self.get_est(count, total)

        # Format elapsed time
        if elapsed_hours >= 1.0:
            elapsed_str = f"{elapsed_hours:.2f}h"
        else:
            elapsed_str = f"{elapsed_hours * 60:.2f}m"

        # Format remaining time
        if remaining_hours >= 1.0:
            remaining_str = f"{remaining_hours:.2f}h"
        else:
            remaining_str = f"{remaining_hours * 60:.2f}m"

        return elapsed_str, remaining_str

    def print_est_time(self, count: int, total: int) -> None:
        """
        Print time estimates to logger.

        Args:
            count: Current progress count
            total: Total count to complete
        """
        elapsed_str, remaining_str = self.get_est_string(count, total)
        self.logger.info(
            f"Epoch {count:3d}/{total:3d}: Time Est.: "
            f"Elapsed[{elapsed_str}], Remain[{remaining_str}]"
        )
