"""Utility for sampling seed vectors as described in PolyNet paper."""

import itertools
from typing import Optional

import torch


class SeedVectorSampler:
    """
    Manages binary vector pool creation and uniform sampling.

    Creates a pool of binary vectors of dimension z_dim and provides
    efficient sampling.
    """

    def __init__(self, z_dim: int, device: torch.device):
        """
        Initialize the sampler with a binary vector pool.

        Args:
            z_dim: Dimensionality of seed vectors (binary vector length)
            device: Target device for tensor allocation
        """
        self.z_dim = z_dim
        self.device = device
        self.binary_pool = self._create_binary_pool()

    def _create_binary_pool(self) -> torch.Tensor:
        """
        Create a pool of binary seed vectors of dimension z_dim.

        Returns:
            Tensor of shape (2^z_dim, z_dim) containing all binary vectors
        """
        binary_vectors = [list(i) for i in itertools.product([0, 1], repeat=self.z_dim)]
        return torch.tensor(binary_vectors, device=self.device, dtype=torch.float32)

    def sample(self, batch_size: int, rollout_size: int) -> torch.Tensor:
        """
        Sample seed vectors uniformly from the binary pool.

        Args:
            batch_size: Number of problem instances in the batch
            rollout_size: Number of rollouts/samples per instance

        Returns:
            Sampled seed vectors of shape (batch_size, rollout_size, z_dim)
        """
        # Create uniform distribution over all possible binary vectors
        num_vectors = 2**self.z_dim
        uniform_dist = (
            torch.ones(batch_size, num_vectors, device=self.device) / num_vectors
        )

        # Sample rollout_size vectors for each batch element (without replacement)
        z_indices = torch.multinomial(uniform_dist, rollout_size, replacement=False)

        # Index into binary pool and reshape
        z = self.binary_pool[z_indices]  # (batch_size, rollout_size, z_dim)

        # Reshape for compatibility with model architecture
        # (batch, rollout, z_dim) -> (batch, 1, rollout, z_dim) -> transpose
        # -> (batch, rollout, 1, z_dim) -> (batch, rollout, z_dim)
        return (
            z.reshape(batch_size, 1, rollout_size, self.z_dim)
            .transpose(1, 2)
            .reshape(batch_size, rollout_size, self.z_dim)
        )

    @property
    def pool(self) -> torch.Tensor:
        """Access to the underlying binary pool tensor."""
        return self.binary_pool
