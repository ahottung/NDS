"""Validation helpers for evaluating trained models."""

import os
from logging import getLogger
from typing import Tuple, Dict, Any

import numpy as np
import torch
import wandb

from .env import Env
from .logging_utils import (
    get_result_folder,
    TimeEstimator,
    AverageMeter,
)
from .seed_sampler import SeedVectorSampler


class Validator:
    """Runs periodic validation and aggregates performance metrics."""

    def __init__(
        self,
        device: torch.device,
        env_params: Dict[str, Any],
        trainer_params: Dict[str, Any],
        model_params: Dict[str, Any],
        logger_params: Dict[str, Any],
    ):
        """Initialize validator with configuration parameters."""
        self.env_params = env_params
        self.trainer_params = trainer_params
        self.device = device

        # Setup logging
        self.logger = getLogger(name="validator")
        self.result_folder = get_result_folder()
        self.time_estimator = TimeEstimator()

        # Setup environment
        self.env = Env(**self.env_params)

        # Seed vector sampler
        self.seed_sampler = SeedVectorSampler(model_params["z_dim"], device)

        # Experiment tracking
        self.use_wandb = logger_params["wandb"]["enable"]

    def run(self, model, frozen_model, training_epoch: int) -> float:
        """Run full validation and return augmented score."""
        self.time_estimator.reset()

        # Initialize metrics
        metrics = {
            "score": AverageMeter(),
            "aug_score": AverageMeter(),
            "diversity_overlap": AverageMeter(),
            "unique_rollouts": AverageMeter(),
        }

        # Load validation dataset if specified
        self._load_validation_data()

        # Determine augmentation factor
        aug_factor = self._get_augmentation_factor()

        # Run validation
        self.logger.info("=" * 80)
        self.logger.info("Validation")
        self.logger.info("=" * 80)
        self._validate_all_episodes(model, frozen_model, aug_factor, metrics)

        # Log results
        self._log_validation_results(metrics)

        # Additional greedy evaluation
        greedy_diversity = self._evaluate_greedy_diversity(
            model, frozen_model, aug_factor
        )

        # Log to W&B if enabled
        if self.use_wandb:
            self._log_to_wandb(training_epoch, metrics, greedy_diversity)

        return metrics["aug_score"].avg

    def _load_validation_data(self) -> None:
        """Load validation dataset if specified in config."""
        if not self.trainer_params["valid_data_load"]["enable"]:
            return

        filename = self.trainer_params["valid_data_load"]["filename"]
        extension = os.path.splitext(filename)[1]

        if extension == ".pkl":
            self.env.load_problem_dataset_pkl(
                filename, self.trainer_params["valid_episodes"]
            )
        elif extension == ".pt":
            self.env.load_problem_dataset_pt(filename, self.device)
        else:
            raise ValueError(f"Unsupported dataset format: {extension}")

    def _get_augmentation_factor(self) -> int:
        """Get augmentation factor from config."""
        if self.trainer_params["valid_augmentation_enable"]:
            return self.trainer_params["valid_aug_factor"]
        return 1

    def _validate_all_episodes(
        self, model, frozen_model, aug_factor: int, metrics: Dict[str, AverageMeter]
    ) -> np.ndarray:
        """Run validation across all episodes and return cost logs."""
        validate_num_episode = self.trainer_params["valid_episodes"]
        episode = 0

        while episode < validate_num_episode:
            remaining = validate_num_episode - episode
            batch_size = min(self.trainer_params["valid_batch_size"], remaining)

            # Validate one batch
            score, aug_score, logs_episode, diversity = self._validate_one_batch(
                model,
                frozen_model,
                batch_size,
                self.trainer_params["valid_iterations"],
                aug_factor=aug_factor,
            )

            # Update metrics
            metrics["score"].update(score, batch_size)
            metrics["aug_score"].update(aug_score, batch_size)
            metrics["diversity_overlap"].update(diversity[0], batch_size)
            metrics["unique_rollouts"].update(diversity[1], batch_size)

            episode += batch_size

            # Log progress
            self._log_episode_progress(episode, validate_num_episode, metrics)

    def _validate_one_batch(
        self,
        model,
        frozen_model,
        batch_size: int,
        nb_iterations: int,
        aug_factor: int = 1,
    ) -> Tuple[float, float, np.ndarray, Tuple[float, float]]:
        """Validate one batch and return (score, aug_score, logs, diversity)."""
        rollout_size = self.trainer_params["valid_rollout_size"]
        z_dim = model.model_params["z_dim"]
        recreate_n = self.env_params["recreate_n"]
        beta = self.env_params["beta"]
        insert_in_new_tours_only = self.env_params["insert_in_new_tours_only"]
        aug_batch_size = batch_size * aug_factor

        logs = np.zeros((batch_size, nb_iterations))

        model.eval()
        with torch.no_grad():
            self.env.init_instances(batch_size, rollout_size, self.device, aug_factor)

            for i in range(nb_iterations):
                # Reset and get state
                state = self.env.reset()
                reset_state = self.env.get_model_input(self.device)

                # Sample latent vectors
                z = self.seed_sampler.sample(aug_batch_size, rollout_size)

                # Forward pass
                with torch.amp.autocast(device_type=self.device.type):
                    model.pre_forward(reset_state, z)

                # Rollout
                done = False
                while not done:
                    with torch.amp.autocast(device_type=self.device.type):
                        selected, _, _ = model(state)
                    state, done = self.env.step(selected)

                # Apply repair
                selected_nodes = self.env.selected_node_list.cpu().numpy()
                self.env.instanceSet.remove_recreate(
                    selected_nodes,
                    recreate_n,
                    "allImp",
                    T=0,
                    beta=beta,
                    insert_in_new_tours_only=insert_in_new_tours_only,
                )

                # Log costs (best across augmentations)
                costs = np.array(self.env.instanceSet.costs)
                logs[:, i] = costs.reshape(aug_factor, -1).min(axis=0)

            # Compute diversity metrics
            diversity = self._calculate_diversity(selected_nodes)

            # Compute final scores
            aug_costs = np.array(self.env.instanceSet.costs).reshape(aug_factor, -1)
            no_aug_score = np.mean(aug_costs[0])
            aug_score = np.mean(aug_costs.min(axis=0))

            return no_aug_score, aug_score, logs, diversity

    def _calculate_diversity(self, selected_nodes: np.ndarray) -> Tuple[float, float]:
        """Calculate diversity metrics: overlap score and unique rollout ratio."""
        batch_size, rollout_size, num_nodes = selected_nodes.shape

        overlap_scores = []
        unique_ratios = []

        for i in range(batch_size):
            # Metric 1: Node overlap across rollouts
            total_overlap = 0
            for j in range(rollout_size):
                for node in selected_nodes[i, j]:
                    # Count how many other rollouts contain this node
                    overlap = (selected_nodes[i] == node).any(axis=1).sum() - 1
                    total_overlap += overlap

            # Normalize by total possible overlaps
            max_overlap = (rollout_size - 1) * num_nodes * rollout_size
            overlap_score = total_overlap / max_overlap if max_overlap > 0 else 0
            overlap_scores.append(overlap_score)

            # Metric 2: Ratio of unique rollouts
            sorted_selections = np.sort(selected_nodes[i], axis=1)
            unique_selections = np.unique(sorted_selections, axis=0)
            unique_ratio = unique_selections.shape[0] / rollout_size
            unique_ratios.append(unique_ratio)

        return np.mean(overlap_scores), np.mean(unique_ratios)

    def _evaluate_greedy_diversity(
        self, model, frozen_model, aug_factor: int
    ) -> Tuple[float, float]:
        """Evaluate diversity with greedy (argmax) decoding."""
        original_eval_type = model.model_params["eval_type"]
        model.model_params["eval_type"] = "argmax"
        self.env.problem.saved_index = 0

        _, _, _, greedy_diversity = self._validate_one_batch(
            model,
            frozen_model,
            self.trainer_params["valid_batch_size"],
            self.trainer_params["valid_iterations"],
            aug_factor=aug_factor,
        )

        model.model_params["eval_type"] = original_eval_type
        return greedy_diversity

    def _log_episode_progress(
        self, episode: int, total: int, metrics: Dict[str, AverageMeter]
    ) -> None:
        """Log progress for current episode."""
        elapsed, remaining = self.time_estimator.get_est_string(episode, total)
        self.logger.info(
            f"Episode {episode:3d}/{total:3d}  |  "
            f"Elapsed: {elapsed}  |  Remain: {remaining}  |  "
            f"Score: {metrics['score'].avg:7.2f}  |  Aug Score: {metrics['aug_score'].avg:7.2f}"
        )

    def _log_validation_results(self, metrics: Dict[str, AverageMeter]) -> None:
        """Log final validation results."""
        self.logger.info("=" * 80)
        self.logger.info("Validation Complete")
        self.logger.info("=" * 80)
        self.logger.info(f"No-Aug Score:    {metrics['score'].avg:7.3f}")
        self.logger.info(f"Aug Score:       {metrics['aug_score'].avg:7.3f}")
        self.logger.info(f"Diversity Score: {metrics['diversity_overlap'].avg:7.4f}")
        self.logger.info(f"Unique Rollouts: {metrics['unique_rollouts'].avg:7.4f}")
        self.logger.info("=" * 80)

    def _log_to_wandb(
        self,
        training_epoch: int,
        metrics: Dict[str, AverageMeter],
        greedy_diversity: Tuple[float, float],
    ) -> None:
        """Log validation metrics to Weights & Biases."""
        wandb.log(
            step=training_epoch,
            data={
                "val/no_aug_score": metrics["score"].avg,
                "val/aug_score": metrics["aug_score"].avg,
                "val/diversity_score": metrics["diversity_overlap"].avg,
                "val/unique_rollouts": metrics["unique_rollouts"].avg,
                "val/greedy_unique_rollout": greedy_diversity[1],
            },
        )
