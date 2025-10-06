"""Training loop and utilities for model optimization."""

import os
import time
from logging import getLogger
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from .env import Env
from .model import Model
from .logging_utils import (
    get_result_folder,
    TimeEstimator,
    AverageMeter,
)
from .validator import Validator
from .seed_sampler import SeedVectorSampler
import wandb


class Trainer:
    """Manages the full training lifecycle: setup, training loop, logging, and checkpoints."""

    def __init__(
        self,
        env_params: Dict[str, Any],
        model_params: Dict[str, Any],
        optimizer_params: Dict[str, Any],
        trainer_params: Dict[str, Any],
        logger_params: Dict[str, Any],
    ):
        """Initialize trainer with configuration parameters."""
        # Store configuration
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # Setup logging and directories
        self.logger = getLogger(name="trainer")
        self.results_dir = get_result_folder()
        self.time_estimator = TimeEstimator()

        # Setup device
        self.device = self._setup_device()

        # Initialize core components
        self.model = Model(**self.model_params).to(self.device)
        self.model_frozen = Model(**self.model_params).to(self.device)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(
            self.model.parameters(), **self.optimizer_params["optimizer"]
        )
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params["scheduler"])
        # Use new AMP GradScaler API; enable scaling only on CUDA
        if self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = torch.amp.GradScaler(enabled=False)

        # Training parameters
        self.batch_size = self.trainer_params["train_batch_size"]
        self.rollout_size = self.trainer_params["rollout_size"]

        # Seed vector sampler
        self.seed_sampler = SeedVectorSampler(model_params["z_dim"], self.device)

        # Restore from checkpoint if needed
        self.start_epoch = 1
        self.wandb_run_id = None
        self._load_checkpoint_if_exists()

        # Setup validator
        self.validator = Validator(
            self.device,
            self.env_params,
            self.trainer_params,
            self.model_params,
            logger_params,
        )

        # Setup experiment tracking
        self.use_wandb = logger_params["wandb"]["enable"]
        if self.use_wandb:
            self._init_wandb(
                logger_params,
                env_params,
                model_params,
                optimizer_params,
                trainer_params,
            )

    def _setup_device(self) -> torch.device:
        """Setup and return the compute device (CPU or CUDA)."""
        use_cuda = self.trainer_params["use_cuda"]
        if use_cuda:
            cuda_device_num = self.trainer_params["cuda_device_num"]
            torch.cuda.set_device(cuda_device_num)
            return torch.device("cuda", cuda_device_num)
        return torch.device("cpu")

    def _load_checkpoint_if_exists(self) -> None:
        """Load model from explicit checkpoint or auto-resume from latest."""
        # Load from explicit checkpoint if specified
        model_load = self.trainer_params["model_load"]
        if model_load["enable"]:
            checkpoint_path = "{path}/checkpoint-{epoch}.pt".format(**model_load)
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.logger.info(f"Loaded model from {checkpoint_path}")

        # Auto-resume from latest if exists
        latest_path = os.path.join(self.results_dir, "latest_model.pt")
        if os.path.isfile(latest_path):
            checkpoint = torch.load(
                latest_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.last_epoch = checkpoint["epoch"] - 1
            self.start_epoch = 1 + checkpoint["epoch"]
            self.wandb_run_id = checkpoint.get("wandb_run_id")
            self.logger.info(f"Resuming from epoch {self.start_epoch}")

    def _init_wandb(
        self,
        logger_params: Dict[str, Any],
        env_params: Dict[str, Any],
        model_params: Dict[str, Any],
        optimizer_params: Dict[str, Any],
        trainer_params: Dict[str, Any],
    ) -> None:
        """Initialize Weights & Biases experiment tracking."""
        run = wandb.init(
            project=logger_params["wandb"]["project"],
            name=logger_params["desc"],
            config={
                "env_params": env_params,
                "model_params": model_params,
                "optimizer_params": optimizer_params,
                "trainer_params": trainer_params,
            },
            id=self.wandb_run_id,
            resume="allow",
        )
        self.wandb_run_id = run.id

    def run(self) -> None:
        """Execute the main training loop across all epochs."""
        self.time_estimator.reset(self.start_epoch)
        total_epochs = self.trainer_params["epochs"]

        for epoch in range(self.start_epoch, total_epochs + 1):
            self.logger.info("=" * 80)

            # Train one epoch
            metrics = self._train_one_epoch(epoch)

            # Update learning rate
            self.scheduler.step()

            # Log timing and save checkpoints
            self._log_timing(epoch, total_epochs)
            self._save_checkpoints(epoch, total_epochs)

            # Run validation periodically
            if epoch % 25 == 0:
                self._run_validation(epoch)

            # Final announcement
            if epoch == total_epochs:
                self.logger.info("=" * 80)
                self.logger.info("Training Complete")
                self.logger.info("=" * 80)

    def _train_one_epoch(self, epoch: int) -> Tuple[float, float, float, float]:
        """Train for one epoch and return (best_reward, loss, mean_reward, mean_final_cost)."""
        grad_acc_iterations = self.trainer_params["grad_acc_iterations"]
        iterations_per_instance = self.env_params["iterations_per_instance"]
        iterations_per_epoch = self.trainer_params["iterations_per_epoch"]

        # Validate gradient accumulation setup
        assert (
            iterations_per_instance % grad_acc_iterations == 0
        ), "iterations_per_instance must be divisible by grad_acc_iterations"
        assert (
            grad_acc_iterations <= iterations_per_instance
        ), "grad_acc_iterations must be <= iterations_per_instance"

        # Initialize metrics tracking
        metrics = {
            "score": AverageMeter(),
            "loss": AverageMeter(),
            "reward": AverageMeter(),
            "improved_frac": AverageMeter(),
        }
        final_costs = []
        processed_iters = 0
        logged_batches = 0
        epoch_start_time = time.time()

        # Main training loop
        self.model.zero_grad()
        while processed_iters < iterations_per_epoch:
            # Initialize new problem instances
            self.env.init_instances(self.batch_size, self.rollout_size, self.device)

            # Warm-up search iterations
            for _ in range(self.trainer_params["nb_skipped_iterations"]):
                self._search_one_batch(self.batch_size)

            # Training iterations with gradient accumulation
            for iteration in range(iterations_per_instance):
                batch_metrics = self._train_one_batch(self.batch_size)
                self._update_metrics(metrics, batch_metrics)

                # Optimizer step after accumulating gradients
                if (iteration + 1) % grad_acc_iterations == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.model.zero_grad()

            final_costs.extend(self.env.instanceSet.costs)
            processed_iters += self.batch_size * iterations_per_instance

            # Log first 10 batches of first epoch
            if epoch == self.start_epoch and logged_batches < 10:
                self._log_batch_progress(
                    epoch, processed_iters, iterations_per_epoch, metrics, final_costs
                )
                logged_batches += 1

        # Log epoch summary
        self._log_epoch_summary(
            epoch, processed_iters, iterations_per_epoch, metrics, final_costs
        )

        # Log to W&B if enabled
        if self.use_wandb:
            self._log_to_wandb(
                epoch, metrics, final_costs, time.time() - epoch_start_time
            )

        return (
            metrics["score"].avg,
            metrics["loss"].avg,
            metrics["reward"].avg,
            np.mean(final_costs),
        )

    def _train_one_batch(self, batch_size: int) -> Tuple[float, float, float, int]:
        """Train on one batch and return (score, loss, reward, nb_improved)."""
        self.model.train()

        # Reset environment and get state
        state = self.env.reset()
        reset_state = self.env.get_model_input(self.device)

        # Sample seed vectors (as described in PolyNet paper)
        z = self.seed_sampler.sample(batch_size, self.rollout_size)

        # Forward pass through model
        with torch.amp.autocast(device_type=self.device.type):
            self.model.pre_forward(reset_state, z)

        # Perform rollout
        step_probs = self._perform_rollout(state)
        selected_nodes = (
            self.env.selected_node_list.cpu().numpy()
        )  # Selected customer nodes for removal

        # Compute reward and loss
        reward = self._compute_reward(
            selected_nodes
        )  # Calculate reward by removing and reinserting customers
        loss = self._compute_policy_loss(reward, step_probs, batch_size)

        # Backward pass
        self.scaler.scale(loss).backward()

        # Compute metrics
        max_reward, _ = reward.max(dim=1)
        score_mean = max_reward.float().mean()
        nb_improved = (max_reward > 1e-5).sum().item()

        return score_mean.item(), loss.item(), reward.mean().item(), nb_improved

    def _perform_rollout(self, state) -> torch.Tensor:
        """Perform environment rollout and return step probabilities."""
        step_probs = []
        done = False

        while not done:
            with torch.amp.autocast(device_type=self.device.type):
                selected, prob, _ = self.model(state)
            state, done = self.env.step(selected)
            step_probs.append(prob)

        return torch.stack(step_probs, dim=2)  # (batch, rollout, steps)

    def _compute_reward(self, selected_nodes: np.ndarray) -> torch.Tensor:
        """Compute reward via destroy-and-repair heuristic."""
        recreate_n = self.env_params["recreate_n"]
        reward_type = self.trainer_params["reward_type"]
        beta = self.env_params["beta"]
        insert_in_new_tours_only = self.env_params["insert_in_new_tours_only"]

        reward = np.zeros((self.batch_size, self.rollout_size))
        old_costs = np.array(
            self.env.instanceSet.costs
        )  # Cost of the original solutions

        # Remove and reinsert selectedcustomers to create new solutions
        new_costs = self.env.instanceSet.remove_recreate(
            selected_nodes,
            recreate_n,
            "singleImp",
            beta=beta,
            insert_in_new_tours_only=insert_in_new_tours_only,
        )

        new_costs = np.array(new_costs)  # Cost of the new solutions

        for b_idx in range(self.batch_size):
            old_cost = old_costs[b_idx]
            costs = new_costs[b_idx]

            if reward_type == "b":
                # Binary reward with small continuous component
                r = (costs < old_cost - 0.0001).astype("float") + (
                    (old_cost - costs) * 0.0001
                )
            else:
                # Absolute improvement reward
                r = np.maximum(old_cost - costs, 0)

            reward[b_idx] = r

        return torch.tensor(reward, device=self.device)

    def _compute_policy_loss(
        self, reward: torch.Tensor, step_probs: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Compute policy gradient loss with advantage and top-1 filtering."""
        # Compute advantage (reward - mean reward)
        reward_reshaped = reward.reshape(batch_size, self.rollout_size, -1)
        advantage = reward_reshaped - reward_reshaped.mean(dim=1, keepdim=True)
        advantage = advantage.reshape(batch_size, -1)

        # Compute log probability of trajectory
        log_prob = step_probs.log().sum(dim=2)

        # Filter best rollout per seed vector
        rollout_costs = -reward.reshape(batch_size, self.rollout_size, -1)
        rank_idx = rollout_costs.argsort(1).argsort(1).reshape(batch_size, -1)
        top1_mask = (rank_idx < 1).float()

        # Policy gradient loss
        loss = -(advantage * log_prob * top1_mask).mean()
        return loss

    def _search_one_batch(self, batch_size: int) -> None:
        """Perform auxiliary search pass for warm-up (no gradients)."""
        recreate_n = self.env_params["recreate_n"]
        beta = self.env_params["beta"]
        insert_in_new_tours_only = self.env_params["insert_in_new_tours_only"]

        self.model.eval()

        with torch.no_grad():
            state = self.env.reset()
            reset_state = self.env.get_model_input(self.device)

            # Sample z vectors
            z = self.seed_sampler.sample(batch_size, self.rollout_size)

            with torch.amp.autocast(device_type=self.device.type):
                self.model.pre_forward(reset_state, z)

            # Rollout
            done = False
            while not done:
                with torch.amp.autocast(device_type=self.device.type):
                    selected, _, _ = self.model(state)
                state, done = self.env.step(selected)

            # Apply repair heuristic
            selected_nodes = self.env.selected_node_list.cpu().numpy()
            self.env.instanceSet.remove_recreate(
                selected_nodes,
                recreate_n,
                "allImp",
                beta=beta,
                insert_in_new_tours_only=insert_in_new_tours_only,
            )

    def _update_metrics(
        self,
        metrics: Dict[str, AverageMeter],
        batch_metrics: Tuple[float, float, float, int],
    ) -> None:
        """Update average meters with batch metrics."""
        score, loss, reward, nb_improved = batch_metrics
        metrics["score"].update(score, self.batch_size)
        metrics["loss"].update(loss, self.batch_size)
        metrics["reward"].update(reward, self.batch_size)
        metrics["improved_frac"].update(nb_improved / self.batch_size, self.batch_size)

    def _log_batch_progress(
        self,
        epoch: int,
        processed: int,
        total: int,
        metrics: Dict[str, AverageMeter],
        final_costs: list,
    ) -> None:
        """Log progress for a single batch."""
        self.logger.info(
            f"Epoch {epoch:3d}  |  Train {processed:4d}/{total:4d} ({100.0 * processed / total:5.1f}%)  |  "
            f'Reward: {metrics["score"].avg:6.4f}  |  Loss: {metrics["loss"].avg:6.4f}  |  '
            f'Improved: {metrics["improved_frac"].avg:5.3f}  |  Cost: {np.mean(final_costs):7.2f}'
        )

    def _log_epoch_summary(
        self,
        epoch: int,
        processed: int,
        total: int,
        metrics: Dict[str, AverageMeter],
        final_costs: list,
    ) -> None:
        """Log summary for entire epoch."""
        self.logger.info(
            f"Epoch {epoch:3d}  |  "
            f'Reward: {metrics["score"].avg:6.4f}  |  Loss: {metrics["loss"].avg:6.4f}  |  '
            f'Improved: {metrics["improved_frac"].avg:5.3f}  |  Cost: {np.mean(final_costs):7.2f}'
        )

    def _log_to_wandb(
        self,
        epoch: int,
        metrics: Dict[str, AverageMeter],
        final_costs: list,
        duration: float,
    ) -> None:
        """Log metrics to Weights & Biases."""
        wandb.log(
            step=epoch,
            data={
                "train/max_reward": metrics["score"].avg,
                "train/loss": metrics["loss"].avg,
                "train/mean_reward": metrics["reward"].avg,
                "train/improvement": metrics["improved_frac"].avg,
                "train/final_costs": np.mean(final_costs),
                "time/epoch": duration,
            },
        )

    def _log_timing(self, epoch: int, total_epochs: int) -> None:
        """Log elapsed and remaining time estimates."""
        elapsed, remaining = self.time_estimator.get_est_string(epoch, total_epochs)
        self.logger.info(
            f"Epoch {epoch:3d}/{total_epochs:3d}  |  Elapsed: {elapsed}  |  Remain: {remaining}"
        )

    def _save_checkpoints(self, epoch: int, total_epochs: int) -> None:
        """Save model checkpoints (periodic and latest)."""
        model_save_interval = self.trainer_params["model_save_interval"]
        all_done = epoch == total_epochs

        # Save periodic checkpoint
        if all_done or (epoch % model_save_interval) == 0:
            self.logger.info("Saving checkpoint")
            checkpoint = self._build_checkpoint(epoch)
            torch.save(
                checkpoint, os.path.join(self.results_dir, f"checkpoint-{epoch}.pt")
            )

        # Always save latest
        checkpoint = self._build_checkpoint(epoch)
        torch.save(checkpoint, os.path.join(self.results_dir, "latest_model.pt"))

    def _run_validation(self, epoch: int) -> None:
        """Run validation and log results."""
        aug_score = self.validator.run(self.model, self.model_frozen, epoch)

    def _build_checkpoint(self, epoch: int) -> Dict[str, Any]:
        """Build checkpoint dictionary."""
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_params": self.model_params,
            "env_params": self.env_params,
            "wandb_run_id": self.wandb_run_id,
        }
