"""
trainer.py — PPO Training Loop với parallel ALNS environments

Chứa:
    PPOTrainer — Training loop chính
    - Parallel environments (multiple instances)
    - Rollout collection
    - PPO update
    - Evaluation & checkpointing

Training loop (theo SPEC.md Section 6.6):
    1. Initialize parallel envs (mỗi env = 1 VRPTW instance)
    2. For each step:
         - Agent chọn actions (A1-A4)
         - Apply vào ALNS, nhận reward + next state
         - Store (s, a, r, s', done)
    3. Khi buffer đầy → PPO update
    4. Evaluate on test instances
    5. Save checkpoints

References:
    SPEC.md Section 6.5, 6.6
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from problem import VRPTWInstance
from ppo_agent import PPOAgent, PPOConfig, RolloutStorage, DEVICE
from state_encoder import StateEncoder


# =============================================================================
# Trainer Config
# =============================================================================

@dataclass
class TrainerConfig:
    """Config cho trainer."""
    # Training
    total_steps: int = 200_000
    num_envs: int = 16            # parallel envs
    eval_interval: int = 5000     # evaluate every N steps
    save_interval: int = 10_000   # checkpoint every N steps
    eval_episodes: int = 5         # episodes per evaluation

    # PPO
    ppo_config: PPOConfig = field(default_factory=PPOConfig)

    # Instances
    train_instances: list[str] = field(default_factory=list)
    eval_instances: list[str] = field(default_factory=list)
    init_method: str = "greedy"   # "greedy" | "cw"
    max_iterations: int = 100

    # Output
    output_dir: str = "checkpoints"
    log_interval: int = 100       # print every N steps


@dataclass
class TrainingStats:
    """Stats collected during training."""
    step: int
    episodes_collected: int
    avg_episode_reward: float
    avg_episode_cost: float
    policy_loss: float
    value_loss: float
    entropy: float
    time_elapsed: float


# =============================================================================
# ALNSEnv wrapper (re-exported)
# =============================================================================

ALNSEnv: type = None  # lazy import


def _get_env_class():
    global ALNSEnv
    if ALNSEnv is None:
        from env import ALNSEnv
    return ALNSEnv


# =============================================================================
# PPOTrainer
# =============================================================================

class PPOTrainer:
    """
    PPO-ALNS Trainer.

    Usage:
        config = TrainerConfig(
            train_instances=["data/n20/n20_M_1.json", ...],
            eval_instances=["data/n20/n20_M_2.json"],
            total_steps=200_000,
        )
        trainer = PPOTrainer(config)
        trainer.train()
        trainer.save("checkpoints/ppo_alns_final.pt")
    """

    def __init__(
        self,
        config: TrainerConfig,
        device: torch.device = DEVICE,
    ) -> None:
        self.config = config
        self.device = device

        # Output dir
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build state encoder từ first instance
        first_inst = VRPTWInstance.from_json(config.train_instances[0])
        encoder = StateEncoder(first_inst)
        self.state_dim = encoder.state_dim

        # Create agent
        self.agent = PPOAgent(
            state_dim=self.state_dim,
            config=config.ppo_config,
            device=device,
        )

        # Create env pool (round-robin assignment)
        self._env_pool: list[VRPTWInstance] = []
        self._load_instances(config.train_instances)

        # Parallel envs (recreated per eval)
        self._parallel_envs: list = []

        # Training stats
        self._step: int = 0
        self._episode_count: int = 0
        self._start_time: float = 0.0

        # Logging
        self._log_history: list[TrainingStats] = []

    def _load_instances(self, paths: list[str]) -> None:
        """Load all training instances into memory."""
        for p in paths:
            inst = VRPTWInstance.from_json(p)
            self._env_pool.append(inst)
        print(f"  Loaded {len(self._env_pool)} training instances")

    def _make_parallel_envs(self, num_envs: int) -> list:
        """Create num_envs parallel ALNSEnv instances."""
        from env import ALNSEnv

        envs = []
        for i in range(num_envs):
            # Round-robin assign instances from pool
            inst = self._env_pool[i % len(self._env_pool)]
            # Different seed per env
            seed = self.config.ppo_config.seed + i
            env = ALNSEnv(
                instance=inst,
                init_method=self.config.init_method,
                max_iterations=self.config.max_iterations,
                seed=seed,
            )
            envs.append(env)
        return envs

    # ---------- Training ----------

    def train(self) -> None:
        """Main training loop."""
        cfg = self.config
        print(f"\n{'='*60}")
        print(f"PPO-ALNS Training")
        print(f"  Total steps: {cfg.total_steps:,}")
        print(f"  Parallel envs: {cfg.num_envs}")
        print(f"  State dim: {self.state_dim}")
        print(f"  Device: {self.device}")
        print(f"  Init method: {cfg.init_method}")
        print(f"{'='*60}")

        self._start_time = time.time()

        # Initialize parallel envs
        self._parallel_envs = self._make_parallel_envs(cfg.num_envs)

        # Reset all envs → initial states
        states = [env.reset() for env in self._parallel_envs]

        while self._step < cfg.total_steps:
            # Collect rollouts from ALL parallel envs in one batch
            rollouts = self._collect_rollouts(states)
            n_collected = sum(len(r.rewards) for r in rollouts)

            # PPO update
            metrics = self.agent.update(rollouts)
            self._step += n_collected

            # Log
            if self._step % cfg.log_interval < n_collected:
                self._log_step(metrics)

            # Evaluate
            if self._step % cfg.eval_interval == 0 and self._step > 0:
                eval_reward = self._evaluate()
                print(f"  [Eval @ step {self._step:,}] eval_reward={eval_reward:.4f}")

            # Save checkpoint
            if self._step % cfg.save_interval == 0 and self._step > 0:
                self._save_checkpoint(f"step_{self._step}.pt")

        # Final save
        self._save_checkpoint("final.pt")
        elapsed = time.time() - self._start_time
        print(f"\nTraining DONE in {elapsed:.1f}s ({self._step:,} steps)")

    def _collect_rollouts(
        self,
        current_states: list[np.ndarray],
    ) -> list[RolloutStorage]:
        """
        Collect 1 step worth of rollouts from ALL parallel envs (batch).

        Uses batch inference on GPU for maximum efficiency.
        """
        envs = self._parallel_envs
        num_envs = len(envs)

        # Batch inference
        action_tuples, log_probs_list, values_list = self.agent.get_actions_batch(
            current_states, deterministic=False
        )

        # Step all envs and collect
        rollout = RolloutStorage.empty()

        for i in range(num_envs):
            action = action_tuples[i]
            log_probs = log_probs_list[i]
            value = values_list[i]

            d_a, r_a, acc_a, term_a = action

            # Step env
            next_state, reward, done, info = envs[i].step(action)

            rollout.states.append(current_states[i])
            rollout.actions_destroy.append(d_a)
            rollout.actions_repair.append(r_a)
            rollout.actions_accept.append(acc_a)
            rollout.actions_terminate.append(term_a)
            rollout.rewards.append(reward)
            rollout.values.append(value)
            rollout.log_probs_destroy.append(log_probs[0])
            rollout.log_probs_repair.append(log_probs[1])
            rollout.log_probs_accept.append(log_probs[2])
            rollout.log_probs_terminate.append(log_probs[3])
            rollout.dones.append(done)

            # Update state
            current_states[i] = next_state

            # If done, reset env
            if done:
                self._episode_count += 1
                current_states[i] = envs[i].reset()

        return [rollout]

    def _evaluate(self) -> float:
        """Evaluate agent on eval instances (greedy action selection)."""
        from env import ALNSEnv

        if not self.config.eval_instances:
            return 0.0

        eval_rewards: list[float] = []

        for inst_path in self.config.eval_instances[:self.config.eval_episodes]:
            inst = VRPTWInstance.from_json(inst_path)
            env = ALNSEnv(
                instance=inst,
                init_method=self.config.init_method,
                max_iterations=self.config.max_iterations,
                seed=123,  # fixed seed for eval
            )
            state = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action_tuple, _, _ = self.agent.get_action(state, deterministic=True)
                action = action_tuple  # tuple[int, int, int, int]
                state, reward, done, _ = env.step(action)
                total_reward += reward

            eval_rewards.append(total_reward)

        return sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0.0

    def _log_step(self, metrics: dict[str, float]) -> None:
        """Print training progress."""
        step = self._step
        elapsed = time.time() - self._start_time
        print(
            f"  step={step:>7,}  "
            f"policy_loss={metrics['policy_loss']:>8.4f}  "
            f"value_loss={metrics['value_loss']:>8.4f}  "
            f"entropy={metrics['entropy']:>7.4f}  "
            f"time={elapsed:>7.1f}s"
        )

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.output_dir / filename
        self.agent.save(str(path))
        print(f"  [Checkpoint saved] {path}")

    # ---------- Inference ----------

    def solve(self, instance_path: str, deterministic: bool = True) -> dict:
        """
        Run trained agent on a single instance.

        Args:
            instance_path: path to JSON instance
            deterministic: use argmax instead of sampling

        Returns:
            Result dict với best_solution, best_cost, ...
        """
        from env import ALNSEnv

        inst = VRPTWInstance.from_json(instance_path)
        env = ALNSEnv(
            instance=inst,
            init_method=self.config.init_method,
            max_iterations=self.config.max_iterations,
            seed=999,
        )
        state = env.reset()
        done = False

        while not done:
            action_tuple, _, _ = self.agent.get_action(state, deterministic=deterministic)
            state, _, done, _ = env.step(action_tuple)

        return env.get_result()


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path("data/n20")
    if not data_dir.exists():
        print("Run generate_data.py first.")
    else:
        import glob

        instances = sorted(data_dir.glob("*.json"))[:4]
        print(f"Training instances: {len(instances)}")

        config = TrainerConfig(
            total_steps=1000,
            num_envs=2,
            eval_interval=500,
            save_interval=500,
            ppo_config=PPOConfig(
                ppo_epochs=2,
                batch_size=32,
                seed=42,
            ),
            train_instances=[str(p) for p in instances],
            eval_instances=[str(p) for p in instances[:2]],
            output_dir="checkpoints/test",
        )

        trainer = PPOTrainer(config)
        trainer.train()

        # Quick solve test
        print("\n--- Solve test ---")
        result = trainer.solve(str(instances[0]), deterministic=True)
        print(f"Best cost: {result['best_cost']:.4f}")
        print(f"Iterations: {result['iteration']}")