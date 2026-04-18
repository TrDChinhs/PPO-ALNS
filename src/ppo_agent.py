"""
ppo_agent.py — PPO Agent & Neural Network for PPO-ALNS

Chứa:
    1. PPOMLP          — PyTorch MLP network (shared encoder + 4 action heads)
    2. PPOAgent        — Agent wrapper với action selection, update
    3. ReplayBuffer    — Storage for rollout data (states, actions, rewards)

Architecture (theo SPEC.md Section 6.4):
    Shared MLP:  Input → 512 → 256 → 128 (ReLU)
    Policy heads:
        destroy:  128 → 5  (Categorical)
        repair:   128 → 3  (Categorical)
        accept:   128 → 2  (Categorical)
        terminate:128 → 2  (Categorical)
    Value head:  128 → 1

References:
    SPEC.md Section 6
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from problem import VRPTWInstance
from state_encoder import StateEncoder


# =============================================================================
# Constants
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DESTROY = 5
NUM_REPAIR = 3
NUM_ACCEPT = 2
NUM_TERMINATE = 2


# =============================================================================
# Rollout Storage
# =============================================================================

class RolloutStorage(NamedTuple):
    """Storage cho 1 episode rollout."""
    states: list[np.ndarray]  # s_t
    actions_destroy: list[int]  # a1_t
    actions_repair: list[int]  # a2_t
    actions_accept: list[int]  # a3_t
    actions_terminate: list[int]  # a4_t
    rewards: list[float]  # r_t
    values: list[float]  # V(s_t)
    log_probs_destroy: list[float]
    log_probs_repair: list[float]
    log_probs_accept: list[float]
    log_probs_terminate: list[float]
    dones: list[bool]

    @classmethod
    def empty(cls) -> "RolloutStorage":
        return cls(
            states=[],
            actions_destroy=[],
            actions_repair=[],
            actions_accept=[],
            actions_terminate=[],
            rewards=[],
            values=[],
            log_probs_destroy=[],
            log_probs_repair=[],
            log_probs_accept=[],
            log_probs_terminate=[],
            dones=[],
        )


# =============================================================================
# PPOMLP Network
# =============================================================================

class PPOMLP(nn.Module):
    """
    Shared MLP cho PPO-ALNS.

    Architecture:
        Input: state_dim (variable per n)
        Layer1: Linear(state_dim, 512) → ReLU
        Layer2: Linear(512, 256) → ReLU
        Layer3: Linear(256, 128) → ReLU

        Policy heads:
            destroy:  Linear(128, 5)
            repair:   Linear(128, 3)
            accept:   Linear(128, 2)
            terminate: Linear(128, 2)

        Value head:
            Linear(128, 1)
    """

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim

        # Shared encoder with LayerNorm for training stability
        self.fc1 = nn.Linear(state_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)

        # Policy heads
        self.head_destroy = nn.Linear(128, NUM_DESTROY)
        self.head_repair = nn.Linear(128, NUM_REPAIR)
        self.head_accept = nn.Linear(128, NUM_ACCEPT)
        self.head_terminate = nn.Linear(128, NUM_TERMINATE)

        # Value head
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: state tensor (batch, state_dim)

        Returns:
            (action_logits_dict, value)
            action_logits_dict: dict với 'destroy', 'repair', 'accept', 'terminate'
            value: V(s) scalar
        """
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))

        value = self.value_head(x).squeeze(-1)

        # Build logits dict with clamping for numerical stability
        logits_dict = {
            "destroy": torch.clamp(self.head_destroy(x), min=-10.0, max=10.0),
            "repair": torch.clamp(self.head_repair(x), min=-10.0, max=10.0),
            "accept": torch.clamp(self.head_accept(x), min=-10.0, max=10.0),
            "terminate": torch.clamp(self.head_terminate(x), min=-10.0, max=10.0),
        }

        return logits_dict, value

    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[tuple[int, int, int, int], tuple[float, float, float, float], float]:
        """
        Chọn action từ state.

        Args:
            state: numpy array (state_dim,)
            deterministic: nếu True, chọn argmax thay vì sample

        Returns:
            (a_destroy, a_repair, a_accept, a_terminate),
            (log_prob_destroy, log_prob_repair, log_prob_accept, log_prob_terminate),
            value(s)
        """
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            logits, value = self.forward(x)
            value = value.item()

            actions: tuple[int, ...] = ()
            log_probs: tuple[float, ...] = ()

            for key, num_actions in [
                ("destroy", NUM_DESTROY),
                ("repair", NUM_REPAIR),
                ("accept", NUM_ACCEPT),
                ("terminate", NUM_TERMINATE),
            ]:
                logit = logits[key].squeeze(0)
                if deterministic:
                    action = int(torch.argmax(logit).item())
                else:
                    dist = Categorical(logits=logit)
                    action = int(dist.sample().item())
                log_prob = float(Categorical(logits=logit).log_prob(torch.tensor(action, device=logits[key].device)).item())
                actions += (action,)
                log_probs += (log_prob,)

        return actions, log_probs, value

    def get_actions_batch(
        self,
        states: list[np.ndarray],
        deterministic: bool = False,
    ) -> tuple[
        list[tuple[int, int, int, int]],
        list[tuple[float, float, float, float]],
        list[float],
    ]:
        """Batch version of get_action for multiple states."""
        with torch.no_grad():
            x = torch.FloatTensor(np.array(states)).to(DEVICE)
            logits, values = self.forward(x)

            actions_list: list[tuple[int, int, int, int]] = []
            log_probs_list: list[tuple[float, float, float, float]] = []

            for i in range(len(states)):
                actions: tuple[int, ...] = ()
                log_probs: tuple[float, ...] = ()
                for key in ["destroy", "repair", "accept", "terminate"]:
                    logit = logits[key][i]
                    if deterministic:
                        action = int(torch.argmax(logit).item())
                    else:
                        dist = Categorical(logits=logit)
                        action = int(dist.sample().item())
                    lp = float(Categorical(logits=logit).log_prob(
                        torch.tensor(action, device=logit.device)
                    ).item())
                    actions += (action,)
                    log_probs += (lp,)
                actions_list.append(actions)
                log_probs_list.append(log_probs)

            values_list = [float(v.item()) for v in values.squeeze(-1)]

        return actions_list, log_probs_list, values_list

    def evaluate_actions(
        self,
        states: Tensor,
        actions_destroy: Tensor,
        actions_repair: Tensor,
        actions_accept: Tensor,
        actions_terminate: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate actions cho PPO update.

        Args:
            states: (batch, state_dim)
            actions: (batch,) each

        Returns:
            (log_probs_sum, entropy, value)
        """
        logits, values = self.forward(states)

        log_probs = []
        entropies = []

        for key, action_tensor, num_actions in [
            ("destroy", actions_destroy, NUM_DESTROY),
            ("repair", actions_repair, NUM_REPAIR),
            ("accept", actions_accept, NUM_ACCEPT),
            ("terminate", actions_terminate, NUM_TERMINATE),
        ]:
            dist = Categorical(logits=logits[key])
            log_probs.append(dist.log_prob(action_tensor))
            entropies.append(dist.entropy())

        log_prob_sum = sum(log_probs)
        entropy = torch.stack(entropies).sum(dim=0).mean()

        return log_prob_sum, entropy, values


# =============================================================================
# PPO Agent
# =============================================================================

@dataclass
class PPOConfig:
    """Hyperparameters cho PPO."""
    lr: float = 1e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 10
    batch_size: int = 128
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42


class PPOAgent:
    """
    PPO Agent cho ALNS operator selection.

    Chứa:
        - Neural network (PPOMLP)
        - Optimizer (Adam)
        - Config (hyperparameters)
        - Action selection methods

    Usage:
        agent = PPOAgent(state_dim=491, config=PPOConfig())
        action, log_prob, value = agent.get_action(state)
        agent.update(rollouts, ...)

    State dim = n² + 7n + 11:
        n=20:  491
        n=50:  2786
        n=100: 10711
    """

    def __init__(
        self,
        state_dim: int,
        config: PPOConfig | None = None,
        device: torch.device = DEVICE,
    ) -> None:
        self.config = config or PPOConfig()
        self.device = device
        self.state_dim = state_dim

        # Set seeds
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)

        # Network
        self.network = PPOMLP(state_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.lr,
            eps=1e-5,
        )

        # Training stats
        self.total_updates = 0
        self.value_losses: list[float] = []
        self.policy_losses: list[float] = []
        self.entropies: list[float] = []

    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[tuple[int, int, int, int], tuple[float, float, float, float], float]:
        """
        Sample/deterministic action from policy.

        Returns:
            ((destroy_idx, repair_idx, accept, terminate),
             (log_probs...),
             value)
        """
        return self.network.get_action(state, deterministic=deterministic)

    def get_actions_batch(
        self,
        states: list[np.ndarray],
        deterministic: bool = False,
    ) -> tuple[
        list[tuple[int, int, int, int]],
        list[tuple[float, float, float, float]],
        list[float],
    ]:
        """
        Batch version of get_action for multiple states.

        Returns:
            (actions_list, log_probs_list, values_list)
            Each list has length = len(states)
        """
        return self.network.get_actions_batch(states, deterministic=deterministic)

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "updates": self.total_updates,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.config = checkpoint["config"]
        self.total_updates = checkpoint["updates"]

    def update(
        self,
        rollouts: list[RolloutStorage],
    ) -> dict[str, float]:
        """
        PPO update từ collected rollouts.

        Args:
            rollouts: list of RolloutStorage, mỗi storage = 1 episode

        Returns:
            Dict of training metrics
        """
        cfg = self.config

        # Flatten rollouts into tensors
        states_list = []
        actions_destroy_list = []
        actions_repair_list = []
        actions_accept_list = []
        actions_terminate_list = []
        rewards_list = []
        values_list = []
        old_log_probs_destroy_list = []
        old_log_probs_repair_list = []
        old_log_probs_accept_list = []
        old_log_probs_terminate_list = []
        dones_list = []

        for rollout in rollouts:
            states_list.extend(rollout.states)
            actions_destroy_list.extend(rollout.actions_destroy)
            actions_repair_list.extend(rollout.actions_repair)
            actions_accept_list.extend(rollout.actions_accept)
            actions_terminate_list.extend(rollout.actions_terminate)
            rewards_list.extend(rollout.rewards)
            values_list.extend(rollout.values)
            old_log_probs_destroy_list.extend(rollout.log_probs_destroy)
            old_log_probs_repair_list.extend(rollout.log_probs_repair)
            old_log_probs_accept_list.extend(rollout.log_probs_accept)
            old_log_probs_terminate_list.extend(rollout.log_probs_terminate)
            dones_list.extend(rollout.dones)

        n = len(states_list)
        if n == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Stack tensors
        states = torch.FloatTensor(np.array(states_list)).to(self.device)
        actions_destroy = torch.LongTensor(actions_destroy_list).to(self.device)
        actions_repair = torch.LongTensor(actions_repair_list).to(self.device)
        actions_accept = torch.LongTensor(actions_accept_list).to(self.device)
        actions_terminate = torch.LongTensor(actions_terminate_list).to(self.device)
        rewards = torch.FloatTensor(rewards_list).to(self.device)
        old_values = torch.FloatTensor(values_list).to(self.device)
        old_log_probs = (
            torch.FloatTensor(old_log_probs_destroy_list).to(self.device)
            + torch.FloatTensor(old_log_probs_repair_list).to(self.device)
            + torch.FloatTensor(old_log_probs_accept_list).to(self.device)
            + torch.FloatTensor(old_log_probs_terminate_list).to(self.device)
        )
        dones = torch.FloatTensor(dones_list).to(self.device)

        # --- Compute GAE ---
        advantages, returns = self._compute_gae(rewards, old_values, dones)

        # Guard: replace NaN/Inf with 0
        advantages = torch.where(torch.isfinite(advantages), advantages, torch.zeros_like(advantages))
        returns = torch.where(torch.isfinite(returns), returns, torch.zeros_like(returns))

        # Guard states
        states = torch.where(torch.isfinite(states), states, torch.zeros_like(states))

        # Guard rewards
        rewards = torch.where(torch.isfinite(rewards), rewards, torch.zeros_like(rewards))

        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = torch.zeros_like(advantages)

        # --- PPO Update ---
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        indices = torch.randperm(n).to(self.device)

        for _ in range(cfg.ppo_epochs):
            # Shuffle
            perm_indices = indices[torch.randperm(n)]

            for start in range(0, n, cfg.batch_size):
                end = min(start + cfg.batch_size, n)
                batch_idx = perm_indices[start:end]

                batch_states = states[batch_idx]
                batch_actions_destroy = actions_destroy[batch_idx]
                batch_actions_repair = actions_repair[batch_idx]
                batch_actions_accept = actions_accept[batch_idx]
                batch_actions_terminate = actions_terminate[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]

                # Evaluate actions
                log_probs, entropy, values = self.network.evaluate_actions(
                    batch_states,
                    batch_actions_destroy,
                    batch_actions_repair,
                    batch_actions_accept,
                    batch_actions_terminate,
                )

                # PPO clipped objective
                ratio = torch.exp(torch.clamp(log_probs - batch_old_log_probs, min=-20.0, max=20.0))
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (guard NaN)
                raw_value_loss = F.mse_loss(values, batch_returns)
                value_loss = torch.where(torch.isfinite(torch.tensor(raw_value_loss)), raw_value_loss, torch.tensor(0.0, device=self.device))

                # Total loss
                loss = (
                    policy_loss
                    - cfg.entropy_coef * entropy
                    + cfg.value_loss_coef * value_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        self.total_updates += 1

        metrics = {
            "policy_loss": total_policy_loss / (cfg.ppo_epochs * math.ceil(n / cfg.batch_size)),
            "value_loss": total_value_loss / (cfg.ppo_epochs * math.ceil(n / cfg.batch_size)),
            "entropy": total_entropy / (cfg.ppo_epochs * math.ceil(n / cfg.batch_size)),
            "num_samples": n,
        }
        self.policy_losses.append(metrics["policy_loss"])
        self.value_losses.append(metrics["value_loss"])
        self.entropies.append(metrics["entropy"])

        return metrics

    def _compute_gae(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: (n,) rewards
            values: (n,) value estimates V(s_t)
            dones: (n,) episode done flags

        Returns:
            (advantages, returns) each (n,)
        """
        cfg = self.config
        n = len(rewards)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0.0
        next_value = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]

            delta = rewards[t] + cfg.gamma * next_val * next_non_terminal - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Test với n=20
    state_dim = 20 * 20 + 7 * 20 + 11
    print(f"State dim (n=20): {state_dim}")

    agent = PPOAgent(state_dim=state_dim)
    print(f"Network parameters: {sum(p.numel() for p in agent.network.parameters()):,}")

    # Test action selection
    dummy_state = np.random.randn(state_dim).astype(np.float32)
    action, log_prob, value = agent.get_action(dummy_state)
    print(f"Action: {action}")
    print(f"Log prob: {log_prob}")
    print(f"Value: {value:.4f}")

    # Test update
    rollouts = [
        RolloutStorage(
            states=[dummy_state],
            actions_destroy=[action[0]],
            actions_repair=[action[1]],
            actions_accept=[action[2]],
            actions_terminate=[action[3]],
            rewards=[0.1],
            values=[value],
            log_probs_destroy=[log_prob[0]],
            log_probs_repair=[log_prob[1]],
            log_probs_accept=[log_prob[2]],
            log_probs_terminate=[log_prob[3]],
            dones=[False],
        )
    ]
    metrics = agent.update(rollouts)
    print(f"Metrics after 1 update: {metrics}")
