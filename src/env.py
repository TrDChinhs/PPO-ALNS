"""
env.py — MDP Environment wrapping ALNS for PPO training

Chứa:
    ALNSEnv — Gym-style environment: reset(), step(), render()

Thuật toán như 1 MDP:
    State:  encoded ALNS state (StateEncoder)
    Action: (destroy_idx, repair_idx, accept, terminate)
    Reward: theo SPEC.md Section 6.3
    Done:   terminate action hoặc đạt max_iterations

Luồng step():
    1. Apply destroy operator (dựa trên action[0])
    2. Apply repair operator (dựa trên action[1])
    3. SA acceptance hoặc dùng action[2] để quyết định accept/reject
    4. Kiểm tra action[3] để terminate
    5. Tính reward
    6. Trả về (next_state, reward, done, info)

References:
    SPEC.md Section 6
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from problem import VRPTWInstance
from solution import Solution
from state_encoder import StateEncoder
from operators import (
    d1_random_destroy,
    d2_string_destroy,
    d3_route_destroy,
    d4_worst_destroy,
    d5_sequence_destroy,
    r1_greedy_repair,
    r2_critical_repair,
    r3_regret_repair,
    DESTROY_OPERATORS,
    REPAIR_OPERATORS,
)


# =============================================================================
# Constants (Reward parameters theo SPEC.md Section 6.3)
# =============================================================================

ALPHA = 1.0    # Reject but better
BETA = 0.5     # Accept but worse
GAMMA = 2.0    # New best found
F1 = 10.0      # Terminal reward — improvement
F2 = 5.0       # Terminal reward — early termination bonus

LAMBDA_TW = 100.0
LAMBDA_VEHICLES = 10.0


# =============================================================================
# Environment Result
# =============================================================================

@dataclass
class EnvResult:
    """Kết quả của 1 step."""
    state: np.ndarray
    reward: float
    done: bool
    info: dict


# =============================================================================
# ALNS Environment
# =============================================================================

class ALNSEnv:
    """
    Gym-style environment wrapping ALNS.

    Dùng cho PPO training: mỗi episode = 1 ALNS run.

    State space:  Variable (n² + 7n + 11 dims)
    Action space: Discrete(5) × Discrete(3) × Discrete(2) × Discrete(2)
                 Được encode thành 4 separate action heads trong agent.

    Usage:
        env = ALNSEnv(inst, max_iterations=100, seed=42)
        state = env.reset()
        for t in range(max_iterations):
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            buffer.push(state, action, reward, ...)
            if done:
                state = env.reset()
            else:
                state = next_state

        agent.update(buffer)

    Index convention:
        Depot = 0
        Customers = 1..N
    """

    def __init__(
        self,
        instance: VRPTWInstance,
        init_method: Literal["greedy", "cw"] = "greedy",
        max_iterations: int = 100,
        seed: int | None = None,
    ) -> None:
        self.instance = instance
        self.init_method = init_method
        self.max_iterations = max_iterations
        self.seed = seed

        self.rng = random.Random(seed)

        # State encoder
        self.encoder = StateEncoder(instance)

        # Destroy operators
        self._destroy_ops = [
            d1_random_destroy,
            d2_string_destroy,
            d3_route_destroy,
            d4_worst_destroy,
            d5_sequence_destroy,
        ]

        # Repair operators
        self._repair_ops = [
            r1_greedy_repair,
            r2_critical_repair,
            r3_regret_repair,
        ]

        # Destroy scale range
        n = instance.n_customers
        self._d_min = 1
        self._d_max = max(2, n // 10)

        # --- Episode state ---
        self._current: Solution | None = None  # current solution
        self._best: Solution | None = None  # best solution found
        self._iteration: int = 0
        self._init_cost: float = 0.0
        self._best_cost: float = 0.0
        self._prev_cost: float = 0.0
        self._terminate: bool = False

        # Operator usage tracking
        self._destroy_counts: list[int] = [0] * 5
        self._repair_counts: list[int] = [0] * 3

        # SA state
        self._sa_temp: float = 0.05

    # ---------- Core API ----------

    def reset(self) -> np.ndarray:
        """
        Reset environment — tạo initial solution.

        Returns:
            Initial state array
        """
        from solution import make_initial_solution

        self._iteration = 0
        self._terminate = False
        self._sa_temp = 0.05

        self._destroy_counts = [0] * 5
        self._repair_counts = [0] * 3

        # Initial solution
        init_sol = make_initial_solution(
            self.instance,
            method=self.init_method,
            seed=self.seed,
        )

        self._current = init_sol.copy()
        self._best = init_sol.copy()
        self._init_cost = init_sol.calc_cost()
        self._best_cost = self._init_cost
        self._prev_cost = self._init_cost

        return self._get_state(prev_cost=None)

    def step(
        self,
        action: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute 1 ALNS step với PPO action.

        Args:
            action: (destroy_idx, repair_idx, accept_action, terminate_action)
                    destroy_idx: 0-4
                    repair_idx: 0-2
                    accept_action: 0=reject, 1=accept
                    terminate_action: 0=continue, 1=stop

        Returns:
            (next_state, reward, done, info)
        """
        iter_ = self._iteration
        max_iter = self.max_iterations

        # --- 1. Termination check ---
        if iter_ >= max_iter:
            return self._terminal_step()

        if self._terminate:
            return self._terminal_step()

        # If terminate action == 1 → stop
        _, _, _, term_act = action
        if term_act == 1:
            self._terminate = True

        prev_cost = self._current.calc_cost()
        prev_best = self._best_cost

        # --- 2. Destroy ---
        d_idx = action[0]
        self._destroy_counts[d_idx] += 1
        d_scale = self.rng.randint(self._d_min, self._d_max)
        d_scale = min(d_scale, self.instance.n_customers)

        destroyed = self._current.copy()
        removed = self._destroy_ops[d_idx](destroyed, d_scale, self.rng)

        # --- 3. Repair ---
        r_idx = action[1]
        self._repair_counts[r_idx] += 1
        repaired = self._repair_ops[r_idx](destroyed, removed, self.rng)

        # --- 4. Acceptance ---
        accept_action = action[2]
        new_cost = repaired.calc_cost()
        delta = new_cost - prev_cost

        accepted = False
        if accept_action == 1:
            # Agent says accept
            accepted = True
        else:
            # Agent says reject → use SA fallback
            if delta < 0:
                accepted = True
            elif self._sa_temp > 1e-9:
                prob = np.exp(-delta / self._sa_temp)
                if self.rng.random() < prob:
                    accepted = True

        # --- 5. Update ---
        is_new_best = new_cost < prev_best
        if accepted:
            self._current = repaired
            if is_new_best:
                self._best = repaired.copy()
                self._best_cost = new_cost

        self._prev_cost = new_cost
        self._iteration += 1

        # SA cooling
        self._sa_temp *= 0.99

        # --- 6. Reward ---
        reward = self._compute_reward(
            prev_cost=prev_cost,
            new_cost=new_cost if accepted else prev_cost,
            accepted=accepted,
            is_new_best=is_new_best,
            terminate=self._terminate,
        )

        # --- 7. Done ---
        done = self._terminate or self._iteration >= max_iter

        # --- 8. Next state ---
        next_state = self._get_state(prev_cost=prev_cost)

        info = {
            "cost": new_cost if accepted else prev_cost,
            "best_cost": self._best_cost,
            "iteration": self._iteration,
            "accepted": accepted,
            "d_idx": d_idx,
            "r_idx": r_idx,
        }

        return next_state, reward, done, info

    def _terminal_step(self) -> tuple[np.ndarray, float, bool, dict]:
        """Return terminal reward when episode ends."""
        init_cost = self._init_cost
        best_cost = self._best_cost

        # Terminal reward: F1 * improvement_ratio + F2 * early_bonus
        improvement = (init_cost - best_cost) / max(init_cost, 1e-9)
        time_used = self._iteration / max(self.max_iterations, 1)
        terminal_reward = F1 * improvement + F2 * (1.0 - time_used)

        state = self._get_state(prev_cost=None)
        info = {
            "cost": best_cost,
            "best_cost": best_cost,
            "init_cost": init_cost,
            "iteration": self._iteration,
            "terminal": True,
        }

        return state, terminal_reward, True, info

    def _compute_reward(
        self,
        prev_cost: float,
        new_cost: float,
        accepted: bool,
        is_new_best: bool,
        terminate: bool,
    ) -> float:
        """
        Compute immediate reward — shaped rewards for effective learning.

        Key insight: SA fallback overrides agent's accept/reject when cost clearly
        improves or worsens. We reward based on the AGENT'S DECISION (not the
        final outcome), plus the cost change signal.

        Agent reward signal:
            Agent accept + cost improves  -> +1.0   (correct)
            Agent accept + cost worsens   -> -0.5   (wrong)
            Agent reject + cost improves   -> -1.0   (wrong, SA overrode)
            Agent reject + cost worsens   -> +0.5   (correct)

        Cost improvement signal:
            New best found               -> +2.0   (bonus)

        All scaled by relative magnitude so larger improvements = bigger rewards.
        """
        delta_abs = abs(prev_cost - new_cost) / max(prev_cost, 1e-9)

        if accepted and new_cost < prev_cost:
            reward = 1.0 * delta_abs
        elif accepted and new_cost >= prev_cost:
            reward = -0.5 * delta_abs
        elif not accepted and new_cost < prev_cost:
            reward = -1.0 * delta_abs
        else:
            reward = 0.5 * delta_abs

        if is_new_best:
            reward += 2.0

        return reward

    def _get_state(self, prev_cost: float | None) -> np.ndarray:
        """Build state array from current environment state."""
        return self.encoder.encode(
            solution=self._current,
            iteration=self._iteration,
            max_iterations=self.max_iterations,
            destroy_counts=self._destroy_counts,
            repair_counts=self._repair_counts,
            prev_cost=prev_cost,
        )

    # ---------- Accessors ----------

    @property
    def current_solution(self) -> Solution:
        """Current solution."""
        return self._current

    @property
    def best_solution(self) -> Solution:
        """Best solution found so far."""
        return self._best

    @property
    def iteration(self) -> int:
        """Current iteration count."""
        return self._iteration

    def get_result(self) -> dict:
        """Return final result dict."""
        bd = self._best.calc_cost_breakdown()
        return {
            "best_solution": self._best,
            "best_cost": self._best_cost,
            "best_cost_breakdown": bd,
            "init_cost": self._init_cost,
            "iteration": self._iteration,
            "max_iterations": self.max_iterations,
        }


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path

    inst_path = Path("data/n20/n20_M_1.json")
    if not inst_path.exists():
        print("Run generate_data.py first.")
    else:
        from problem import VRPTWInstance

        inst = VRPTWInstance.from_json(inst_path)
        print(f"Instance: {inst.name}  n={inst.n_customers}")

        env = ALNSEnv(inst, init_method="greedy", max_iterations=20, seed=42)
        state = env.reset()
        print(f"State dim: {state.shape}")

        # Random actions
        for t in range(5):
            action = (
                random.randint(0, 4),
                random.randint(0, 2),
                random.randint(0, 1),
                random.randint(0, 1),
            )
            next_state, reward, done, info = env.step(action)
            print(
                f"  t={t}: action={action} reward={reward:.4f} "
                f"cost={info['cost']:.4f} best={info['best_cost']:.4f} "
                f"done={done}"
            )
            if done:
                break

        result = env.get_result()
        print(f"\nFinal: cost={result['best_cost']:.4f}  iter={result['iteration']}")
