"""
alns.py — Adaptive Large Neighborhood Search (ALNS) framework

ĐỌC SAU operators.py.

Chứa:
    ALNSConfig     — cấu hình thuật toán
    ALNSResult    — kết quả chạy ALNS
    ALNS          — class chính

Luồng ALNS:
    1. Tạo initial solution (Greedy/CW/random)
    2. Lặp max_iterations lần:
         a. Chọn Destroy operator (Roulette wheel)
         b. Apply Destroy → removed customers
         c. Chọn Repair operator (Roulette wheel)
         d. Apply Repair → new solution
         e. SA Acceptance check
         f. Update operator weights
    3. Trả về best solution

References:
    SPEC.md Section 5
"""

from __future__ import annotations

import copy
import dataclasses
import math
import random
import time
from dataclasses import dataclass
from typing import Literal

from problem import VRPTWInstance
from solution import Solution, make_initial_solution
from operators import (
    DESTROY_OPERATORS,
    REPAIR_OPERATORS,
    d1_random_destroy,
    d2_string_destroy,
    d3_route_destroy,
    d4_worst_destroy,
    d5_sequence_destroy,
    r1_greedy_repair,
    r2_critical_repair,
    r3_regret_repair,
)


# =============================================================================
# Section 1: Configuration
# =============================================================================

@dataclass
class ALNSConfig:
    """
    Cấu hình ALNS.

    Attributes:
        init_method: cách tạo initial solution
                   "greedy" | "cw" | "random_feasible"
        max_iterations: số lần lặp destroy-repair
        destroy_scale_min: số lượng customer tối thiểu bị xóa
        destroy_scale_max: số lượng customer tối đa bị xóa
        sa_temperature_0: temperature ban đầu cho SA
        sa_cooling_rate: cooling rate cho SA (0 < rate < 1)
                         temperature_t = t0 * (rate ^ t)
        operator_weights_rw: reaction factor cho weight update (0.9 như paper)
        operator_scores: roulette wheel scores cho 4 quality levels
        seed: random seed
        verbose: in ra trạng thái trong quá trình chạy
    """
    init_method: Literal["greedy", "cw", "random_feasible"] = "greedy"
    max_iterations: int = 100
    destroy_scale_min: int = 1
    destroy_scale_max: int | None = None  # None = auto (n//10)
    sa_temperature_0: float = 0.05
    sa_cooling_rate: float = 0.99
    operator_weights_rw: float = 0.9  # Reaction factor (paper: 0.9)
    # Roulette wheel scores cho 4 quality levels:
    #   global best, local best, accepted worse, rejected
    operator_scores: tuple[int, int, int, int] = (25, 5, 1, 0)
    seed: int = 42
    verbose: bool = True


# =============================================================================
# Section 2: Operator Weights (Roulette Wheel)
# =============================================================================

@dataclass
class OperatorScoreTracker:
    """
    Track scores và weights cho mỗi operator.
    Dùng cho Roulette wheel selection.

    Quality levels:
        0 = global best found
        1 = local best (tốt hơn current)
        2 = accepted worse solution
        3 = rejected solution
    """
    num_destroy: int
    num_repair: int
    rw: float = 0.9

    def __post_init__(self) -> None:
        # Scores (rewards) cho mỗi quality level
        self._scores = [25, 5, 1, 0]

        # Weights hiện tại của mỗi operator
        self.destroy_weights: list[float] = [0.0] * self.num_destroy
        self.repair_weights: list[float] = [0.0] * self.num_repair

        # Số lần mỗi operator được chọn
        self.destroy_counts: list[int] = [0] * self.num_destroy
        self.repair_counts: list[int] = [0] * self.num_repair

    def update_destroy(self, op_idx: int, quality: int) -> None:
        """Update weight của destroy operator theo quality."""
        reward = self._scores[quality]
        w = self.destroy_weights[op_idx]
        n = self.destroy_counts[op_idx]

        if w > 0 and n > 0:
            new_w = (1 - self.rw) * w + self.rw * reward / n
        else:
            new_w = reward

        self.destroy_weights[op_idx] = new_w
        self.destroy_counts[op_idx] += 1

    def update_repair(self, op_idx: int, quality: int) -> None:
        """Update weight của repair operator theo quality."""
        reward = self._scores[quality]
        w = self.repair_weights[op_idx]
        n = self.repair_counts[op_idx]

        if w > 0 and n > 0:
            new_w = (1 - self.rw) * w + self.rw * reward / n
        else:
            new_w = reward

        self.repair_weights[op_idx] = new_w
        self.repair_counts[op_idx] += 1

    def select_destroy(self, rng: random.Random) -> int:
        """Chọn destroy operator theo Roulette wheel."""
        return _roulette_select(self.destroy_weights, rng)

    def select_repair(self, rng: random.Random) -> int:
        """Chọn repair operator theo Roulette wheel."""
        return _roulette_select(self.repair_weights, rng)


def _roulette_select(weights: list[float], rng: random.Random) -> int:
    """
    Roulette wheel selection theo weights.

    Args:
        weights: list of weights (float >= 0)
        rng: random generator

    Returns:
        Index của operator được chọn
    """
    total = sum(weights)
    if total == 0:
        # Tất cả weights = 0 → chọn ngẫu nhiên
        return rng.randint(0, len(weights) - 1)

    r = rng.uniform(0, total)
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return i

    return len(weights) - 1  # fallback


# =============================================================================
# Section 3: ALNS Core
# =============================================================================

@dataclass
class ALNSResult:
    """Kết quả của 1 lần chạy ALNS."""
    best_solution: Solution
    best_cost: float
    best_cost_breakdown: tuple[float, float, int]
        # (distance, tw_violations, num_vehicles)
    iterations: int
    best_found_at_iter: int
    runtime_seconds: float
    operator_selection_history: list[tuple[int, int]] | None = None
        # [(destroy_idx, repair_idx), ...] theo thứ tự iterations
    cost_history: list[float] | None = None


class ALNS:
    """
    ALNS Framework cho VRPTW.

    Thuật toán chính:
        1. Tạo initial solution S
        2. current = best = S
        3. Với iter = 1..max_iterations:
             a. Chọn destroy operator d_idx (Roulette)
             b. Chọn repair operator r_idx (Roulette)
             c. new_sol = apply(destroy, repair) on current
             d. if sa_accept(new_sol, current):
                    current = new_sol
                    quality = evaluate(new_sol vs best)
                    update_weights(d_idx, r_idx, quality)
             e. if new_sol.cost < best.cost:
                    best = new_sol
                    best_found_at = iter
        4. Return best

    Attributes:
        instance: VRPTWInstance
        config: ALNSConfig

    Usage:
        inst = VRPTWInstance.from_json("data/n20/n20_M_1.json")
        config = ALNSConfig(init_method="greedy", max_iterations=100, seed=42)
        alns = ALNS(inst, config)
        result = alns.run()
        print(result.best_cost)
    """

    def __init__(
        self,
        instance: VRPTWInstance,
        config: ALNSConfig | None = None,
    ) -> None:
        self.instance = instance
        self.config = config or ALNSConfig()
        self.rng = random.Random(self.config.seed)

        # Operator functions
        self._destroy_ops = [
            d1_random_destroy,
            d2_string_destroy,
            d3_route_destroy,
            d4_worst_destroy,
            d5_sequence_destroy,
        ]
        self._repair_ops = [
            r1_greedy_repair,
            r2_critical_repair,
            r3_regret_repair,
        ]

        # Score tracker cho Roulette wheel
        self._tracker = OperatorScoreTracker(
            num_destroy=len(self._destroy_ops),
            num_repair=len(self._repair_ops),
            rw=self.config.operator_weights_rw,
        )

        # Destroy scale range
        n = instance.n_customers
        if self.config.destroy_scale_max is None:
            self._d_min = self.config.destroy_scale_min
            self._d_max = max(2, n // 10)
        else:
            self._d_min = self.config.destroy_scale_min
            self._d_max = self.config.destroy_scale_max

    def run(self) -> ALNSResult:
        """
        Chạy ALNS.

        Returns:
            ALNSResult chứa best solution và statistics
        """
        start_time = time.time()
        cfg = self.config
        rng = self.rng

        # --- 1. Initial solution ---
        if cfg.init_method == "random_feasible":
            current = _make_random_feasible(self.instance, rng)
        else:
            current = make_initial_solution(
                self.instance,
                method=cfg.init_method,
                seed=cfg.seed,
            )

        best = current.copy()
        best_cost = current.calc_cost()
        best_cost_breakdown = self._get_cost_breakdown_tuple(best)
        best_found_at_iter = 0

        # SA temperature
        temperature = cfg.sa_temperature_0

        # History
        op_history: list[tuple[int, int]] = []
        cost_history: list[float] = [best_cost]

        if cfg.verbose:
            print(f"  [ALNS] init cost: {best_cost:.4f}  (method={cfg.init_method})")

        # --- 2. Main loop ---
        for iteration in range(1, cfg.max_iterations + 1):
            prev_cost = current.calc_cost()

            # --- 2a. Select Destroy operator ---
            d_idx = self._tracker.select_destroy(rng)
            d_name = DESTROY_OPERATORS[d_idx][0]

            # --- 2b. Select Repair operator ---
            r_idx = self._tracker.select_repair(rng)
            r_name = REPAIR_OPERATORS[r_idx][0]

            # --- 2c. Apply Destroy → removed customers ---
            # Copy solution trước khi destroy
            destroyed = current.copy()

            # Destroy scale: sample ngẫu nhiên trong range
            d_scale = rng.randint(self._d_min, self._d_max)
            d_scale = min(d_scale, self.instance.n_customers)

            removed = self._destroy_ops[d_idx](destroyed, d_scale, rng)

            # --- 2d. Apply Repair ---
            repaired = self._repair_ops[r_idx](destroyed, removed, rng)

            # --- 2e. SA Acceptance ---
            new_cost = repaired.calc_cost()
            delta = new_cost - prev_cost

            accepted = False
            if delta < 0:
                # Tốt hơn → luôn accept
                accepted = True
            else:
                # Xấu hơn → SA probability
                prob = math.exp(-delta / temperature) if temperature > 1e-9 else 0.0
                if rng.random() < prob:
                    accepted = True

            # --- 2f. Update ---
            new_best_cost = best.calc_cost()

            if accepted:
                current = repaired

                # Quality evaluation cho weight update
                if new_cost < new_best_cost:
                    # Global best
                    d_quality = 0
                    r_quality = 0
                elif new_cost < prev_cost:
                    # Local best
                    d_quality = 1
                    r_quality = 1
                else:
                    # Accepted worse
                    d_quality = 2
                    r_quality = 2

                # Update operator weights
                self._tracker.update_destroy(d_idx, d_quality)
                self._tracker.update_repair(r_idx, r_quality)

                # Update best if needed
                if new_cost < new_best_cost:
                    best = current.copy()
                    best_cost = new_cost
                    best_cost_breakdown = self._get_cost_breakdown_tuple(best)
                    best_found_at_iter = iteration

                    if cfg.verbose:
                        bd = best_cost_breakdown
                        print(
                            f"  [ALNS] iter {iteration:3d}: NEW BEST "
                            f"cost={best_cost:.4f} "
                            f"(d={bd[0]:.2f}, tw={bd[1]:.2f}, k={bd[2]}) "
                            f"d={d_name}, r={r_name}"
                        )
            else:
                # Rejected
                self._tracker.update_destroy(d_idx, 3)
                self._tracker.update_repair(r_idx, 3)

            # SA cooling
            temperature *= cfg.sa_cooling_rate

            # History
            op_history.append((d_idx, r_idx))
            cost_history.append(new_cost if accepted else prev_cost)

        runtime = time.time() - start_time

        if cfg.verbose:
            bd = best_cost_breakdown
            print(
                f"  [ALNS] DONE: cost={best_cost:.4f} "
                f"(d={bd[0]:.2f}, tw={bd[1]:.2f}, k={bd[2]}) "
                f"iter={best_found_at_iter}/{cfg.max_iterations} "
                f"time={runtime:.2f}s"
            )

        return ALNSResult(
            best_solution=best,
            best_cost=best_cost,
            best_cost_breakdown=best_cost_breakdown,
            iterations=cfg.max_iterations,
            best_found_at_iter=best_found_at_iter,
            runtime_seconds=round(runtime, 2),
            operator_selection_history=op_history,
            cost_history=cost_history,
        )

    def _get_cost_breakdown_tuple(self, sol: Solution) -> tuple[float, float, int]:
        """Trả về (distance, tw_violations, num_vehicles)."""
        bd = sol.calc_cost_breakdown()
        return (bd.total_distance, bd.tw_violations, bd.num_vehicles)


# =============================================================================
# Section 4: Random Feasible Initial Solution
# =============================================================================

def _make_random_feasible(inst: VRPTWInstance, rng: random.Random) -> Solution:
    """
    Tạo 1 feasible solution đơn giản:
    Mỗi customer = 1 route riêng [0, i, 0].

    Đảm bảo feasible vì:
        - Mỗi route chỉ có 1 customer → demand = customer.demand ≤ 64
        - Không gộp customers → không có TW conflict

    Args:
        inst: VRPTWInstance
        rng: random generator

    Returns:
        Solution feasible
    """
    routes: list[list[int]] = [[0, i, 0] for i in range(1, inst.n_customers + 1)]
    return Solution(routes=routes, instance=inst)


# =============================================================================
# Section 5: Quick Test
# =============================================================================

if __name__ == "__main__":
    # Quick test trên 1 instance
    from pathlib import Path

    inst_path = Path("data/n20/n20_M_1.json")
    if inst_path.exists():
        inst = VRPTWInstance.from_json(inst_path)
        print(f"\nInstance: {inst.name}  n={inst.n_customers}")

        for method in ["greedy", "cw"]:
            print(f"\n--- ALNS({method}) ---")
            config = ALNSConfig(
                init_method=method,
                max_iterations=50,
                seed=42,
                verbose=True,
            )
            alns = ALNS(inst, config)
            result = alns.run()

            print(f"\n  Final: cost={result.best_cost:.4f}")
            print(f"  Routes: {len(result.best_solution.routes)}")
            print(f"  Breakdown: dist={result.best_cost_breakdown[0]:.2f}, "
                  f"tw={result.best_cost_breakdown[1]:.2f}, "
                  f"k={result.best_cost_breakdown[2]}")
            print(f"  Time: {result.runtime_seconds:.2f}s")
    else:
        print("No test instance found. Run generate_data.py first.")
