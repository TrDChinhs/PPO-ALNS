"""
state_encoder.py — Extract state features for PPO agent

Mô tả 10 features theo SPEC.md Section 6.1:
    1. search_progress     — t / T_max
    2. solution_delta      — (c_prev - c_curr) / c_prev
    3. init_cost          — c_init / baseline
    4. best_cost          — c_best / baseline
    5. destroy_usage      — frequency of each D op (5 features)
    6. repair_usage       — frequency of each R op (3 features)
    7. demand             — per customer demand (n features)
    8. time_windows       — [a, b] per customer (2n features)
    9. service_times      — per customer (n features)
    10. travel_times      — pairwise distances (n×n features)

Total state dim = n² + 4n + 12
  - n=20: 492 dims
  - n=50: 2712 dims
  - n=100: 10412 dims

References:
    SPEC.md Section 6.1
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from problem import VRPTWInstance
from solution import Solution


# =============================================================================
# Constants
# =============================================================================

NUM_DESTROY_OPS = 5
NUM_REPAIR_OPS = 3
STATE_DIM_MULTIPLIER = 1.0  # For dynamic n


# =============================================================================
# State Encoder
# =============================================================================

class StateEncoder:
    """
    Encode VRPTW state → numpy array cho PPO.

    Dùng flattened representation. Mỗi feature được normalize về [0, 1].

    Usage:
        encoder = StateEncoder(inst)
        state = encoder.encode(solution=sol, iteration=10, max_iter=100,
                               d_counts=[...], r_counts=[...])
        # state.shape = (n² + 7n + 11,)
    """

    def __init__(self, inst: VRPTWInstance) -> None:
        self.inst = inst
        self.n = inst.n_customers

        # Precompute static features
        self._travel_times = self._compute_travel_matrix()
        self._demand_array = self._build_demand_array()
        self._tw_array = self._build_tw_array()
        self._service_array = self._build_service_array()

        # Baseline cost: max possible distance + max penalties
        # max_dist ≈ n * sqrt(2) / 50 ≈ n * 0.0283
        max_dist = self.n * 0.0283 * 2  # round trip estimate
        self._baseline = max_dist + 100.0 * self.n + 10.0 * self.n

    def _compute_travel_matrix(self) -> NDArray[np.float64]:
        """Compute normalized pairwise distance matrix (n×n)."""
        n = self.n
        mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                mat[i, j] = self.inst.normalized_distance(i + 1, j + 1)
        return mat

    def _build_demand_array(self) -> NDArray[np.float64]:
        """Build demand array for all customers, normalized [0,1]."""
        demands = np.array([
            self.inst.customer_of(i).demand
            for i in range(1, self.n + 1)
        ], dtype=np.float64)
        # Normalize: demand range is [1, 16]
        return demands / 16.0

    def _build_tw_array(self) -> NDArray[np.float64]:
        """Build TW array: [earliest, latest] per customer, normalized [0,1]."""
        arr = np.zeros((self.n, 2), dtype=np.float64)
        for i in range(1, self.n + 1):
            cust = self.inst.customer_of(i)
            # max_travel_time = 1.0 (normalized)
            arr[i - 1, 0] = cust.time_window[0]  # already [0, 1]
            arr[i - 1, 1] = cust.time_window[1]
        return arr

    def _build_service_array(self) -> NDArray[np.float64]:
        """Build service time array, normalized [0,1]."""
        services = np.array([
            self.inst.customer_of(i).service_time
            for i in range(1, self.n + 1)
        ], dtype=np.float64)
        # Service range is [0.05, 0.1]
        return services / 0.1

    @property
    def state_dim(self) -> int:
        """Tính state dimension cho instance size n.

        Breakdown: 4(scalars) + 5(destroy) + 3(repair) + n(demand)
                 + 2n(tw) + n(service) + n²(travel)
                 = n² + 4n + 12
        """
        n = self.n
        return n * n + 4 * n + 12

    def encode(
        self,
        solution: Solution,
        iteration: int,
        max_iterations: int,
        destroy_counts: list[int],
        repair_counts: list[int],
        prev_cost: float | None = None,
    ) -> NDArray[np.float32]:
        """
        Encode current ALNS state → numpy array.

        Args:
            solution: current Solution
            iteration: current ALNS iteration (1-based)
            max_iterations: total iterations (T_max)
            destroy_counts: số lần mỗi destroy op được gọi (len=5)
            repair_counts: số lần mỗi repair op được gọi (len=3)
            prev_cost: cost ở iteration trước (None nếu là step đầu)

        Returns:
            numpy array shape (state_dim,) dtype float32, normalized [0,1]
        """
        n = self.n
        parts: list[NDArray[np.float32]] = []

        # --- Feature 1: search_progress ---
        progress = iteration / max_iterations
        parts.append(np.array([progress], dtype=np.float32))

        # --- Feature 2: solution_delta ---
        curr_cost = solution.calc_cost()
        if prev_cost is not None and prev_cost > 0:
            delta = (prev_cost - curr_cost) / prev_cost
        else:
            delta = 0.0
        delta_clamped = np.array([np.clip(delta, -1.0, 1.0)], dtype=np.float32)
        parts.append(delta_clamped)

        # --- Feature 3: init_cost ---
        init_cost = solution.calc_cost()
        init_cost_norm = np.array([np.clip(curr_cost / self._baseline, 0.0, 1.0)], dtype=np.float32)
        parts.append(init_cost_norm)

        # --- Feature 4: best_cost ---
        best_cost_norm = np.array([np.clip(curr_cost / self._baseline, 0.0, 1.0)], dtype=np.float32)
        parts.append(best_cost_norm)

        # --- Feature 5: destroy_usage (5 features) ---
        total_d = sum(destroy_counts)
        if total_d > 0:
            d_freq = np.array(destroy_counts, dtype=np.float32) / total_d
        else:
            d_freq = np.ones(5, dtype=np.float32) * (1.0 / 5.0)
        parts.append(d_freq)

        # --- Feature 6: repair_usage (3 features) ---
        total_r = sum(repair_counts)
        if total_r > 0:
            r_freq = np.array(repair_counts, dtype=np.float32) / total_r
        else:
            r_freq = np.ones(3, dtype=np.float32) * (1.0 / 3.0)
        parts.append(r_freq)

        # --- Feature 7: demand (n features) ---
        parts.append(self._demand_array.astype(np.float32))

        # --- Feature 8: time_windows (2n features) ---
        parts.append(self._tw_array.flatten().astype(np.float32))

        # --- Feature 9: service_times (n features) ---
        parts.append(self._service_array.astype(np.float32))

        # --- Feature 10: travel_times (n×n features) ---
        parts.append(self._travel_times.flatten().astype(np.float32))

        # Concatenate all parts
        state = np.concatenate(parts)
        return state

    def encode_from_stats(
        self,
        curr_cost: float,
        iteration: int,
        max_iterations: int,
        destroy_counts: list[int],
        repair_counts: list[int],
        prev_cost: float | None = None,
    ) -> NDArray[np.float32]:
        """
        Encode state từ cost stats (không cần full solution object).

        Dùng khi chỉ cần cost information, không cần solution details.

        Args:
            curr_cost: current best cost
            iteration: current iteration
            max_iterations: total iterations
            destroy_counts: len=5
            repair_counts: len=3
            prev_cost: previous cost

        Returns:
            numpy array shape (state_dim,)
        """
        n = self.n
        parts: list[NDArray[np.float32]] = []

        # Feature 1: search_progress
        progress = iteration / max_iterations
        parts.append(np.array([progress], dtype=np.float32))

        # Feature 2: solution_delta
        if prev_cost is not None and prev_cost > 0:
            delta = (prev_cost - curr_cost) / prev_cost
        else:
            delta = 0.0
        delta_clamped = np.array([np.clip(delta, -1.0, 1.0)], dtype=np.float32)
        parts.append(delta_clamped)

        # Feature 3 & 4: init_cost, best_cost (normalized by baseline)
        init_norm = np.array([np.clip(curr_cost / self._baseline, 0.0, 1.0)], dtype=np.float32)
        parts.append(init_norm)
        parts.append(init_norm)

        # Feature 5: destroy_usage
        total_d = sum(destroy_counts)
        if total_d > 0:
            d_freq = np.array(destroy_counts, dtype=np.float32) / total_d
        else:
            d_freq = np.ones(5, dtype=np.float32) * (1.0 / 5.0)
        parts.append(d_freq)

        # Feature 6: repair_usage
        total_r = sum(repair_counts)
        if total_r > 0:
            r_freq = np.array(repair_counts, dtype=np.float32) / total_r
        else:
            r_freq = np.ones(3, dtype=np.float32) * (1.0 / 3.0)
        parts.append(r_freq)

        # Static instance features (7-10)
        parts.append(self._demand_array.astype(np.float32))
        parts.append(self._tw_array.flatten().astype(np.float32))
        parts.append(self._service_array.astype(np.float32))
        parts.append(self._travel_times.flatten().astype(np.float32))

        return np.concatenate(parts)
