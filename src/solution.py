"""
solution.py — Solution representation, cost calc, feasibility & initial solutions

ĐỌC SAU problem.py.

Chứa:
    1. Solution class — representation của 1 lời giải VRPTW
    2. calc_cost()     — tính cost (distance + TW violations + vehicles)
    3. is_feasible()   — kiểm tra capacity, time windows
    4. greedy_init()   — Greedy Insertion heuristic
    5. cw_init()       — Clarke-Wright Savings heuristic

References:
    SPEC.md Section 4
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import NamedTuple

from problem import VRPTWInstance


# =============================================================================
# Section 1: Cost & Feasibility Result
# =============================================================================

class CostBreakdown(NamedTuple):
    """Phân rã cost thành các thành phần để debug."""
    total_distance: float
    tw_violations: float
    num_vehicles: int
    total_cost: float
    violation_details: list[tuple[int, float]] | None = None
        # list of (customer_id, violation_amount) for customers arriving late


# =============================================================================
# Section 2: Solution Class
# =============================================================================

@dataclass
class Solution:
    """
    Đại diện cho 1 lời giải VRPTW — tập hợp các routes.

    Attributes:
        routes: list of routes.
                 Mỗi route là list[int] bắt đầu và kết thúc bằng depot (0).
                 Ví dụ: [[0, 3, 7, 0], [0, 1, 5, 2, 0], [0, 4, 6, 8, 0]]
                 - Depot = 0
                 - Customer ids = 1..N
        instance: VRPTWInstance (để tính cost, check feasible)

    Usage:
        sol = Solution(routes=[[0, 1, 2, 0]], instance=inst)
        cost = sol.calc_cost()
        ok, violations = sol.is_feasible()
        sol.set_routes(new_routes)

    NOTE: Một số method trả về MUTABLE copies. Gọi các method này xong
          KHÔNG mutate trực tiếp kết quả — tạo copy mới nếu cần thay đổi.
    """
    routes: list[list[int]] = field(default_factory=list)
    instance: VRPTWInstance | None = None

    # ---------- Cost Calculation ----------

    def calc_cost(
        self,
        lambda_tw: float = 100.0,
        lambda_vehicles: float = 10.0,
    ) -> float:
        """
        Tính total cost theo công thức:
            cost = distance + λ_tw * TW_violations + λ_vehicles * num_vehicles

        Args:
            lambda_tw: penalty weight cho TW violations (default 100.0)
                      TW violations rất đắt → cao hơn nhiều so với distance
            lambda_vehicles: penalty weight cho số vehicles (default 10.0)
                           fewer vehicles quan trọng hơn distance

        Returns:
            Total cost (float)
        """
        if self.instance is None:
            raise ValueError("Solution.instance is None — cannot calculate cost")

        inst = self.instance
        n = inst.n_customers

        # --- 1. Tổng khoảng cách ---
        total_dist = 0.0
        current_time = 0.0
        for route in self.routes:
            for k in range(len(route) - 1):
                i, j = route[k], route[k + 1]
                travel = inst.normalized_distance(i, j)
                current_time += travel
                total_dist += travel

        # --- 2. TW violations ---
        tw_violations = 0.0
        violation_details: list[tuple[int, float]] = []

        for route in self.routes:
            current_time = 0.0
            for k in range(len(route) - 1):
                i, j = route[k], route[k + 1]
                current_time += inst.normalized_distance(i, j)

                if j != 0:  # là customer (không phải depot)
                    tw = inst.customer_of(j).time_window
                    svc = inst.customer_of(j).service_time

                    # arrival_time > latest_allowed → TW violation
                    if current_time > tw[1]:
                        violation = current_time - tw[1]
                        tw_violations += violation
                        violation_details.append((j, violation))

                    # Nếu đến sớm → chờ đến earliest rồi mới phục vụ
                    if current_time < tw[0]:
                        current_time = tw[0]

                    # Cộng service time
                    current_time += svc

        # --- 3. Số vehicles ---
        num_vehicles = len(self.routes)

        # --- 4. Total cost ---
        total_cost = total_dist + lambda_tw * tw_violations + lambda_vehicles * num_vehicles

        return total_cost

    def calc_cost_breakdown(
        self,
        lambda_tw: float = 100.0,
        lambda_vehicles: float = 10.0,
    ) -> CostBreakdown:
        """
        Tính cost với phân rã chi tiết từng thành phần.
        Dùng cho debug và reporting.
        """
        if self.instance is None:
            raise ValueError("Solution.instance is None")

        inst = self.instance

        total_dist = 0.0
        tw_violations = 0.0
        violation_details: list[tuple[int, float]] = []

        for route in self.routes:
            current_time = 0.0
            for k in range(len(route) - 1):
                i, j = route[k], route[k + 1]
                travel = inst.normalized_distance(i, j)
                current_time += travel
                total_dist += travel

                if j != 0:
                    tw = inst.customer_of(j).time_window
                    svc = inst.customer_of(j).service_time

                    if current_time > tw[1]:
                        violation = current_time - tw[1]
                        tw_violations += violation
                        violation_details.append((j, violation))

                    if current_time < tw[0]:
                        current_time = tw[0]
                    current_time += svc

        num_vehicles = len(self.routes)
        total_cost = total_dist + lambda_tw * tw_violations + lambda_vehicles * num_vehicles

        return CostBreakdown(
            total_distance=round(total_dist, 4),
            tw_violations=round(tw_violations, 4),
            num_vehicles=num_vehicles,
            total_cost=round(total_cost, 4),
            violation_details=violation_details if violation_details else None,
        )

    # ---------- Feasibility Check ----------

    def is_feasible(self) -> tuple[bool, int]:
        """
        Kiểm tra solution có feasible không.

        Check:
            1. Mỗi customer xuất hiện đúng 1 lần
            2. Tổng demand mỗi route ≤ capacity
            3. TW violations (đến muộn) = 0 (return count nếu có)

        Returns:
            (True, 0) nếu feasible
            (False, violation_count) nếu có violations
        """
        if self.instance is None:
            raise ValueError("Solution.instance is None")

        inst = self.instance
        n = inst.n_customers

        # --- 1. Kiểm tra mỗi customer xuất hiện đúng 1 lần ---
        served = set()
        for route in self.routes:
            for node in route:
                if node != 0:  # bỏ depot
                    served.add(node)

        if len(served) != n or served != set(range(1, n + 1)):
            # Thiếu hoặc thừa customer
            missing = set(range(1, n + 1)) - served
            if missing:
                return False, len(missing)

        # --- 2. Kiểm tra capacity ---
        for route in self.routes:
            route_demand = sum(inst.customer_of(node).demand for node in route if node != 0)
            if route_demand > inst.vehicle_capacity:
                return False, -1  # capacity violation

        # --- 3. Kiểm tra TW (chỉ đếm violations, không reject) ---
        tw_violations = 0
        for route in self.routes:
            current_time = 0.0
            for k in range(len(route) - 1):
                i, j = route[k], route[k + 1]
                current_time += inst.normalized_distance(i, j)

                if j != 0:
                    tw = inst.customer_of(j).time_window
                    svc = inst.customer_of(j).service_time

                    if current_time > tw[1]:
                        tw_violations += 1

                    if current_time < tw[0]:
                        current_time = tw[0]
                    current_time += svc

        return True, tw_violations

    def get_served_customers(self) -> set[int]:
        """Trả về set các customer ids đang được phục vụ."""
        served = set()
        for route in self.routes:
            for node in route:
                if node != 0:
                    served.add(node)
        return served

    def get_removed_customers(self) -> set[int]:
        """Trả về set các customer ids CHƯA được phục vụ (cần repair)."""
        if self.instance is None:
            return set()
        served = self.get_served_customers()
        all_customers = set(range(1, self.instance.n_customers + 1))
        return all_customers - served

    def copy(self) -> Solution:
        """Tạo deep copy của solution."""
        return Solution(
            routes=copy.deepcopy(self.routes),
            instance=self.instance,
        )

    def __repr__(self) -> str:
        n_routes = len(self.routes)
        n_vehicles = len(self.routes)
        cost = self.calc_cost() if self.instance else 0.0
        return (
            f"Solution(routes={n_routes}, vehicles={n_vehicles}, "
            f"cost={cost:.4f})"
        )


# =============================================================================
# Section 3: Initial Solutions
# =============================================================================

def greedy_init(inst: VRPTWInstance, seed: int | None = None) -> Solution:
    """
    Greedy Insertion Heuristic cho VRPTW.

    Thuật toán:
        1. Bắt đầu với tất cả customers chưa được gán (unassigned)
        2. Mỗi lần lặp: chọn customer có chi phí chèn thấp nhất
           vào route hiện có hoặc tạo route mới
        3. Feasibility check: capacity + time window
        4. Lặp đến khi tất cả customers được gán

    Đặc điểm:
        - O(N² × routes) — chấp nhận được cho N=100
        - Ưu tiên customers dễ gán trước → solution nhanh nhưng sub-optimal

    Args:
        inst: VRPTWInstance
        seed: random seed (optional)

    Returns:
        Solution với routes feasible (hoặc gần feasible)
    """
    rng = random.Random(seed)

    n = inst.n_customers
    assigned: set[int] = set()
    routes: list[list[int]] = []

    # Các customers chưa được gán
    unassigned = set(range(1, n + 1))

    while unassigned:
        # Tìm customer tốt nhất để chèn
        # Thử chèn từng customer chưa gán vào mỗi vị trí có thể
        best_customer = None
        best_insertion = None
        best_cost = float("inf")

        for cust in unassigned:
            # Thử chèn customer vào các vị trí khác nhau
            # Tính cost tăng thêm khi chèn cust vào route r hoặc tạo route mới

            # --- Option A: tạo route mới riêng cho cust ---
            new_route = [0, cust, 0]
            if _is_route_feasible(new_route, inst):
                # cost = chỉ khoảng cách từ depot → cust → depot
                route_dist = (
                    inst.normalized_distance(0, cust) +
                    inst.normalized_distance(cust, 0)
                )
                if route_dist < best_cost:
                    best_cost = route_dist
                    best_customer = cust
                    best_insertion = ("new_route", new_route)

            # --- Option B: chèn vào route hiện có ---
            for route in routes:
                # Thử chèn vào từng vị trí trong route
                for pos in range(1, len(route)):  # sau depot hoặc giữa các nodes
                    new_route = route[:pos] + [cust] + route[pos:]
                    if _is_route_feasible(new_route, inst):
                        # Tính cost increase
                        prev, next_node = route[pos - 1], route[pos]
                        old_dist = inst.normalized_distance(prev, next_node)
                        new_dist = (
                            inst.normalized_distance(prev, cust) +
                            inst.normalized_distance(cust, next_node)
                        )
                        cost_increase = new_dist - old_dist
                        if cost_increase < best_cost:
                            best_cost = cost_increase
                            best_customer = cust
                            best_insertion = ("insert", route, pos)

        if best_customer is None:
            # Không tìm được insertion feasible → tạo route mới (force)
            # Thử từng customer tạo route riêng
            for cust in unassigned:
                new_route = [0, cust, 0]
                routes.append(new_route)
                assigned.add(cust)
                unassigned.remove(cust)
                break
            continue

        # Apply insertion
        if best_insertion[0] == "new_route":
            routes.append(best_insertion[1])
        else:
            # ("insert", route, pos)
            _, route, pos = best_insertion
            route_idx = routes.index(route)
            routes[route_idx] = route[:pos] + [best_customer] + route[pos:]

        assigned.add(best_customer)
        unassigned.remove(best_customer)

    return Solution(routes=routes, instance=inst)


def cw_init(inst: VRPTWInstance, seed: int | None = None) -> Solution:
    """
    Clarke-Wright Savings Algorithm cho VRPTW.

    Thuật toán:
        1. Khởi tạo: mỗi customer = 1 route riêng [0, i, 0]
        2. Tính savings s(i,j) = d(i,0) + d(0,j) - d(i,j) cho mọi cặp (i,j)
        3. Sắp xếp savings giảm dần
        4. Lần lượt gộp routes nếu:
           - i và j ở cuối 2 routes khác nhau
           - Gộp không vi phạm capacity và time windows
        5. Lặp đến khi không gộp được nữa

    Đặc điểm:
        - O(N² log N) cho savings computation
        - Thường cho solution tốt hơn Greedy
        - Dùng trong paper làm initial cho PPO-ALNS

    Args:
        inst: VRPTWInstance
        seed: random seed (optional)

    Returns:
        Solution với routes feasible
    """
    rng = random.Random(seed)
    n = inst.n_customers

    # --- Bước 1: Khởi tạo routes riêng biệt ---
    routes: list[list[int]] = [[0, i, 0] for i in range(1, n + 1)]

    # Map mỗi customer → route index hiện tại
    cust_to_route: dict[int, int] = {i: i for i in range(n)}

    # --- Bước 2: Tính savings cho tất cả cặp ---
    # savings[i,j] = d(i,0) + d(0,j) - d(i,j)
    savings: list[tuple[float, int, int]] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = (
                inst.normalized_distance(i, 0) +
                inst.normalized_distance(0, j) -
                inst.normalized_distance(i, j)
            )
            savings.append((s, i, j))

    # Sắp xếp giảm dần theo savings
    savings.sort(key=lambda x: x[0], reverse=True)

    # --- Bước 3: Gộp routes theo savings ---
    for s_val, i, j in savings:
        # Tìm route hiện tại của i và j
        if i not in cust_to_route or j not in cust_to_route:
            continue

        route_i_idx = cust_to_route[i]
        route_j_idx = cust_to_route[j]

        if route_i_idx == route_j_idx:
            continue  # đã cùng route

        route_i = routes[route_i_idx]
        route_j = routes[route_j_idx]

        # Kiểm tra i ở cuối route_i, j ở đầu route_j
        # Có 2 trường hợp gộp: i ở cuối route_i, j ở đầu route_j
        # Hoặc ngược lại: j ở cuối route_j, i ở đầu route_i

        merged = _try_merge_routes(route_i, route_j, i, j, inst)

        if merged is not None:
            # Xóa 2 routes cũ, thêm route mới
            # Cập nhật cust_to_route
            new_route = merged

            # Route nhỏ hơn bị xóa trước (để index không bị lệch)
            r1, r2 = sorted([route_i_idx, route_j_idx])

            for cust in route_j:
                if cust != 0:
                    cust_to_route[cust] = r1
            for cust in route_i:
                if cust != 0 and cust not in cust_to_route:
                    pass  # đã update ở trên

            # Rebuild cust_to_route với routes mới
            cust_to_route = {}
            for ridx, r in enumerate(routes):
                for cust in r:
                    if cust != 0:
                        cust_to_route[cust] = ridx

            # Thực hiện gộp
            routes[r1] = new_route
            del routes[r2]

            # Cập nhật cust_to_route lại
            cust_to_route = {}
            for ridx, r in enumerate(routes):
                for cust in r:
                    if cust != 0:
                        cust_to_route[cust] = ridx

    return Solution(routes=routes, instance=inst)


# =============================================================================
# Section 4: Private helpers
# =============================================================================

def _is_route_feasible(route: list[int], inst: VRPTWInstance) -> bool:
    """
    Kiểm tra 1 route có feasible không (capacity + TW).

    Args:
        route: [0, c1, c2, ..., 0]
        inst: VRPTWInstance

    Returns:
        True nếu feasible
    """
    # Capacity check
    demand = sum(inst.customer_of(node).demand for node in route if node != 0)
    if demand > inst.vehicle_capacity:
        return False

    # TW check (tính arrival time)
    current_time = 0.0
    for k in range(len(route) - 1):
        i, j = route[k], route[k + 1]
        current_time += inst.normalized_distance(i, j)

        if j != 0:
            tw = inst.customer_of(j).time_window
            svc = inst.customer_of(j).service_time

            # Late arrival → infeasible
            if current_time > tw[1]:
                return False

            # Earliest arrival → wait
            if current_time < tw[0]:
                current_time = tw[0]

            current_time += svc

    return True


def _try_merge_routes(
    route_i: list[int],
    route_j: list[int],
    i: int,
    j: int,
    inst: VRPTWInstance,
) -> list[int] | None:
    """
    Thử gộp route_i và route_j qua savings (i,j).
    Chỉ gộp khi i ở cuối route_i và j ở đầu route_j.
    Kiểm tra capacity + TW sau khi gộp.

    Returns:
        Route mới nếu feasible, None nếu không
    """
    # i phải ở cuối route_i (trước depot), j ở đầu route_j (sau depot)
    if route_i[-2] != i or route_j[1] != j:
        return None

    # Tạo route mới: route_i[:-1] + route_j[1:]
    # route_i = [0, ..., i, 0] → bỏ [0, i] cuối → [..., i]
    # route_j = [0, j, ...] → bỏ [0] đầu → [j, ...]
    merged = route_i[:-1] + route_j[1:]

    # Kiểm tra capacity
    demand = sum(inst.customer_of(node).demand for node in merged if node != 0)
    if demand > inst.vehicle_capacity:
        return None

    # Kiểm tra TW
    current_time = 0.0
    for k in range(len(merged) - 1):
        a, b = merged[k], merged[k + 1]
        current_time += inst.normalized_distance(a, b)

        if b != 0:
            tw = inst.customer_of(b).time_window
            svc = inst.customer_of(b).service_time

            if current_time > tw[1]:
                return None  # TW violation

            if current_time < tw[0]:
                current_time = tw[0]
            current_time += svc

    return merged


def _route_distance(route: list[int], inst: VRPTWInstance) -> float:
    """Tính tổng khoảng cách của 1 route."""
    return sum(
        inst.normalized_distance(route[k], route[k + 1])
        for k in range(len(route) - 1)
    )


# =============================================================================
# Section 5: Factory functions
# =============================================================================

def make_initial_solution(
    inst: VRPTWInstance,
    method: str = "greedy",
    seed: int | None = None,
) -> Solution:
    """
    Tạo initial solution bằng method được chỉ định.

    Args:
        inst: VRPTWInstance
        method: "greedy" | "cw"
        seed: random seed

    Returns:
        Solution

    Raises:
        ValueError: method không hợp lệ
    """
    if method == "greedy":
        return greedy_init(inst, seed=seed)
    elif method == "cw":
        return cw_init(inst, seed=seed)
    else:
        raise ValueError(f"Unknown init method: {method!r}. Use 'greedy' or 'cw'")
