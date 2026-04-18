"""
operators.py — Destroy & Repair operators cho ALNS

ĐỌC SAU solution.py.

Chứa:
    5 Destroy Operators:
        D1_random_destroy()
        D2_string_destroy()
        D3_route_destroy()
        D4_worst_destroy()
        D5_sequence_destroy()

    3 Repair Operators:
        R1_greedy_repair()
        R2_critical_repair()
        R3_regret_repair()

Mỗi operator nhận (solution, instance, destroy_scale) → trả về solution MỚI (sau khi destroy/repair).

References:
    SPEC.md Section 5.1 & 5.2
"""

from __future__ import annotations

import random
from typing import Protocol

from problem import VRPTWInstance
from solution import Solution


# =============================================================================
# Section 1: Operator Interfaces
# =============================================================================

class DestroyOperator(Protocol):
    """
    Protocol cho destroy operator.

    Signature:
        def destroy(solution: Solution, d: int, rng: random.Random) -> set[int]:
            ...
            return set of removed customer IDs

    Args:
        solution: solution hiện tại
        d: số lượng customer cần xóa (destroy scale)
        rng: random generator

    Returns:
        Set of customer IDs đã bị xóa (để repair sử dụng)
    """

    def __call__(self, solution: Solution, d: int, rng: random.Random) -> set[int]:
        ...


class RepairOperator(Protocol):
    """
    Protocol cho repair operator.

    Signature:
        def repair(solution: Solution, removed: set[int], rng: random.Random) -> Solution:
            ...
            return NEW solution (đã chèn removed customers)

    Args:
        solution: solution SAU KHI destroy (đã xóa removed customers)
        removed: set of customer IDs cần chèn lại
        rng: random generator

    Returns:
        NEW Solution với removed customers đã được chèn
    """

    def __call__(self, solution: Solution, removed: set[int], rng: random.Random) -> Solution:
        ...


# =============================================================================
# Section 2: Destroy Operators (5 operators)
# =============================================================================

def d1_random_destroy(solution: Solution, d: int, rng: random.Random) -> set[int]:
    """
    D1 — Random Destroy.

    Xóa ngẫu nhiên d customers khỏi solution.

    Thuật toán:
        1. Liệt kê tất cả customers đang được phục vụ
        2. Sample ngẫu nhiên d customers
        3. Xóa khỏi routes

    Args:
        d: số lượng customers cần xóa

    Returns:
        Set of removed customer IDs
    """
    served = list(solution.get_served_customers())
    if not served:
        return set()

    # Số lượng thực sự xóa = min(d, len(served))
    actual_d = min(d, len(served))

    # Sample ngẫu nhiên
    removed = set(rng.sample(served, actual_d))

    # Xóa khỏi routes (tạo routes mới)
    new_routes: list[list[int]] = []
    for route in solution.routes:
        new_route = [node for node in route if node not in removed]
        if len(new_route) > 1:  # còn ít nhất depot và 1 node
            new_routes.append(new_route)
        elif len(new_route) == 1 and new_route[0] == 0:
            # Route chỉ còn depot → bỏ route này
            pass

    solution.routes = new_routes
    return removed


def d2_string_destroy(solution: Solution, d: int, rng: random.Random) -> set[int]:
    """
    D2 — String Destroy.

    Xóa đoạn route liên tục (string) chứa 1 random anchor customer.

    Thuật toán:
        1. Chọn ngẫu nhiên 1 anchor customer
        2. Xóa anchor + các customers liền kề trong cùng route
        3. Số lượng xóa = min(d, độ dài route)

    Args:
        d: số lượng customers cần xóa

    Returns:
        Set of removed customer IDs
    """
    served = list(solution.get_served_customers())
    if not served:
        return set()

    # Chọn random anchor
    anchor = rng.choice(served)

    # Tìm route chứa anchor
    target_route = None
    route_idx = -1
    node_idx_in_route = -1
    for ridx, route in enumerate(solution.routes):
        if anchor in route:
            target_route = route
            route_idx = ridx
            node_idx_in_route = route.index(anchor)
            break

    if target_route is None:
        return set()

    # Xóa d customers bắt đầu từ anchor position (bao gồm cả anchor)
    actual_d = min(d, len(target_route) - 2)  # -2 vì bỏ depot
    removed: set[int] = set()

    for offset in range(actual_d):
        pos = node_idx_in_route + offset
        if pos < len(target_route):
            node = target_route[pos]
            if node != 0:  # không xóa depot
                removed.add(node)

    # Cập nhật routes
    route = solution.routes[route_idx]
    new_route = [node for node in route if node not in removed]
    if len(new_route) > 1:
        solution.routes[route_idx] = new_route
    else:
        # Bỏ route chỉ còn depot
        del solution.routes[route_idx]

    return removed


def d3_route_destroy(solution: Solution, d: int, rng: random.Random) -> set[int]:
    """
    D3 — Route Destroy.

    Xóa cả 1 route hoặc một phần route dựa trên inverse-length probability.

    Thuật toán:
        1. Tính probability cho mỗi route: p(route) ∝ 1/length
           (route ngắn có xác suất cao hơn bị xóa)
        2. Chọn 1 route theo probability
        3. Xóa toàn bộ customers trong route đó

    Args:
        d: số lượng customers cần xóa (dùng để quyết định xóa mấy routes)

    Returns:
        Set of removed customer IDs
    """
    if not solution.routes:
        return set()

    # Tính inverse length weights
    weights: list[float] = []
    for route in solution.routes:
        # weight = 1 / (số customer trong route)
        n_cust = len(route) - 2  # bỏ 2 depot
        if n_cust <= 0:
            weights.append(0.0)
        else:
            weights.append(1.0 / n_cust)

    # Normalize thành probability
    total = sum(weights)
    if total == 0:
        return set()

    probs = [w / total for w in weights]

    # Chọn route theo probability
    route_idx = rng.choices(range(len(solution.routes)), weights=probs, k=1)[0]
    removed_route = solution.routes[route_idx]

    # Xóa tất cả customers trong route đó
    removed: set[int] = set(node for node in removed_route if node != 0)

    # Xóa route
    del solution.routes[route_idx]

    return removed


def d4_worst_destroy(solution: Solution, d: int, rng: random.Random) -> set[int]:
    """
    D4 — Worst-Cost Destroy.

    Xóa customers có cost impact cao nhất (có randomization để explore).

    Thuật toán:
        1. Tính cost impact của mỗi customer trong solution
           impact(c) = reduction in total distance nếu c bị xóa
        2. Sắp xếp giảm dần theo impact
        3. Chọn top d customers với probability proportional to impact
           (thêm randomization factor)
        4. Xóa các customers được chọn

    Args:
        d: số lượng customers cần xóa

    Returns:
        Set of removed customer IDs
    """
    if solution.instance is None:
        return set()

    inst = solution.instance
    served = list(solution.get_served_customers())
    if not served:
        return set()

    # Tính cost impact cho mỗi customer
    # impact(c) = tổng khoảng cách kề cạnh - khoảng cách trực tiếp
    impacts: dict[int, float] = {}
    for cust in served:
        # Tìm vị trí trong routes
        for route in solution.routes:
            if cust not in route:
                continue
            idx = route.index(cust)

            # prev và next nodes (có thể là depot 0)
            prev_node = route[idx - 1]
            next_node = route[idx + 1] if idx + 1 < len(route) else route[idx - 1]

            # Nếu prev/next là depot → không có impact
            if prev_node == next_node:  # chỉ có 1 customer trong route
                impact = 0.0
            else:
                # Khoảng cách hiện tại: prev→cust→next
                current_dist = (
                    inst.normalized_distance(prev_node, cust) +
                    inst.normalized_distance(cust, next_node)
                )
                # Khoảng cách nếu bỏ cust: prev→next
                bypass_dist = inst.normalized_distance(prev_node, next_node)
                impact = current_dist - bypass_dist

            impacts[cust] = impact
            break

    # Sắp xếp giảm dần theo impact
    sorted_customers = sorted(served, key=lambda c: impacts.get(c, 0.0), reverse=True)

    # Chọn top d với probability proportional to impact + randomization
    actual_d = min(d, len(sorted_customers))
    candidates = sorted_customers[:actual_d * 2]  # lấy top 2d làm candidates

    # Probability proportional to impact (thêm small random noise)
    weights: list[float] = []
    for cust in candidates:
        base = impacts.get(cust, 0.0)
        noise = rng.uniform(0.01, 0.1) * base  # 1-10% noise
        weights.append(max(0.01, base + noise))

    total = sum(weights)
    probs = [w / total for w in weights]

    removed = set(rng.choices(candidates, weights=probs, k=actual_d))

    # Xóa khỏi routes
    new_routes: list[list[int]] = []
    for route in solution.routes:
        new_route = [node for node in route if node not in removed]
        if len(new_route) > 1:
            new_routes.append(new_route)
    solution.routes = new_routes

    return removed


def d5_sequence_destroy(solution: Solution, d: int, rng: random.Random) -> set[int]:
    """
    D5 — Sequence Destroy.

    Xóa contiguous subsequence từ concatenated route representation.

    Thuật toán:
        1. Nối tất cả routes thành 1 sequence: [c1, c2, ..., cN]
           (bỏ depot, lấy customer thứ tự)
        2. Chọn random start position
        3. Xóa d customers liên tiếp từ start

    Args:
        d: số lượng customers cần xóa

    Returns:
        Set of removed customer IDs
    """
    served = list(solution.get_served_customers())
    if not served:
        return set()

    actual_d = min(d, len(served))

    # Nối tất cả customers theo thứ tự xuất hiện trong routes
    all_customers: list[int] = []
    for route in solution.routes:
        for node in route:
            if node != 0:
                all_customers.append(node)

    if not all_customers:
        return set()

    # Chọn random start position
    start = rng.randint(0, len(all_customers) - 1)

    # Xóa d customers liên tiếp (wrap around nếu cần)
    removed: set[int] = set()
    pos = start
    for _ in range(actual_d):
        removed.add(all_customers[pos % len(all_customers)])
        pos += 1

    # Xóa khỏi routes
    new_routes: list[list[int]] = []
    for route in solution.routes:
        new_route = [node for node in route if node not in removed]
        if len(new_route) > 1:
            new_routes.append(new_route)
    solution.routes = new_routes

    return removed


# Registry cho destroy operators (để Roulette wheel dùng)
DESTROY_OPERATORS: list[tuple[str, DestroyOperator]] = [
    ("random", d1_random_destroy),
    ("string", d2_string_destroy),
    ("route", d3_route_destroy),
    ("worst", d4_worst_destroy),
    ("sequence", d5_sequence_destroy),
]


# =============================================================================
# Section 3: Repair Operators (3 operators)
# =============================================================================

def r1_greedy_repair(
    solution: Solution,
    removed: set[int],
    rng: random.Random,
) -> Solution:
    """
    R1 — Greedy Repair.

    Chèn mỗi removed customer vào vị trí tốt nhất (minimize cost increase).

    Thuật toán:
        1. Với mỗi removed customer:
           a. Thử chèn vào mọi vị trí trong mọi route
           b. Tính cost increase cho mỗi insertion
           c. Chọn vị trí có cost increase NHỎ NHẤT
           d. Nếu không chèn được vào route nào → tạo route mới
        2. Lặp đến khi tất cả removed được chèn

    Args:
        removed: set of customer IDs cần chèn lại

    Returns:
        NEW Solution với removed customers đã được chèn
    """
    if solution.instance is None:
        return solution

    inst = solution.instance
    remaining = set(removed)

    while remaining:
        cust = remaining.pop()

        best_insertion = None
        best_cost_increase = float("inf")

        # --- Option 1: Chèn vào route hiện có ---
        for route_idx, route in enumerate(solution.routes):
            for pos in range(1, len(route)):  # không chèn trước depot (pos=0 là depot)
                # Tạo route mới tạm
                new_route = route[:pos] + [cust] + route[pos:]

                if not _is_route_feasible(new_route, inst):
                    continue

                # Tính cost increase
                prev_node = route[pos - 1]
                next_node = route[pos]
                old_dist = inst.normalized_distance(prev_node, next_node)
                new_dist = (
                    inst.normalized_distance(prev_node, cust) +
                    inst.normalized_distance(cust, next_node)
                )
                cost_increase = new_dist - old_dist

                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_insertion = ("insert", route_idx, pos)

        # --- Option 2: Tạo route mới ---
        new_route = [0, cust, 0]
        if _is_route_feasible(new_route, inst):
            route_dist = (
                inst.normalized_distance(0, cust) +
                inst.normalized_distance(cust, 0)
            )
            if route_dist < best_cost_increase:
                best_cost_increase = route_dist
                best_insertion = ("new",)

        # --- Apply best insertion ---
        if best_insertion is None:
            # Không tìm được vị trí feasible → force tạo route mới
            solution.routes.append([0, cust, 0])
        elif best_insertion[0] == "insert":
            _, route_idx, pos = best_insertion
            route = solution.routes[route_idx]
            solution.routes[route_idx] = route[:pos] + [cust] + route[pos:]
        else:
            # new route
            solution.routes.append([0, cust, 0])

    return solution


def r2_critical_repair(
    solution: Solution,
    removed: set[int],
    rng: random.Random,
) -> Solution:
    """
    R2 — Criticality-Based Repair.

    Ưu tiên chèn customers có criticality CAO trước.
    Criticality = weighted sum của:
        - Demand magnitude (demand lớn → khó fit vào route)
        - Time window tightness (TW hẹp → ít thời gian linh hoạt)
        - Depot proximity (xa depot → khó sắp xếp)

    Thuật toán:
        1. Tính criticality cho mỗi removed customer
        2. Sắp xếp removed customers giảm dần theo criticality
        3. Lần lượt chèn mỗi customer (greedy best insertion)

    Args:
        removed: set of customer IDs cần chèn lại

    Returns:
        NEW Solution với removed customers đã được chèn
    """
    if solution.instance is None:
        return solution

    inst = solution.instance

    # Tính criticality cho mỗi customer
    criticalities: dict[int, float] = {}
    for cust in removed:
        c_data = inst.customer_of(cust)
        demand = c_data.demand
        tw_start, tw_end = c_data.time_window
        tw_width = tw_end - tw_start
        tw_tightness = 1.0 / max(tw_width, 0.01)  # TW hẹp → cao

        depot_dist = inst.normalized_distance(0, cust)

        # Normalize
        demand_score = demand / 16.0
        tw_score = min(tw_tightness / 20.0, 1.0)  # scale
        depot_score = depot_dist / 0.02  # scale

        # Criticality = weighted sum
        crit = 0.4 * demand_score + 0.4 * tw_score + 0.2 * depot_score
        criticalities[cust] = crit

    # Sắp xếp giảm dần theo criticality
    sorted_customers = sorted(removed, key=lambda c: criticalities.get(c, 0.0), reverse=True)

    # Lần lượt chèn mỗi customer (greedy best insertion)
    for cust in sorted_customers:
        best_insertion = None
        best_cost_increase = float("inf")

        # Thử chèn vào route hiện có
        for route_idx, route in enumerate(solution.routes):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]

                if not _is_route_feasible(new_route, inst):
                    continue

                prev_node = route[pos - 1]
                next_node = route[pos]
                old_dist = inst.normalized_distance(prev_node, next_node)
                new_dist = (
                    inst.normalized_distance(prev_node, cust) +
                    inst.normalized_distance(cust, next_node)
                )
                cost_increase = new_dist - old_dist

                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_insertion = ("insert", route_idx, pos)

        # Tạo route mới nếu cần
        new_route = [0, cust, 0]
        if _is_route_feasible(new_route, inst):
            route_dist = (
                inst.normalized_distance(0, cust) +
                inst.normalized_distance(cust, 0)
            )
            if route_dist < best_cost_increase:
                best_insertion = ("new",)

        # Apply
        if best_insertion is None:
            solution.routes.append([0, cust, 0])
        elif best_insertion[0] == "insert":
            _, route_idx, pos = best_insertion
            route = solution.routes[route_idx]
            solution.routes[route_idx] = route[:pos] + [cust] + route[pos:]
        else:
            solution.routes.append([0, cust, 0])

    return solution


def r3_regret_repair(
    solution: Solution,
    removed: set[int],
    rng: random.Random,
) -> Solution:
    """
    R3 — Regret-k Repair.

    Chọn customer có "regret" lớn nhất để chèn trước.
    Regret(c) = (cost của insertion tốt thứ 2) - (cost của insertion tốt nhất)

    Ý tưởng: Nếu customer chỉ có 1 vị trí tốt → chèn sớm.
    Nếu customer có nhiều vị trí tốt → chèn sau cũng được.

    Thuật toán:
        1. Với mỗi removed customer:
           a. Tính cost của tất cả insertions (vào mọi route, mọi vị trí)
           b. Sắp xếp tăng dần
           c. Regret = cost[1] - cost[0] (nếu có ít nhất 2 options)
        2. Chọn customer có regret LỚN NHẤT
        3. Chèn customer đó vào vị trí tốt nhất
        4. Lặp đến khi hết removed

    Args:
        removed: set of customer IDs cần chèn lại

    Returns:
        NEW Solution với removed customers đã được chèn
    """
    if solution.instance is None:
        return solution

    inst = solution.instance
    remaining = set(removed)

    while remaining:
        # Tính regret cho mỗi customer
        regrets: dict[int, float] = {}
        best_insertions: dict[int, tuple[int, int]] = {}  # cust → (route_idx, pos)

        for cust in remaining:
            # Liệt kê tất cả insertions
            insertion_costs: list[tuple[float, int, int]] = []  # (cost, route_idx, pos)

            # Vào route hiện có
            for route_idx, route in enumerate(solution.routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    if not _is_route_feasible(new_route, inst):
                        continue

                    prev_node = route[pos - 1]
                    next_node = route[pos]
                    old_dist = inst.normalized_distance(prev_node, next_node)
                    new_dist = (
                        inst.normalized_distance(prev_node, cust) +
                        inst.normalized_distance(cust, next_node)
                    )
                    insertion_costs.append((new_dist - old_dist, route_idx, pos))

            # Tạo route mới
            new_route = [0, cust, 0]
            if _is_route_feasible(new_route, inst):
                route_dist = (
                    inst.normalized_distance(0, cust) +
                    inst.normalized_distance(cust, 0)
                )
                insertion_costs.append((route_dist, -1, -1))  # -1 = new route

            if not insertion_costs:
                # Không có insertion feasible → bỏ qua (edge case)
                regrets[cust] = -float("inf")
                best_insertions[cust] = (-1, -1)
                continue

            insertion_costs.sort(key=lambda x: x[0])

            # Regret = cost của tốt thứ 2 - cost của tốt nhất
            if len(insertion_costs) >= 2:
                regret = insertion_costs[1][0] - insertion_costs[0][0]
            else:
                regret = insertion_costs[0][0]  # chỉ có 1 option → regret = cost đó

            regrets[cust] = regret
            best_insertions[cust] = (insertion_costs[0][1], insertion_costs[0][2])

        # Chọn customer có regret lớn nhất
        cust = max(remaining, key=lambda c: regrets.get(c, -float("inf")))
        remaining.remove(cust)

        # Chèn customer đó
        route_idx, pos = best_insertions.get(cust, (-1, -1))

        if route_idx == -1:
            # Tạo route mới
            solution.routes.append([0, cust, 0])
        else:
            route = solution.routes[route_idx]
            solution.routes[route_idx] = route[:pos] + [cust] + route[pos:]

    return solution


# Registry cho repair operators
REPAIR_OPERATORS: list[tuple[str, RepairOperator]] = [
    ("greedy", r1_greedy_repair),
    ("criticality", r2_critical_repair),
    ("regret", r3_regret_repair),
]


# =============================================================================
# Section 4: Private helpers
# =============================================================================

def _is_route_feasible(route: list[int], inst: VRPTWInstance) -> bool:
    """
    Kiểm tra 1 route có feasible không (capacity + TW).
    Copy từ solution.py để operators độc lập.
    """
    # Capacity check
    demand = sum(inst.customer_of(node).demand for node in route if node != 0)
    if demand > inst.vehicle_capacity:
        return False

    # TW check
    current_time = 0.0
    for k in range(len(route) - 1):
        i, j = route[k], route[k + 1]
        current_time += inst.normalized_distance(i, j)

        if j != 0:
            tw = inst.customer_of(j).time_window
            svc = inst.customer_of(j).service_time

            if current_time > tw[1]:
                return False

            if current_time < tw[0]:
                current_time = tw[0]

            current_time += svc

    return True
