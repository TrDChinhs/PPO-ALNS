"""
problem.py — VRPTW data structures & validation

ĐỌC TRƯỚC KHI CODE BẤT KỲ FILE NÀO KHÁC.
Chứa định nghĩa VRPTW instance: Depot, Customer, VRPTWInstance.
Cách load từ JSON, tính distance, kiểm tra feasibility.

References:
    SPEC.md Section 3
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple


# =============================================================================
# Section 1: Core data structures
# =============================================================================

@dataclass(frozen=True)
class Coord:
    """2D coordinate, immutable."""
    x: float
    y: float


@dataclass(frozen=True)
class Customer:
    """
    Một customer trong VRPTW.

    - id: unique identifier (1..N)
    - coord: vị trí (x, y)
    - demand: nhu cầu (1-16)
    - time_window: [earliest_arrival, latest_arrival]
    - service_time: thời gian phục vụ khi có mặt
    """
    id: int
    coord: Coord
    demand: int
    time_window: tuple[float, float]
    service_time: float


@dataclass(frozen=True)
class Depot:
    """Depot — điểm xuất phát/kết thúc của mọi routes."""
    coord: Coord


class TimeWindow(NamedTuple):
    """[earliest, latest] — thời gian cho phép xe đến customer."""
    earliest: float
    latest: float


# =============================================================================
# Section 2: VRPTWInstance — đại diện cho 1 bài toán
# =============================================================================

@dataclass
class VRPTWInstance:
    """
    Một instance VRPTW hoàn chỉnh.

    Attributes:
        name: tên instance (e.g. "n20_M_1")
        n_customers: số lượng customers
        difficulty: S (tight) / M (medium) / L (loose)
        depot: tọa độ depot (luôn là geometric center)
        customers: danh sách customers (đánh index 1..N)
        vehicle_capacity: sức chứa xe (default 64)
        max_travel_time: thời gian travel tối đa (default 1.0)
        _dist_cache: lazy cache cho distance matrix

    Usage:
        inst = VRPTWInstance.from_json(Path("data/n20/n20_M_1.json"))
        dist = inst.distance_matrix()  # lazy compute
        coord = inst.coord_of(0)        # depot
        coord = inst.coord_of(3)        # customer id=3
    """
    name: str
    n_customers: int
    difficulty: str  # "S" | "M" | "L"
    depot: Depot
    customers: tuple[Customer, ...]
    vehicle_capacity: int = 64
    max_travel_time: float = 1.0

    # ---------- factory ----------

    @classmethod
    def from_json(cls, path: str | Path) -> VRPTWInstance:
        """
        Load instance từ file JSON.

        Args:
            path: đường dẫn đến file JSON

        Returns:
            VRPTWInstance đã được parse

        Raises:
            FileNotFoundError: file không tồn tại
            ValueError: JSON không hợp lệ hoặc thiếu required fields
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Instance file not found: {path}")

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        # --- parse depot ---
        d = raw["depot"]
        depot = Depot(coord=Coord(x=d["x"], y=d["y"]))

        # --- parse customers ---
        customers_list: list[Customer] = []
        for c in raw["customers"]:
            customers_list.append(
                Customer(
                    id=c["id"],
                    coord=Coord(x=c["x"], y=c["y"]),
                    demand=c["demand"],
                    time_window=(c["time_window"][0], c["time_window"][1]),
                    service_time=c["service_time"],
                )
            )
        customers = tuple(customers_list)

        return cls(
            name=raw["name"],
            n_customers=raw["n_customers"],
            difficulty=raw["difficulty"],
            depot=depot,
            customers=customers,
            vehicle_capacity=raw.get("vehicle_capacity", 64),
            max_travel_time=raw.get("max_travel_time", 1.0),
        )

    # ---------- helpers ----------

    def all_coords(self) -> list[Coord]:
        """Tất cả coords: depot (index 0) + customers (index 1..N)."""
        coords = [self.depot.coord]
        coords.extend(c.coord for c in self.customers)
        return coords

    def coord_of(self, node_idx: int) -> Coord:
        """
        Lấy coord của node theo index.
        Index 0 = depot, 1..N = customers.
        """
        if node_idx == 0:
            return self.depot.coord
        return self.customers[node_idx - 1].coord

    def customer_of(self, node_idx: int) -> Customer:
        """Lấy Customer object từ node index (1..N)."""
        if not (1 <= node_idx <= self.n_customers):
            raise ValueError(f"Customer index must be 1..{self.n_customers}, got {node_idx}")
        return self.customers[node_idx - 1]

    def distance(self, i: int, j: int) -> float:
        """
        Euclidean distance giữa 2 nodes (CHƯA chia 50).
        Để normalized: d(i,j) = distance(i,j) / 50.

        Args:
            i: node index (0=depot, 1..N=customers)
            j: node index (0=depot, 1..N=customers)
        """
        ci = self.coord_of(i)
        cj = self.coord_of(j)
        return math.sqrt((ci.x - cj.x) ** 2 + (ci.y - cj.y) ** 2)

    def normalized_distance(self, i: int, j: int) -> float:
        """Euclidean distance normalized = / 50."""
        return self.distance(i, j) / 50.0

    def distance_matrix(self) -> list[list[float]]:
        """
        Full distance matrix (CHƯA normalized, tức là giá trị thực / 50).
        Returns list-of-list với index 0..N (0=depot).
        Lazy computed — chỉ tính 1 lần.
        """
        n = self.n_customers
        matrix: list[list[float]] = [
            [0.0 for _ in range(n + 1)] for _ in range(n + 1)
        ]
        for i in range(n + 1):
            for j in range(n + 1):
                matrix[i][j] = self.normalized_distance(i, j)
        return matrix

    def travel_time_matrix(self) -> list[list[float]]:
        """
        Travel time matrix = distance matrix (vì travel time = normalized distance).
        Returns list-of-list với index 0..N.
        """
        return self.distance_matrix()

    def demands(self) -> list[int]:
        """Danh sách demand theo thứ tự customer index 1..N."""
        return [c.demand for c in self.customers]

    def time_windows(self) -> list[tuple[float, float]]:
        """Danh sách time windows [earliest, latest] theo thứ tự customer index 1..N."""
        return [c.time_window for c in self.customers]

    def service_times(self) -> list[float]:
        """Danh sách service time theo thứ tự customer index 1..N."""
        return [c.service_time for c in self.customers]

    def __repr__(self) -> str:
        return (
            f"VRPTWInstance(name={self.name}, n={self.n_customers}, "
            f"difficulty={self.difficulty}, capacity={self.vehicle_capacity})"
        )


# =============================================================================
# Section 3: Instance generation helpers
# =============================================================================

@dataclass
class InstanceConfig:
    """
    Cấu hình generation cho VRPTW instance.
    Đọc từ SPEC.md Section 3.4 & 3.5.
    """
    n_customers: int                          # 20, 50, 100
    difficulty: str                           # "S" | "M" | "L"
    seed: int = 42

    # Generation params (theo paper)
    demand_min: int = 1
    demand_max: int = 16
    service_time_min: float = 0.05
    service_time_max: float = 0.10
    max_travel_time: float = 1.0
    vehicle_capacity: int = 64

    # TW width ranges (time windows are generated as [start, start + width])
    TW_WIDTHS: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "S": (0.06, 0.16),   # Small/tight    (~10% of T_max=1.0)
        "M": (0.10, 0.30),  # Medium         (~20% of T_max=1.0)
        "L": (0.30, 0.50),  # Large/loose    (~40% of T_max=1.0)
    })

    def tw_width_range(self) -> tuple[float, float]:
        """Trả về [min, max] TW width cho difficulty hiện tại."""
        return self.TW_WIDTHS[self.difficulty]


@dataclass
class GeneratedInstance:
    """JSON-serializable format của 1 generated instance."""
    name: str
    n_customers: int
    difficulty: str
    depot: dict
    customers: list[dict]
    vehicle_capacity: int
    max_travel_time: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_customers": self.n_customers,
            "difficulty": self.difficulty,
            "depot": self.depot,
            "customers": self.customers,
            "vehicle_capacity": self.vehicle_capacity,
            "max_travel_time": self.max_travel_time,
        }

    def to_json_file(self, path: str | Path) -> None:
        """Ghi ra file JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
