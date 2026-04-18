"""
generate_data.py — Generate VRPTW random instances

Tạo 90 instances: 3 scales (20/50/100) × 3 difficulties (S/M/L) × 10 instances.
Lưu vào folder data/{n20,n50,n100}/.

Usage:
    python generate_data.py
    python generate_data.py --n 20 --difficulty M --count 5
    python generate_data.py --regenerate  # xóa và tạo lại tất cả

References:
    SPEC.md Section 3.4 & 3.5
    src/problem.py (VRPTWInstance, InstanceConfig, GeneratedInstance)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.problem import (
    InstanceConfig,
    GeneratedInstance,
)


# =============================================================================
# Generator
# =============================================================================

def generate_instance(
    config: InstanceConfig,
    instance_name: str | None = None,
) -> GeneratedInstance:
    """
    Generate một VRPTW instance hoàn toàn ngẫu nhiên.

    Quy trình:
        1. Sinh N customer coords từ uniform [0,1] × [0,1]
        2. Depot = geometric center của tất cả customers
        3. Sinh demand [1, 16], service_time [0.05, 0.1]
        4. Sinh time windows dựa trên difficulty

    Args:
        config: InstanceConfig chứa N, difficulty, seed
        instance_name: tên instance override (default: n{N}_{D}_{idx})

    Returns:
        GeneratedInstance (JSON-serializable)
    """
    rng = random.Random(config.seed)

    n = config.n_customers

    # --- 1. Customer coordinates ---
    customer_coords = [(rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)) for _ in range(n)]

    # --- 2. Depot = geometric center ---
    cx = sum(x for x, y in customer_coords) / n
    cy = sum(y for x, y in customer_coords) / n

    # --- 3. Demands, service times ---
    demands = [rng.randint(config.demand_min, config.demand_max) for _ in range(n)]
    service_times = [rng.uniform(config.service_time_min, config.service_time_max) for _ in range(n)]

    # --- 4. Time windows ---
    # Với mỗi customer: sinh TW width từ range của difficulty,
    # rồi sinh start = uniform [0, T_max - width].
    # Paper dùng approach tương tự (random TW widths trong range).
    tw_width_min, tw_width_max = config.tw_width_range()
    max_t = config.max_travel_time

    time_windows: list[tuple[float, float]] = []
    for _ in range(n):
        tw_width = rng.uniform(tw_width_min, tw_width_max)
        # TW start: uniform từ 0 đến (max_t - tw_width)
        tw_start = rng.uniform(0.0, max(0.0, max_t - tw_width))
        tw_end = tw_start + tw_width
        time_windows.append((round(tw_start, 4), round(tw_end, 4)))

    # --- 5. Build customer list ---
    depot_x = round(cx, 4)
    depot_y = round(cy, 4)

    customers_list: list[dict] = []
    for i in range(n):
        cx_i, cy_i = customer_coords[i]
        customers_list.append({
            "id": i + 1,
            "x": round(cx_i, 4),
            "y": round(cy_i, 4),
            "demand": demands[i],
            "time_window": list(time_windows[i]),
            "service_time": round(service_times[i], 4),
        })

    # Tên instance: dùng instance_name override hoặc build từ config
    if instance_name is None:
        # instance_name sẽ được set bởi caller (n{N}_{D}_{idx})
        instance_name = f"n{n}_{config.difficulty}_x"

    return GeneratedInstance(
        name=instance_name,
        n_customers=n,
        difficulty=config.difficulty,
        depot={"x": depot_x, "y": depot_y},
        customers=customers_list,
        vehicle_capacity=config.vehicle_capacity,
        max_travel_time=config.max_travel_time,
    )


# =============================================================================
# Batch generation
# =============================================================================

def generate_all_instances(
    base_output_dir: Path | str = "data",
    scales: list[int] | None = None,
    difficulties: list[str] | None = None,
    instances_per_group: int = 10,
    base_seed: int = 42,
) -> None:
    """
    Generate tất cả instances theo cấu hình SPEC.

    Tạo: scales × difficulties × instances_per_group instances
    = 3 × 3 × 10 = 90 instances.

    Folder structure:
        data/n20/n20_S_1.json, n20_S_2.json, ...
        data/n50/n50_M_1.json, ...
        data/n100/n100_L_10.json, ...

    Args:
        base_output_dir: thư mục gốc chứa data
        scales: list số customers (default [20, 50, 100])
        difficulties: list difficulties (default ["S", "M", "L"])
        instances_per_group: số instances mỗi group (default 10)
        base_seed: seed bắt đầu (mỗi instance dùng seed khác nhau)
    """
    if scales is None:
        scales = [20, 50, 100]
    if difficulties is None:
        difficulties = ["S", "M", "L"]

    base_output_dir = Path(base_output_dir)
    total = 0

    for n in scales:
        n_dir = base_output_dir / f"n{n}"
        n_dir.mkdir(parents=True, exist_ok=True)

        for diff in difficulties:
            for idx in range(1, instances_per_group + 1):
                seed = base_seed + (n * 10000) + (["S", "M", "L"].index(diff) * 1000) + idx
                config = InstanceConfig(
                    n_customers=n,
                    difficulty=diff,
                    seed=seed,
                )

                instance_name = f"n{n}_{diff}_{idx}"
                inst = generate_instance(config, instance_name=instance_name)
                filename = f"{inst.name}.json"
                filepath = n_dir / filename

                # Kiểm tra xem đã tồn tại chưa (trừ khi regenerate)
                if filepath.exists():
                    print(f"  SKIP (exists): {filepath}")
                    continue

                inst.to_json_file(filepath)
                print(f"  Created: {filepath}")
                total += 1

    print(f"\nDone. Generated {total} instances.")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate VRPTW random instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_data.py                          # Generate all 90 instances
  python generate_data.py --n 20 --difficulty M     # Chỉ n=20, M
  python generate_data.py --regenerate             # Xóa và tạo lại tất cả
  python generate_data.py --list                   # Liệt kê instances hiện có
        """,
    )
    parser.add_argument("--n", type=int, choices=[20, 50, 100],
                        help="Number of customers (default: all)")
    parser.add_argument("--difficulty", type=str, choices=["S", "M", "L"],
                        help="Difficulty level (default: all)")
    parser.add_argument("--count", type=int, default=10,
                        help="Instances per group (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate all instances (delete existing first)")
    parser.add_argument("--list", action="store_true",
                        help="List existing instances and exit")
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory (default: data/)")

    args = parser.parse_args()

    # --- List mode ---
    if args.list:
        base = Path(args.output)
        for n_dir in sorted(base.iterdir()):
            if n_dir.is_dir():
                instances = sorted(n_dir.glob("*.json"))
                print(f"{n_dir.name}/: {len(instances)} instances")
        return

    # --- Regenerate mode ---
    if args.regenerate:
        confirm = input(f"Delete all existing instances in '{args.output}/' and regenerate? (y/N): ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return
        # Xóa các folder n20, n50, n100
        for n in [20, 50, 100]:
            n_dir = Path(args.output) / f"n{n}"
            if n_dir.exists():
                for f in n_dir.glob("*.json"):
                    f.unlink()
                print(f"  Cleared: {n_dir}")

    # --- Determine scales and difficulties ---
    scales = [args.n] if args.n else [20, 50, 100]
    difficulties = [args.difficulty] if args.difficulty else ["S", "M", "L"]

    print(f"Generating instances: scales={scales}, difficulties={difficulties}, "
          f"count={args.count}, base_seed={args.seed}")
    print(f"Output: {args.output}/")
    print()

    total = 0
    base_output = Path(args.output)

    for n in scales:
        n_dir = base_output / f"n{n}"
        n_dir.mkdir(parents=True, exist_ok=True)

        for diff in difficulties:
            for idx in range(1, args.count + 1):
                seed = args.seed + (n * 10000) + (["S", "M", "L"].index(diff) * 1000) + idx
                config = InstanceConfig(
                    n_customers=n,
                    difficulty=diff,
                    seed=seed,
                )
                instance_name = f"n{n}_{diff}_{idx}"
                inst = generate_instance(config, instance_name=instance_name)
                filepath = n_dir / f"{inst.name}.json"
                inst.to_json_file(filepath)
                print(f"  {filepath}")
                total += 1

    print(f"\nDone. Generated {total} instances.")


if __name__ == "__main__":
    main()
