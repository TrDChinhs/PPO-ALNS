"""
evaluator.py — Run ALNS methods and save/compare results

ĐỌC SAU alns.py.

Chứa:
    evaluate_method()    — chạy 1 method trên 1 instance
    evaluate_all()      — chạy nhiều methods/instances
    save_result()       — lưu kết quả ra JSON
    load_result()       — load kết quả từ JSON
    compare_results()  — so sánh các methods

Usage:
    from evaluator import evaluate_all, compare_results

    results = evaluate_all(
        instance_path="data/n20/n20_M_1.json",
        methods=["ALNS_Greedy", "ALNS_CW"],
        max_iterations=100,
    )
    comparison = compare_results(results)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

from problem import VRPTWInstance
from alns import ALNS, ALNSConfig, ALNSResult
from solution import make_initial_solution


# =============================================================================
# Section 1: Method Configuration
# =============================================================================

@dataclass
class MethodConfig:
    """Cấu hình cho 1 method để evaluate."""
    name: str
    init_method: Literal["greedy", "cw"] = "greedy"
    use_ppo_destroy: bool = False  # Phase 2
    use_ppo_repair: bool = False  # Phase 2
    use_ppo_accept: bool = False  # Phase 2
    use_ppo_terminate: bool = False  # Phase 2
    ppo_model_path: str | None = None  # checkpoint path
    max_iterations: int = 100
    seed: int = 42


# Supported methods
METHOD_CONFIGS: dict[str, MethodConfig] = {
    "ALNS_Greedy": MethodConfig(name="ALNS_Greedy", init_method="greedy"),
    "ALNS_CW": MethodConfig(name="ALNS_CW", init_method="cw"),
    "PPO_ALNS_Greedy": MethodConfig(
        name="PPO_ALNS_Greedy",
        init_method="greedy",
        use_ppo_destroy=True,
        use_ppo_repair=True,
        use_ppo_accept=True,
        use_ppo_terminate=True,
    ),
    "PPO_ALNS_CW": MethodConfig(
        name="PPO_ALNS_CW",
        init_method="cw",
        use_ppo_destroy=True,
        use_ppo_repair=True,
        use_ppo_accept=True,
        use_ppo_terminate=True,
    ),
}


# =============================================================================
# Section 2: Evaluation
# =============================================================================

@dataclass
class EvaluationResult:
    """Kết quả evaluate 1 method trên 1 instance."""
    instance_name: str
    method: str
    init_method: str
    best_cost: float
    distance: float
    tw_violations: float
    num_vehicles: int
    runtime_seconds: float
    iterations: int
    best_found_at_iter: int
    init_cost: float | None = None
    routes: list[list[int]] | None = None
    cost_history: list[float] | None = None
    # Instance data for visualization
    depot: dict | None = None  # {"x": float, "y": float}
    customers: list[dict] | None = None  # [{id, x, y, demand, tw_start, tw_end, service_time}, ...]


def evaluate_method(
    instance: VRPTWInstance,
    config: MethodConfig,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Chạy 1 method trên 1 instance.

    Args:
        instance: VRPTWInstance
        config: MethodConfig
        verbose: in ra trạng thái

    Returns:
        EvaluationResult
    """
    if verbose:
        print(f"\n  Running {config.name}...")

    # Record initial cost
    init_sol = make_initial_solution(instance, config.init_method, config.seed)
    init_cost = init_sol.calc_cost()

    start = time.time()

    if config.use_ppo_destroy or config.use_ppo_repair or config.use_ppo_accept or config.use_ppo_terminate:
        # PPO-ALNS method
        result = _evaluate_ppo(instance, config)
    else:
        # Baseline ALNS
        alns_config = ALNSConfig(
            init_method=config.init_method,
            max_iterations=config.max_iterations,
            seed=config.seed,
            verbose=False,
        )
        alns = ALNS(instance, alns_config)
        result = alns.run()

    runtime = time.time() - start

    # Extract instance data for visualization
    depot_data, customers_data = _instance_to_dict(instance)

    # Build result
    eval_result = EvaluationResult(
        instance_name=instance.name,
        method=config.name,
        init_method=config.init_method,
        best_cost=round(result.best_cost, 4),
        distance=round(result.best_cost_breakdown[0], 4),
        tw_violations=round(result.best_cost_breakdown[1], 4),
        num_vehicles=result.best_cost_breakdown[2],
        runtime_seconds=round(runtime, 2),
        iterations=result.iterations,
        best_found_at_iter=result.best_found_at_iter,
        init_cost=round(init_cost, 4),
        routes=result.best_solution.routes,
        cost_history=result.cost_history,
        depot=depot_data,
        customers=customers_data,
    )

    if verbose:
        bd = result.best_cost_breakdown
        print(
            f"  {config.name}: cost={result.best_cost:.4f} "
            f"(d={bd[0]:.4f}, tw={bd[1]:.4f}, k={bd[2]}) "
            f"time={runtime:.2f}s"
        )

    return eval_result


def _instance_to_dict(instance: VRPTWInstance) -> tuple[dict, list[dict]]:
    """
    Extract depot + customers data from instance for visualization.

    Returns:
        (depot_dict, customers_list)
        depot_dict: {"x": float, "y": float}
        customers_list: [{"id": int, "x": float, "y": float, "demand": int,
                          "tw_start": float, "tw_end": float, "service_time": float}, ...]
    """
    depot_dict = {
        "x": instance.depot.coord.x,
        "y": instance.depot.coord.y,
    }
    customers_list = []
    for c in instance.customers:
        customers_list.append({
            "id": c.id,
            "x": c.coord.x,
            "y": c.coord.y,
            "demand": c.demand,
            "tw_start": c.time_window[0],
            "tw_end": c.time_window[1],
            "service_time": c.service_time,
        })
    return depot_dict, customers_list


def _evaluate_ppo(
    instance: VRPTWInstance,
    config: MethodConfig,
) -> "ALNSResult":
    """
    Run PPO-ALNS on a single instance using a trained agent.

    Args:
        instance: VRPTWInstance
        config: MethodConfig with use_ppo_* flags

    Returns:
        ALNSResult
    """
    from env import ALNSEnv
    from state_encoder import StateEncoder
    from ppo_agent import PPOAgent
    from alns import ALNSResult as PPOALNSResult
    from solution import Solution

    # Load agent
    if config.ppo_model_path is None:
        raise ValueError(f"PPO model path not set for {config.name}")

    encoder = StateEncoder(instance)
    agent = PPOAgent(state_dim=encoder.state_dim)
    agent.load(config.ppo_model_path)

    # Create env
    env = ALNSEnv(
        instance=instance,
        init_method=config.init_method,
        max_iterations=config.max_iterations,
        seed=config.seed,
    )
    state = env.reset()
    done = False

    while not done:
        (d_a, r_a, acc_a, term_a), _, _ = agent.get_action(state, deterministic=True)
        action = (d_a, r_a, acc_a, term_a)
        state, _, done, info = env.step(action)

    result = env.get_result()

    # Convert to ALNSResult format
    bd = result["best_cost_breakdown"]
    return PPOALNSResult(
        best_solution=result["best_solution"],
        best_cost=result["best_cost"],
        best_cost_breakdown=(bd.total_distance, bd.tw_violations, bd.num_vehicles),
        iterations=result["iteration"],
        best_found_at_iter=result["iteration"],
        runtime_seconds=0.0,
    )


def evaluate_all(
    instance_path: str | Path | None = None,
    instance: VRPTWInstance | None = None,
    method_names: list[str] | None = None,
    max_iterations: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> list[EvaluationResult]:
    """
    Chạy tất cả methods trên 1 instance.

    Args:
        instance_path: đường dẫn đến file JSON instance
        instance: hoặc VRPTWInstance trực tiếp
        method_names: list method names (default: all supported)
        max_iterations: số iterations cho ALNS
        seed: random seed
        verbose: in ra trạng thái

    Returns:
        List of EvaluationResult
    """
    # Load instance
    if instance is not None:
        inst = instance
    elif instance_path is not None:
        inst = VRPTWInstance.from_json(instance_path)
    else:
        raise ValueError("Must provide either instance_path or instance")

    # Determine methods
    if method_names is None:
        method_names = list(METHOD_CONFIGS.keys())

    if verbose:
        print(f"\n{'='*60}")
        print(f"Instance: {inst.name}  n={inst.n_customers}  diff={inst.difficulty}")
        print(f"Methods: {method_names}")
        print(f"{'='*60}")

    results: list[EvaluationResult] = []

    for method_name in method_names:
        if method_name not in METHOD_CONFIGS:
            print(f"  WARNING: Unknown method '{method_name}', skipping.")
            continue

        cfg = METHOD_CONFIGS[method_name]
        cfg.max_iterations = max_iterations
        cfg.seed = seed

        try:
            result = evaluate_method(inst, cfg, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"  ERROR in {method_name}: {e}")
            continue

    return results


# =============================================================================
# Section 3: Save / Load Results
# =============================================================================

def save_result(
    result: EvaluationResult,
    output_dir: str | Path = "results",
    overwrite: bool = False,
) -> Path:
    """
    Lưu EvaluationResult ra JSON file.

    Args:
        result: EvaluationResult
        output_dir: thư mục lưu
        overwrite: ghi đè nếu đã tồn tại

    Returns:
        Đường dẫn file đã lưu
    """
    output_dir = Path(output_dir)

    # Determine subfolder
    instance_name = result.instance_name
    n_match = instance_name.split("_")[0]  # e.g. "n20"
    subfolder = output_dir / n_match
    subfolder.mkdir(parents=True, exist_ok=True)

    # Filename: instance__method__date.json
    date_str = "2026-04-17"  # TODO: use actual date
    filename = f"{instance_name}__{result.method}__{date_str}.json"
    filepath = subfolder / filename

    if filepath.exists() and not overwrite:
        print(f"  File exists: {filepath} (skipping, use overwrite=True)")
        return filepath

    # Build dict
    data = {
        "instance": result.instance_name,
        "method": result.method,
        "init_method": result.init_method,
        "results": {
            "best_cost": result.best_cost,
            "distance": result.distance,
            "num_vehicles": result.num_vehicles,
            "tw_violations": result.tw_violations,
            "runtime_seconds": result.runtime_seconds,
            "iterations": result.iterations,
            "best_found_at_iter": result.best_found_at_iter,
            "init_cost": result.init_cost,
        },
        "routes": result.routes,
        "depot": result.depot,
        "customers": result.customers,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"  Saved: {filepath}")
    return filepath


def load_result(filepath: str | Path) -> EvaluationResult:
    """Load EvaluationResult từ JSON file."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    return EvaluationResult(
        instance_name=data["instance"],
        method=data["method"],
        init_method=data["init_method"],
        best_cost=data["results"]["best_cost"],
        distance=data["results"]["distance"],
        num_vehicles=data["results"]["num_vehicles"],
        tw_violations=data["results"]["tw_violations"],
        runtime_seconds=data["results"]["runtime_seconds"],
        iterations=data["results"]["iterations"],
        best_found_at_iter=data["results"]["best_found_at_iter"],
        init_cost=data["results"].get("init_cost"),
        routes=data.get("routes"),
    )


# =============================================================================
# Section 4: Comparison
# =============================================================================

@dataclass
class MethodStats:
    """Statistics cho 1 method across instances."""
    method: str
    avg_cost: float
    min_cost: float
    max_cost: float
    avg_distance: float
    avg_vehicles: float
    avg_tardies: float
    avg_runtime: float
    win_count: int = 0
    instances_run: int = 0


def compare_results(
    results: list[EvaluationResult],
) -> dict[str, MethodStats]:
    """
    So sánh các methods từ evaluation results.

    Args:
        results: list of EvaluationResult (từ evaluate_all)

    Returns:
        Dict[method_name -> MethodStats]
    """
    if not results:
        return {}

    # Group by method
    by_method: dict[str, list[EvaluationResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    stats: dict[str, MethodStats] = {}

    for method, method_results in by_method.items():
        costs = [r.best_cost for r in method_results]
        distances = [r.distance for r in method_results]
        vehicles = [r.num_vehicles for r in method_results]
        tardies = [r.tw_violations for r in method_results]
        runtimes = [r.runtime_seconds for r in method_results]

        stats[method] = MethodStats(
            method=method,
            avg_cost=round(sum(costs) / len(costs), 4),
            min_cost=min(costs),
            max_cost=max(costs),
            avg_distance=round(sum(distances) / len(distances), 4),
            avg_vehicles=round(sum(vehicles) / len(vehicles), 2),
            avg_tardies=round(sum(tardies) / len(tardies), 4),
            avg_runtime=round(sum(runtimes) / len(runtimes), 2),
            instances_run=len(method_results),
        )

    # Win count
    if len(by_method) > 1:
        # Group by instance
        by_instance: dict[str, list[EvaluationResult]] = {}
        for r in results:
            by_instance.setdefault(r.instance_name, []).append(r)

        for instance_results in by_instance.values():
            best = min(instance_results, key=lambda r: r.best_cost)
            stats[best.method].win_count += 1

    return stats


def print_comparison(stats: dict[str, MethodStats]) -> None:
    """In comparison table ra console."""
    if not stats:
        print("No stats to display.")
        return

    methods = sorted(stats.keys())
    print(f"\n{'='*85}")
    print(f"{'Method':<15} {'Avg Cost':>10} {'Avg Dist':>10} {'Avg Vehicles':>13} "
          f"{'Avg Tardies':>12} {'Avg Time':>8} {'Wins':>5}")
    print(f"{'-'*85}")
    for m in methods:
        s = stats[m]
        print(
            f"{m:<15} {s.avg_cost:>10.4f} {s.avg_distance:>10.4f} "
            f"{s.avg_vehicles:>13.1f} {s.avg_tardies:>12.4f} "
            f"{s.avg_runtime:>7.2f}s {s.win_count:>5d}"
        )
    print(f"{'='*85}")


# =============================================================================
# Section 5: Batch Evaluation on Multiple Instances
# =============================================================================

def evaluate_batch(
    instance_dir: str | Path = "data/n20",
    method_names: list[str] | None = None,
    max_iterations: int = 100,
    max_instances: int | None = None,
    seed: int = 42,
    save: bool = True,
    verbose: bool = True,
) -> list[EvaluationResult]:
    """
    Chạy tất cả methods trên tất cả instances trong 1 folder.

    Args:
        instance_dir: folder chứa instance JSON files
        method_names: methods cần chạy
        max_iterations: ALNS iterations
        max_instances: giới hạn số instances (None = tất cả)
        seed: random seed
        save: lưu kết quả ra JSON
        verbose: in ra trạng thái

    Returns:
        List of all EvaluationResult
    """
    instance_dir = Path(instance_dir)
    instance_files = sorted(instance_dir.glob("*.json"))

    if max_instances:
        instance_files = instance_files[:max_instances]

    if verbose:
        print(f"\nBatch evaluation: {len(instance_files)} instances from {instance_dir}")
        print(f"Methods: {method_names}")

    all_results: list[EvaluationResult] = []

    for i, inst_file in enumerate(instance_files, 1):
        if verbose:
            print(f"\n[{i}/{len(instance_files)}] {inst_file.name}")

        try:
            results = evaluate_all(
                instance_path=inst_file,
                method_names=method_names,
                max_iterations=max_iterations,
                seed=seed,
                verbose=verbose,
            )
            all_results.extend(results)

            if save:
                for r in results:
                    save_result(r)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Summary
    if all_results:
        stats = compare_results(all_results)
        print_comparison(stats)

    return all_results


# =============================================================================
# Section 6: Quick Test
# =============================================================================

if __name__ == "__main__":
    # Quick test on 1 instance
    results = evaluate_all(
        instance_path="data/n20/n20_M_1.json",
        method_names=["ALNS_Greedy", "ALNS_CW"],
        max_iterations=50,
        seed=42,
        verbose=True,
    )

    if results:
        stats = compare_results(results)
        print_comparison(stats)
