"""
run.py — Main entry point for PPO-ALNS VRPTW

Usage:
    # Baseline ALNS
    python run.py --instance n20_M_1 --methods ALNS_Greedy --iters 100
    python run.py --batch data/n20 --limit 5 --iters 100

    # PPO Training
    python run.py --train --batch data/n20 --steps 5000 --save checkpoints/ppo_greedy.pt

    # PPO Evaluation (sau khi train xong)
    python run.py --instance n20_M_1 --methods PPO_ALNS_Greedy --checkpoint checkpoints/ppo_greedy.pt

    # Compare all methods
    python run.py --instance n20_M_1 --methods ALNS_Greedy ALNS_CW PPO_ALNS_Greedy --checkpoint checkpoints/ppo_greedy.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.evaluator import (
    evaluate_all,
    evaluate_batch,
    compare_results,
    print_comparison,
    save_result,
    METHOD_CONFIGS,
)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPO-ALNS for VRPTW — Main Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline ALNS
  python run.py --instance n20_M_1 --methods ALNS_Greedy ALNS_CW --iters 100
  python run.py --batch data/n20 --limit 5 --iters 100

  # Train PPO agent
  python run.py --train --batch data/n20 --limit 10 --steps 5000 --save checkpoints/ppo.pt

  # Evaluate with trained agent
  python run.py --instance n20_M_1 --methods PPO_ALNS_Greedy --checkpoint checkpoints/ppo.pt

  # Compare all methods
  python run.py --instance n20_M_1 --methods ALNS_Greedy ALNS_CW PPO_ALNS_Greedy \\
      --checkpoint checkpoints/ppo.pt --iters 100
        """,
    )

    # --- Instance / Batch ---
    parser.add_argument(
        "--instance", "-i", type=str, default=None,
        help="Instance name or path (e.g. n20_M_1)"
    )
    parser.add_argument(
        "--batch", "-b", type=str, default=None,
        help="Batch mode: evaluate on all instances in folder"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max instances per batch (default: all)"
    )

    # --- Methods ---
    parser.add_argument(
        "--methods", "-m", nargs="+",
        choices=list(METHOD_CONFIGS.keys()),
        default=None,
        help="Methods to run (default: all)"
    )

    # --- ALNS config ---
    parser.add_argument(
        "--iters", type=int, default=100,
        help="Max ALNS iterations (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--nosave", action="store_true",
        help="Don't save results to JSON"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Verbose output (default: True)"
    )

    # --- PPO Training ---
    parser.add_argument(
        "--train", action="store_true",
        help="Enter training mode"
    )
    parser.add_argument(
        "--steps", type=int, default=200_000,
        help="Total training steps (default: 200,000)"
    )
    parser.add_argument(
        "--num-envs", type=int, default=16,
        help="Parallel environments for training (default: 16)"
    )
    parser.add_argument(
        "--save", type=str, default="checkpoints/ppo_alns.pt",
        help="Save trained model to this path"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=5000,
        help="Evaluation interval during training (default: 5000)"
    )
    parser.add_argument(
        "--init-method", type=str, default="greedy",
        choices=["greedy", "cw"],
        help="Initial solution method for PPO (default: greedy)"
    )

    # --- PPO Evaluation ---
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained PPO checkpoint for evaluation"
    )

    args = parser.parse_args()

    # ---------- TRAINING MODE ----------
    if args.train:
        _run_training(args)
        return

    # ---------- EVALUATION MODE ----------
    if args.instance:
        _run_instance(args)
    elif args.batch:
        _run_batch(args)
    else:
        parser.print_help()
        print("\nError: Must provide --instance or --batch")
        sys.exit(1)


def _run_training(args: argparse.Namespace) -> None:
    """Run PPO training."""
    from src.trainer import PPOTrainer, TrainerConfig
    from src.ppo_agent import PPOConfig

    batch_dir = Path(args.batch) if args.batch else Path("data/n20")
    if not batch_dir.exists():
        print(f"ERROR: Batch directory not found: {batch_dir}")
        sys.exit(1)

    # Collect instance files
    instance_files = sorted(batch_dir.glob("*.json"))
    if args.limit:
        instance_files = instance_files[:args.limit]

    if not instance_files:
        print(f"ERROR: No instances found in {batch_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"PPO-ALNS Training")
    print(f"  Instances: {len(instance_files)} from {batch_dir}")
    print(f"  Steps: {args.steps:,}")
    print(f"  Parallel envs: {args.num_envs}")
    print(f"  Init method: {args.init_method}")
    print(f"  Output: {args.save}")
    print(f"{'='*60}")

    ppo_config = PPOConfig(
        lr=3e-4,
        gamma=1.0,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        ppo_epochs=10,
        batch_size=128,
        entropy_coef=0.01,
        seed=args.seed,
    )

    trainer_config = TrainerConfig(
        total_steps=args.steps,
        num_envs=args.num_envs,
        eval_interval=args.eval_interval,
        save_interval=args.eval_interval,
        ppo_config=ppo_config,
        train_instances=[str(p) for p in instance_files],
        eval_instances=[str(p) for p in instance_files[:min(5, len(instance_files))]],
        init_method=args.init_method,
        max_iterations=args.iters,
        output_dir=str(Path(args.save).parent),
    )

    trainer = PPOTrainer(trainer_config)
    trainer.train()

    # Save final checkpoint
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    trainer.agent.save(args.save)
    print(f"\nTrained model saved to: {args.save}")


def _run_instance(args: argparse.Namespace) -> None:
    """Run evaluation on a single instance."""
    inst_name = args.instance

    # Resolve path
    if Path(inst_name).exists():
        inst_path = Path(inst_name)
    else:
        for folder in ["data/n20", "data/n50", "data/n100"]:
            candidate = Path(folder) / f"{inst_name}.json"
            if candidate.exists():
                inst_path = candidate
                break
        else:
            inst_path = Path(f"data/{inst_name[:3]}/{inst_name}.json")
            if not inst_path.exists():
                print(f"ERROR: Instance not found: {inst_name}")
                sys.exit(1)

    # Resolve methods
    if args.methods is None:
        method_names = ["ALNS_Greedy", "ALNS_CW"]
    else:
        method_names = args.methods

    # Resolve checkpoint
    checkpoint = args.checkpoint

    # Check if any PPO method requested
    has_ppo = any(
        name.startswith("PPO_") for name in method_names
    )

    if has_ppo and checkpoint is None:
        print(f"ERROR: PPO methods require --checkpoint argument")
        print(f"  Run training first: python run.py --train --batch data/n20 ...")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Running on instance: {inst_path}")
    print(f"Methods: {method_names}")
    print(f"Iterations: {args.iters}")
    print(f"Seed: {args.seed}")
    if checkpoint:
        print(f"Checkpoint: {checkpoint}")
    print(f"{'='*60}")

    # Set checkpoint path for PPO methods
    from src.evaluator import METHOD_CONFIGS
    for name in method_names:
        if name.startswith("PPO_"):
            cfg = METHOD_CONFIGS[name]
            cfg.ppo_model_path = str(Path(checkpoint).resolve())
            cfg.max_iterations = args.iters
            cfg.seed = args.seed
        else:
            METHOD_CONFIGS[name].max_iterations = args.iters
            METHOD_CONFIGS[name].seed = args.seed

    results = evaluate_all(
        instance_path=inst_path,
        method_names=method_names,
        max_iterations=args.iters,
        seed=args.seed,
        verbose=args.verbose,
    )

    if results:
        stats = compare_results(results)
        print_comparison(stats)

        if not args.nosave:
            for r in results:
                save_result(r, overwrite=True)


def _run_batch(args: argparse.Namespace) -> None:
    """Run evaluation on batch of instances."""
    batch_dir = Path(args.batch)
    if not batch_dir.exists():
        print(f"ERROR: Batch directory not found: {batch_dir}")
        sys.exit(1)

    instance_files = sorted(batch_dir.glob("*.json"))
    if args.limit:
        instance_files = instance_files[:args.limit]

    if args.methods is None:
        method_names = ["ALNS_Greedy", "ALNS_CW"]
    else:
        method_names = args.methods

    checkpoint = args.checkpoint

    has_ppo = any(name.startswith("PPO_") for name in method_names)
    if has_ppo and checkpoint is None:
        print(f"ERROR: PPO methods require --checkpoint")
        sys.exit(1)

    # Set checkpoint paths
    from src.evaluator import METHOD_CONFIGS
    for name in method_names:
        if name.startswith("PPO_"):
            METHOD_CONFIGS[name].ppo_model_path = str(Path(checkpoint).resolve())
            METHOD_CONFIGS[name].max_iterations = args.iters
            METHOD_CONFIGS[name].seed = args.seed
        else:
            METHOD_CONFIGS[name].max_iterations = args.iters
            METHOD_CONFIGS[name].seed = args.seed

    print(f"\n{'='*60}")
    print(f"Batch mode: {batch_dir}")
    print(f"Methods: {method_names}")
    print(f"Iterations: {args.iters}")
    print(f"Limit: {args.limit or len(instance_files)} instances")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}")

    all_results = evaluate_batch(
        instance_dir=batch_dir,
        method_names=method_names,
        max_iterations=args.iters,
        max_instances=args.limit,
        seed=args.seed,
        save=not args.nosave,
        verbose=args.verbose,
    )

    if all_results:
        stats = compare_results(all_results)
        print(f"\n{'='*60}")
        print(f"OVERALL SUMMARY ({len(all_results)} evaluations)")
        print(f"{'='*60}")
        print_comparison(stats)


if __name__ == "__main__":
    main()
