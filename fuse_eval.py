"""
FUSE Evaluation — Benchmark dense vs sparse on real tasks
==========================================================
Runs standard LLM benchmarks (GSM8K, MMLU, HumanEval, etc.)
on both the dense model and the FUSE-patched sparse model,
then reports the comparison.

Uses EleutherAI's lm-evaluation-harness under the hood.
FUSE patching is applied by monkey-patching the model's MLP
forward passes before handing it to the harness.

Usage:
  # Quick GSM8K comparison (the money benchmark)
  python fuse_eval.py --schedule schedule_deepseek7b.json --tasks gsm8k

  # Full eval suite
  python fuse_eval.py --schedule schedule_deepseek7b.json \
      --tasks gsm8k,mmlu,hellaswag

  # Flat sparsity (no schedule)
  python fuse_eval.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --target-sparsity 0.5 --tasks gsm8k

  # Dense-only baseline (for comparison)
  python fuse_eval.py --schedule schedule_deepseek7b.json \
      --tasks gsm8k --dense-only

  # Sparse-only (skip dense, useful if you already have the baseline)
  python fuse_eval.py --schedule schedule_deepseek7b.json \
      --tasks gsm8k --sparse-only

Install:
  pip install "lm-eval[hf]" torch transformers accelerate

Note:
  GSM8K has 1,319 test examples. On a MacBook M-series with a 7B model,
  expect ~2-4 hours for each run (dense + sparse). The sparse run is
  slower in this RAM-resident implementation because the token-by-token
  loop adds Python overhead — but the results measure *quality*, not
  speed. Speed gains come from the production disk-streaming engine.
"""

import argparse
import json
import time
import sys
import os
import torch

# ── Import FUSE components ──────────────────────────────────
# These are imported from the same directory
from fuse_inference import FUSELayer, patch_model_with_fuse


def load_model_and_tokenizer(model_name, dtype="float16", device="auto", token=None):
    """Load model + tokenizer with proper device/dtype handling."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    print(f"  Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=token, trust_remote_code=True
    )

    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=token, dtype=dtype_map[dtype],
        device_map=device if device in ("cpu", "mps") else "auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    return model, tokenizer, device


def run_eval(model, tokenizer, tasks, device, batch_size=1,
             num_fewshot=None, limit=None):
    """
    Run lm-evaluation-harness on a (possibly FUSE-patched) model.

    Returns a dict of {task_name: {metric: value}}.
    """
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    # Wrap the pre-loaded model in lm_eval's HFLM interface.
    # This avoids lm_eval re-loading the model from scratch.
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=str(device),
    )

    results = simple_evaluate(
        model=lm,
        tasks=tasks.split(","),
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
    )

    return results


def extract_scores(results):
    """Pull out the key metric per task from lm_eval results."""
    scores = {}
    if "results" not in results:
        return scores

    for task_name, task_results in results["results"].items():
        # lm_eval stores metrics with different keys per task
        # Common patterns: acc, acc_norm, exact_match, em
        for key in ["exact_match,strict-match", "exact_match,flexible-extract",
                     "acc_norm,none", "acc,none", "em,none",
                     "exact_match,none"]:
            if key in task_results:
                val = task_results[key]
                metric_name = key.split(",")[0]
                scores[task_name] = {
                    "metric": metric_name,
                    "value": val,
                    "value_pct": f"{val * 100:.1f}%",
                }
                break

        # Fallback: grab the first numeric metric
        if task_name not in scores:
            for key, val in task_results.items():
                if isinstance(val, (int, float)) and not key.startswith("_"):
                    scores[task_name] = {
                        "metric": key,
                        "value": val,
                        "value_pct": f"{val * 100:.1f}%" if val <= 1 else str(val),
                    }
                    break

    return scores


def print_comparison(dense_scores, sparse_scores, schedule_info=None):
    """Pretty-print the dense vs sparse comparison table."""
    all_tasks = sorted(set(list(dense_scores.keys()) + list(sparse_scores.keys())))

    print(f"\n  ╔═══════════════════════════════════════════════════════════════╗")
    print(f"  ║                FUSE Evaluation Results                       ║")
    print(f"  ╚═══════════════════════════════════════════════════════════════╝")

    if schedule_info:
        print(f"  Model:    {schedule_info.get('model_name', 'unknown')}")
        print(f"  Sparsity: {schedule_info.get('overall_avg_sparsity', 0):.1%} "
              f"(per-layer adaptive)")
        print(f"  Floor:    cos >= {schedule_info.get('quality_floor', '?')}")
    print()

    header = f"  {'Task':<20} {'Metric':<16} {'Dense':>10} {'FUSE Sparse':>14} {'Retained':>10}"
    print(header)
    print("  " + "─" * 72)

    for task in all_tasks:
        d = dense_scores.get(task, {})
        s = sparse_scores.get(task, {})

        d_val = d.get("value", None)
        s_val = s.get("value", None)
        metric = d.get("metric", s.get("metric", "?"))

        d_str = d.get("value_pct", "—")
        s_str = s.get("value_pct", "—")

        if d_val is not None and s_val is not None and d_val > 0:
            retained = s_val / d_val
            ret_str = f"{retained:.1%}"
            # Color code: green if >95%, yellow if >90%, red otherwise
            if retained >= 0.95:
                status = "✓"
            elif retained >= 0.90:
                status = "~"
            else:
                status = "✗"
            ret_str = f"{ret_str} {status}"
        else:
            ret_str = "—"

        print(f"  {task:<20} {metric:<16} {d_str:>10} {s_str:>14} {ret_str:>10}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="FUSE Evaluation — benchmark dense vs sparse on real tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick GSM8K eval on DeepSeek with calibrated schedule
  python fuse_eval.py --schedule schedule_deepseek7b.json --tasks gsm8k

  # Fast sanity check (only 50 examples)
  python fuse_eval.py --schedule schedule_deepseek7b.json \\
      --tasks gsm8k --limit 50

  # Multiple benchmarks
  python fuse_eval.py --schedule schedule_deepseek7b.json \\
      --tasks gsm8k,mmlu,hellaswag

  # Flat sparsity without a schedule
  python fuse_eval.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
      --target-sparsity 0.4 --tasks gsm8k

  # Dense baseline only
  python fuse_eval.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
      --tasks gsm8k --dense-only

  # Just run sparse (you already have dense numbers)
  python fuse_eval.py --schedule schedule_deepseek7b.json \\
      --tasks gsm8k --sparse-only --dense-baseline '{"gsm8k": 78.2}'
        """,
    )

    # Model selection
    parser.add_argument("--model", default=None,
                        help="Model name (auto-detected from schedule if not set)")
    parser.add_argument("--schedule", default=None,
                        help="FUSE calibration schedule JSON")
    parser.add_argument("--target-sparsity", type=float, default=0.5,
                        help="Flat sparsity (used when no schedule provided)")
    parser.add_argument("--relufied", action="store_true")

    # Eval configuration
    parser.add_argument("--tasks", default="gsm8k",
                        help="Comma-separated benchmark names (gsm8k, mmlu, hellaswag, etc.)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples per task (for fast testing)")
    parser.add_argument("--num-fewshot", type=int, default=None,
                        help="Override number of few-shot examples")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for evaluation")

    # Run modes
    parser.add_argument("--dense-only", action="store_true",
                        help="Only run dense baseline (skip sparse)")
    parser.add_argument("--sparse-only", action="store_true",
                        help="Only run FUSE sparse (skip dense)")
    parser.add_argument("--dense-baseline", type=str, default=None,
                        help='Pre-computed dense scores as JSON string, '
                             'e.g. \'{"gsm8k": 78.2}\'')

    # Hardware
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--token", default=None)

    # Output
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║       FUSE Evaluation: Dense vs Sparse Benchmarking         ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    # ── Resolve model and schedule ──
    schedule_data = None
    per_layer_sparsity = None

    if args.schedule:
        with open(args.schedule) as f:
            schedule_data = json.load(f)
        per_layer_sparsity = schedule_data["per_layer_sparsity"]
        if args.model is None:
            args.model = schedule_data["model_name"]
        if schedule_data.get("relufied") and not args.relufied:
            args.relufied = True

    if args.model is None:
        args.model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # ── Load model ──
    model, tokenizer, device = load_model_and_tokenizer(
        args.model, dtype=args.dtype, device=args.device, token=args.token
    )

    if args.relufied:
        print("  ReLUfication: ON")
        for layer in model.model.layers:
            layer.mlp.act_fn = torch.nn.ReLU()

    # ── Dense evaluation ──
    dense_scores = {}

    if not args.sparse_only:
        print(f"\n  ═══ Dense evaluation ({args.tasks}) ═══")
        if args.limit:
            print(f"  Limit: {args.limit} examples per task")
        print(f"  This may take a while...\n")

        t0 = time.perf_counter()
        dense_results = run_eval(
            model, tokenizer, args.tasks, device,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )
        t_dense = time.perf_counter() - t0

        dense_scores = extract_scores(dense_results)
        print(f"\n  Dense eval completed in {t_dense:.0f}s")
        for task, score in dense_scores.items():
            print(f"    {task}: {score['value_pct']} ({score['metric']})")

    elif args.dense_baseline:
        # Use provided dense scores
        baseline = json.loads(args.dense_baseline)
        for task, val in baseline.items():
            dense_scores[task] = {
                "metric": "provided",
                "value": val / 100 if val > 1 else val,
                "value_pct": f"{val:.1f}%" if val > 1 else f"{val * 100:.1f}%",
            }

    # ── FUSE sparse evaluation ──
    sparse_scores = {}

    if not args.dense_only:
        print(f"\n  ═══ FUSE sparse evaluation ({args.tasks}) ═══")

        # Patch the model with FUSE
        if schedule_data:
            avg_sp = schedule_data["overall_avg_sparsity"]
            print(f"  Schedule: {args.schedule}")
            print(f"  Avg sparsity: {avg_sp:.1%}")
            print(f"  Range: {min(per_layer_sparsity):.1%} – "
                  f"{max(per_layer_sparsity):.1%}")
        else:
            print(f"  Flat sparsity: {args.target_sparsity:.1%}")

        fuse_layers = patch_model_with_fuse(
            model,
            target_sparsity=args.target_sparsity,
            strategy="top_k",
            relufied=args.relufied,
            per_layer_sparsity=per_layer_sparsity,
        )

        if args.limit:
            print(f"  Limit: {args.limit} examples per task")
        print(f"  This may take a while (sparse is slower in RAM-resident mode)...\n")

        t0 = time.perf_counter()
        sparse_results = run_eval(
            model, tokenizer, args.tasks, device,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )
        t_sparse = time.perf_counter() - t0

        sparse_scores = extract_scores(sparse_results)
        print(f"\n  Sparse eval completed in {t_sparse:.0f}s")
        for task, score in sparse_scores.items():
            print(f"    {task}: {score['value_pct']} ({score['metric']})")

        # Print FUSE stats
        avg_sparsity = sum(f.avg_sparsity for f in fuse_layers) / len(fuse_layers)
        total_fired = sum(f.fired_neurons for f in fuse_layers)
        total_neurons = sum(f.total_neurons for f in fuse_layers)
        print(f"\n  FUSE stats:")
        print(f"    Realized sparsity: {avg_sparsity:.1%}")
        print(f"    Neurons fired: {total_fired:,} / {total_neurons:,}")

    # ── Comparison ──
    if dense_scores and sparse_scores:
        print_comparison(dense_scores, sparse_scores, schedule_data)

    # ── Save results ──
    output_data = {
        "model": args.model,
        "tasks": args.tasks,
        "schedule": args.schedule,
        "sparsity": schedule_data["overall_avg_sparsity"] if schedule_data else args.target_sparsity,
        "limit": args.limit,
        "dense_scores": {k: v["value"] for k, v in dense_scores.items()},
        "sparse_scores": {k: v["value"] for k, v in sparse_scores.items()},
        "retained": {},
    }

    for task in set(list(dense_scores.keys()) + list(sparse_scores.keys())):
        d = dense_scores.get(task, {}).get("value")
        s = sparse_scores.get(task, {}).get("value")
        if d and s and d > 0:
            output_data["retained"][task] = round(s / d, 4)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  Results saved to: {args.output}")
    else:
        # Print JSON to stdout for easy capture
        print(f"\n  Results JSON:")
        print(f"  {json.dumps(output_data, indent=2)}")


if __name__ == "__main__":
    main()
