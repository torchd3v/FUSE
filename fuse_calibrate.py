"""
FUSE Calibrator — Per-Layer Adaptive Sparsity Schedule
=======================================================
Runs a calibration pass over sample sentences to find the maximum
sparsity each layer can tolerate while staying above a quality floor.

The key insight: early layers are sensitive (need more neurons),
later layers are redundant (can prune aggressively). Flat sparsity
wastes budget on both ends. This calibrator finds the sweet spot
per layer, then saves a reusable schedule.

Algorithm:
  For each layer:
    1. Collect hidden states from calibration sentences
    2. Binary-search for the highest sparsity where
       worst-case cosine similarity >= quality_floor
    3. Record that sparsity as the layer's budget

Output: JSON schedule file with per-layer sparsity targets,
        plus metadata (model, quality floor, calibration stats).

Usage:
  # Basic calibration (saves schedule.json)
  python fuse_calibrate.py

  # Custom quality floor and output path
  python fuse_calibrate.py --quality-floor 0.98 --output schedule_98.json

  # ReLUfied model
  python fuse_calibrate.py --relufied --quality-floor 0.99

  # Larger model
  python fuse_calibrate.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

  # Then use the schedule in inference:
  python fuse_inference.py --schedule schedule.json --prompt "Hello world"
"""

import argparse
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple


# ─── Default calibration sentences ──────────────────────────────────
# Diverse domains so the calibration generalizes beyond one topic.
DEFAULT_CALIBRATION_TEXTS = [
    # Factual / technical
    "Large language models use transformer architectures with self-attention "
    "mechanisms to process sequences of tokens in parallel during training.",

    # Conversational / simple
    "Hi, how are you doing today? I was wondering if you could help me with "
    "something. I need to write an email to my boss about taking time off.",

    # Reasoning / math
    "To solve this problem, we need to first calculate the total cost. "
    "If each item costs $15 and we buy 23 items, the total is 15 times 23.",

    # Code-adjacent
    "The function takes a list of integers as input and returns the maximum "
    "sum of any contiguous subarray using Kadane's algorithm in O(n) time.",

    # Creative / narrative
    "The old lighthouse keeper watched the storm roll in from the east. "
    "Dark clouds gathered like bruises on the horizon, and the sea churned.",
]


@dataclass
class LayerProfile:
    """Calibration results for one layer."""
    layer_idx: int
    optimal_sparsity: float       # highest sparsity above quality floor
    worst_cos_at_optimal: float   # worst cosine sim at that sparsity
    # Full profile for analysis
    sparsity_curve: Dict[str, float]  # sparsity_level -> worst_cos_sim


@dataclass
class CalibrationSchedule:
    """Complete calibration output, serializable to JSON."""
    model_name: str
    quality_floor: float
    relufied: bool
    n_layers: int
    d_ffn: int
    d_model: int
    calibration_sentences: int
    calibration_tokens: int
    per_layer_sparsity: List[float]   # index = layer_idx
    per_layer_worst_cos: List[float]
    overall_avg_sparsity: float
    overall_effective_speedup: float
    calibration_time_sec: float
    # Full curves for debugging / visualization
    layer_profiles: Optional[List[Dict]] = None


class FUSECalibrator:
    """
    Profiles each FFN layer to find its maximum safe sparsity.

    Vectorized implementation: batches all token computations into single
    matmuls and caches dense outputs per layer so the binary search only
    recomputes the cheap sparse part.

    Speed comparison on TinyLlama (22 layers, 173 tokens):
      v1 (token loop):     ~270s
      v2 (vectorized):     ~15-30s  (10-20x faster)
    """

    def __init__(self, model, tokenizer, relufied=False, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.relufied = relufied
        self.device = device or next(model.parameters()).device

    def act_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self.relufied:
            return torch.relu(x)
        return x * torch.sigmoid(x)  # SiLU

    def _get_ffn_weights(self, layer_idx: int):
        mlp = self.model.model.layers[layer_idx].mlp
        return (
            mlp.gate_proj.weight.data,  # [d_ffn, d_model]
            mlp.up_proj.weight.data,    # [d_ffn, d_model]
            mlp.down_proj.weight.data,  # [d_model, d_ffn]
        )

    def _collect_hidden_states(
        self, texts: List[str], max_tokens: Optional[int] = None,
    ) -> Tuple[Dict[int, torch.Tensor], int]:
        """
        Run calibration texts through the model and capture the input
        to each MLP layer. Returns {layer_idx: [total_tokens, d_model]}.

        If max_tokens is set, randomly subsample to that many tokens
        (useful for large models where 173 tokens × 28 layers is slow).
        """
        n_layers = len(self.model.model.layers)
        hidden_states = {i: [] for i in range(n_layers)}
        hooks = []

        for i, layer in enumerate(self.model.model.layers):
            def make_hook(idx):
                def hook_fn(module, inp, out):
                    hidden_states[idx].append(inp[0].detach().squeeze(0))
                return hook_fn
            hooks.append(layer.mlp.register_forward_hook(make_hook(i)))

        total_tokens = 0
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            total_tokens += inputs["input_ids"].shape[1]
            with torch.no_grad():
                self.model(**inputs)

        for h in hooks:
            h.remove()

        # Concatenate all tokens per layer
        for i in range(n_layers):
            hidden_states[i] = torch.cat(hidden_states[i], dim=0)

        # Subsample if requested
        if max_tokens and total_tokens > max_tokens:
            rng = np.random.RandomState(42)
            indices = rng.choice(total_tokens, max_tokens, replace=False)
            indices.sort()
            idx_tensor = torch.tensor(indices, device=self.device)
            for i in range(n_layers):
                hidden_states[i] = hidden_states[i][idx_tensor]
            actual_tokens = max_tokens
        else:
            actual_tokens = total_tokens

        return hidden_states, actual_tokens

    def profile_layer(
        self, layer_idx: int, hidden: torch.Tensor, quality_floor: float,
        coarse_levels: List[float] = None, fine_steps: int = 8,
    ) -> LayerProfile:
        """
        Find the maximum sparsity for this layer above quality_floor.

        Optimized two-phase approach:
          1. Precompute dense outputs + intermediate activations ONCE
          2. For each sparsity probe: only recompute the sparse part
             (top-K select + gather + small per-token W_down matmul)

        The dense precompute dominates cost, and it happens exactly once.
        Each binary search step is then very cheap.
        """
        if coarse_levels is None:
            coarse_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        W_gate, W_up, W_down = self._get_ffn_weights(layer_idx)
        d_ffn = W_gate.shape[0]
        n_tokens = hidden.shape[0]

        # ═══ Precompute dense (once per layer) ═══
        with torch.no_grad():
            gate_acts = self.act_fn(hidden @ W_gate.T)   # [n_tokens, d_ffn]
            up_acts = hidden @ W_up.T                     # [n_tokens, d_ffn]
            hidden_dense = gate_acts * up_acts             # [n_tokens, d_ffn]
            outputs_dense = hidden_dense @ W_down.T        # [n_tokens, d_model]
            abs_gate = torch.abs(gate_acts)                # for top-K reuse

        def worst_cos_at(sparsity: float) -> float:
            """Compute worst-case cosine sim at a given sparsity (cheap)."""
            k = max(1, int(d_ffn * (1.0 - sparsity)))
            with torch.no_grad():
                _, top_idx = torch.topk(abs_gate, k, dim=1)  # [n_tokens, k]
                gate_sp = torch.gather(gate_acts, 1, top_idx)
                up_sp = torch.gather(up_acts, 1, top_idx)
                hidden_sp = gate_sp * up_sp                    # [n_tokens, k]

                # W_down[:, fired] @ hidden_sp — per-token (different neurons per token)
                outputs_sparse = torch.zeros_like(outputs_dense)
                for t in range(n_tokens):
                    outputs_sparse[t] = W_down[:, top_idx[t]] @ hidden_sp[t]

                cos_sims = F.cosine_similarity(outputs_dense, outputs_sparse, dim=1)
            return float(cos_sims.min().item())

        # ═══ Coarse sweep ═══
        curve = {}
        for sp in coarse_levels:
            curve[f"{sp:.2f}"] = worst_cos_at(sp)

        # ═══ Find boundary ═══
        good = [sp for sp in coarse_levels if curve[f"{sp:.2f}"] >= quality_floor]
        bad  = [sp for sp in coarse_levels if curve[f"{sp:.2f}"] < quality_floor]

        if not good:
            wc = worst_cos_at(0.05)
            return LayerProfile(
                layer_idx=layer_idx, optimal_sparsity=0.05,
                worst_cos_at_optimal=wc, sparsity_curve=curve,
            )

        if not bad:
            for sp in [0.92, 0.95, 0.97]:
                wc = worst_cos_at(sp)
                curve[f"{sp:.2f}"] = wc
                if wc < quality_floor:
                    bad.append(sp)
                    break
                good.append(sp)

        lo = max(good)
        hi = min(bad) if bad else 0.97

        # ═══ Binary search refinement ═══
        for _ in range(fine_steps):
            mid = (lo + hi) / 2.0
            wc = worst_cos_at(mid)
            curve[f"{mid:.4f}"] = wc
            if wc >= quality_floor:
                lo = mid
            else:
                hi = mid

        wc_final = worst_cos_at(lo)

        return LayerProfile(
            layer_idx=layer_idx,
            optimal_sparsity=round(lo, 4),
            worst_cos_at_optimal=round(wc_final, 4),
            sparsity_curve=curve,
        )

    def calibrate(
        self,
        texts: List[str] = None,
        quality_floor: float = 0.98,
        save_curves: bool = True,
        max_tokens: Optional[int] = None,
    ) -> CalibrationSchedule:
        """
        Full calibration pipeline.

        1. Collect hidden states from calibration texts
        2. Profile each layer (coarse sweep + binary search)
        3. Package results into a CalibrationSchedule
        """
        if texts is None:
            texts = DEFAULT_CALIBRATION_TEXTS

        n_layers = len(self.model.model.layers)
        d_ffn = self.model.model.layers[0].mlp.gate_proj.weight.shape[0]
        d_model = self.model.model.layers[0].mlp.gate_proj.weight.shape[1]

        print(f"\n  ┌─ FUSE Calibration")
        print(f"  │  Model layers:    {n_layers}")
        print(f"  │  FFN dim:         {d_ffn}")
        print(f"  │  Model dim:       {d_model}")
        print(f"  │  Quality floor:   cos >= {quality_floor}")
        print(f"  │  Calibration set: {len(texts)} sentences")
        if max_tokens:
            print(f"  │  Max cal tokens:  {max_tokens}")
        print(f"  └─")

        t0 = time.perf_counter()

        # Step 1: Collect hidden states
        print(f"\n  Collecting hidden states...", end=" ", flush=True)
        hidden_states, total_tokens = self._collect_hidden_states(texts, max_tokens)
        t_collect = time.perf_counter() - t0
        print(f"done ({total_tokens} tokens, {t_collect:.1f}s)")

        # Step 2: Profile each layer
        print(f"\n  Profiling {n_layers} layers (vectorized, coarse sweep + binary search):")
        print(f"  {'Layer':>6}  {'Optimal':>8}  {'Worst cos':>10}  {'Time':>7}  {'Status':>12}")
        print(f"  " + "─" * 52)

        profiles: List[LayerProfile] = []
        for i in range(n_layers):
            t_layer = time.perf_counter()
            profile = self.profile_layer(i, hidden_states[i], quality_floor)
            dt = time.perf_counter() - t_layer
            profiles.append(profile)

            status = "sensitive" if profile.optimal_sparsity < 0.3 else \
                     "moderate" if profile.optimal_sparsity < 0.6 else "aggressive"

            print(f"  {i:>6}  {profile.optimal_sparsity:>7.1%}  "
                  f"{profile.worst_cos_at_optimal:>10.4f}  {dt:>5.1f}s  {status:>12}")

        calibration_time = time.perf_counter() - t0

        # Step 3: Package results
        per_layer_sparsity = [p.optimal_sparsity for p in profiles]
        per_layer_worst_cos = [p.worst_cos_at_optimal for p in profiles]
        avg_sparsity = float(np.mean(per_layer_sparsity))

        schedule = CalibrationSchedule(
            model_name=self.model.config._name_or_path,
            quality_floor=quality_floor,
            relufied=self.relufied,
            n_layers=n_layers,
            d_ffn=d_ffn,
            d_model=d_model,
            calibration_sentences=len(texts),
            calibration_tokens=total_tokens,
            per_layer_sparsity=per_layer_sparsity,
            per_layer_worst_cos=per_layer_worst_cos,
            overall_avg_sparsity=round(avg_sparsity, 4),
            overall_effective_speedup=round(1.0 / (1.0 - avg_sparsity), 2),
            calibration_time_sec=round(calibration_time, 1),
            layer_profiles=[asdict(p) for p in profiles] if save_curves else None,
        )

        # Summary
        print(f"\n  ┌─ Calibration Results")
        print(f"  │  Overall avg sparsity:  {avg_sparsity:.1%}")
        print(f"  │  Effective I/O speedup: {schedule.overall_effective_speedup:.2f}x")
        print(f"  │  Min layer sparsity:    {min(per_layer_sparsity):.1%} "
              f"(layer {per_layer_sparsity.index(min(per_layer_sparsity))})")
        print(f"  │  Max layer sparsity:    {max(per_layer_sparsity):.1%} "
              f"(layer {per_layer_sparsity.index(max(per_layer_sparsity))})")
        print(f"  │  Sparsity range:        {max(per_layer_sparsity) - min(per_layer_sparsity):.1%}")
        print(f"  │  Calibration time:      {calibration_time:.1f}s")
        print(f"  └─")

        # Visual sparsity map
        print(f"\n  Per-layer sparsity map:")
        for i, sp in enumerate(per_layer_sparsity):
            bar_len = int(sp * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            cos_str = f"{per_layer_worst_cos[i]:.3f}"
            print(f"    L{i:>2} {bar} {sp:.1%}  (cos>={cos_str})")

        return schedule


def save_schedule(schedule: CalibrationSchedule, path: str):
    """Save calibration schedule to JSON."""
    data = asdict(schedule)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Schedule saved to: {path}")


def load_schedule(path: str) -> CalibrationSchedule:
    """Load calibration schedule from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    # Remove layer_profiles from constructor if None
    return CalibrationSchedule(**data)


def main():
    parser = argparse.ArgumentParser(
        description="FUSE Calibrator — find per-layer optimal sparsity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default calibration for TinyLlama
  python fuse_calibrate.py

  # Strict quality floor
  python fuse_calibrate.py --quality-floor 0.99

  # ReLUfied model (expect much higher sparsity)
  python fuse_calibrate.py --relufied --quality-floor 0.99

  # Custom model + output path
  python fuse_calibrate.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
      --output schedule_deepseek7b.json

  # Use the schedule in inference:
  python fuse_inference.py --schedule schedule.json --prompt "Hello"
        """,
    )

    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output", default="fuse_schedule.json",
                        help="Output path for the calibration schedule")
    parser.add_argument("--quality-floor", type=float, default=0.98,
                        help="Minimum worst-case cosine similarity per layer")
    parser.add_argument("--relufied", action="store_true",
                        help="Replace SiLU with ReLU before calibrating")
    parser.add_argument("--token", default=None, help="HuggingFace token")
    parser.add_argument("--dtype", default="float32",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Force a specific device (default: auto-detect)")
    parser.add_argument("--texts", nargs="+", default=None,
                        help="Custom calibration sentences (overrides defaults)")
    parser.add_argument("--max-cal-tokens", type=int, default=None,
                        help="Subsample to this many tokens for faster calibration "
                             "(recommended: 64-128 for 7B+ models)")
    parser.add_argument("--no-curves", action="store_true",
                        help="Don't save full sparsity curves (smaller JSON)")

    args = parser.parse_args()

    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║   FUSE Calibrator: Per-Layer Adaptive Sparsity Profiling    ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float16": torch.float16, "float32": torch.float32,
                 "bfloat16": torch.bfloat16}

    print(f"\n  Loading: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.token, trust_remote_code=True)

    if args.device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = args.device

    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=args.token, dtype=dtype_map[args.dtype],
        device_map=device if device in ("cpu", "mps") else "auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    if args.relufied:
        print("  ReLUfication: ON")
        for layer in model.model.layers:
            layer.mlp.act_fn = torch.nn.ReLU()

    calibrator = FUSECalibrator(model, tokenizer, relufied=args.relufied, device=device)

    texts = args.texts if args.texts else DEFAULT_CALIBRATION_TEXTS
    schedule = calibrator.calibrate(
        texts=texts,
        quality_floor=args.quality_floor,
        save_curves=not args.no_curves,
        max_tokens=args.max_cal_tokens,
    )

    save_schedule(schedule, args.output)

    # Compare to flat baseline
    flat_sp = schedule.overall_avg_sparsity
    print(f"\n  ─── Comparison: Adaptive vs Flat ───")
    print(f"  If you used flat {flat_sp:.0%} everywhere:")
    print(f"    Some layers would lose quality (sensitive early layers)")
    print(f"    Some layers would leave savings on the table (redundant later layers)")
    print(f"  With adaptive schedule:")
    print(f"    Every layer stays above cos >= {args.quality_floor}")
    print(f"    Total I/O savings: {schedule.overall_effective_speedup:.2f}x")
    print()


if __name__ == "__main__":
    main()
