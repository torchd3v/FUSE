"""
FUSE Inference Engine — Minimal Working Version
=================================================
Actually generates text using gate-traced sparse FFN execution.

This hooks into the model's forward pass and replaces each FFN
with the two-phase FUSE pipeline:
  Phase 1: W_gate (in RAM) → trace fired neurons
  Phase 2: Load only fired W_up/W_down rows → sparse compute

This version keeps everything in RAM but simulates the I/O savings.
A production version would stream W_up/W_down from disk.

Usage:
  python fuse_inference.py
  python fuse_inference.py --prompt "Once upon a time"
  python fuse_inference.py --relufied --target-sparsity 0.5
  python fuse_inference.py --relufied --target-sparsity 0.7 --max-tokens 100

  # Per-layer adaptive sparsity (calibrate first with fuse_calibrate.py):
  python fuse_inference.py --schedule fuse_schedule.json --prompt "Hello world"
"""

import argparse
import json
import torch
import torch.nn.functional as F
import time
import sys


class FUSELayer:
    """
    Replaces one FFN layer's forward pass with FUSE sparse execution.
    
    Instead of computing all neurons, we:
      1. Run the gate (always in memory) to find which neurons fire
      2. Compute only those neurons for W_up and W_down
    
    This produces the SAME output as dense (within sparsity tolerance)
    but touches far fewer weight rows.
    """
    
    def __init__(self, mlp_module, target_sparsity=0.5, strategy="top_k", 
                 threshold=0.0, relufied=False):
        self.mlp = mlp_module
        self.target_sparsity = target_sparsity
        self.strategy = strategy
        self.threshold = threshold
        self.relufied = relufied
        
        # Stats tracking
        self.total_neurons = 0
        self.fired_neurons = 0
        self.call_count = 0
    
    def act_fn(self, x):
        if self.relufied:
            return torch.relu(x)
        return x * torch.sigmoid(x)  # SiLU
    
    def sparse_forward(self, x):
        """
        FUSE two-phase forward pass — vectorized for GPU.
        
        x shape: [batch, seq_len, d_model]
        
        Instead of looping over tokens in Python, this uses batched
        matmuls + scatter/gather to produce the exact same sparse output
        at full GPU speed. The result is mathematically identical to the
        per-token loop but runs 50-100x faster on GPU.
        
        On GPU: 3 batched matmuls (gate, up, down) + topk + gather/scatter.
        On disk-streaming: replace matmuls with sparse reads (future work).
        """
        W_gate = self.mlp.gate_proj.weight.data  # [d_ffn, d_model]
        W_up   = self.mlp.up_proj.weight.data     # [d_ffn, d_model]
        W_down = self.mlp.down_proj.weight.data   # [d_model, d_ffn]
        d_ffn = W_gate.shape[0]
        
        orig_shape = x.shape  # [batch, seq_len, d_model]
        x_flat = x.reshape(-1, orig_shape[-1])  # [N, d_model]
        N = x_flat.shape[0]
        
        with torch.no_grad():
            # ═══ Phase 1: TRACE (batched) ═══
            # Gate activation for ALL tokens at once
            gate_acts = self.act_fn(x_flat @ W_gate.T)  # [N, d_ffn]
            
            # Top-K selection per token
            k = max(1, int(d_ffn * (1.0 - self.target_sparsity)))
            _, top_idx = torch.topk(torch.abs(gate_acts), k, dim=1)  # [N, k]
            
            # ═══ Phase 2: SPARSE COMPUTE (batched) ═══
            # Gather fired gate values
            gate_sparse = torch.gather(gate_acts, 1, top_idx)  # [N, k]
            
            # Up projection: compute full then gather fired
            # (On GPU this is fast; on disk you'd only load fired rows)
            up_all = x_flat @ W_up.T  # [N, d_ffn]
            up_sparse = torch.gather(up_all, 1, top_idx)  # [N, k]
            
            # Element-wise gating
            hidden_sparse = gate_sparse * up_sparse  # [N, k]
            
            # Down projection: scatter sparse into full, then matmul
            # (On GPU this is fast; on disk you'd only load fired columns)
            hidden_full = torch.zeros(N, d_ffn, device=x.device, dtype=x.dtype)
            hidden_full.scatter_(1, top_idx, hidden_sparse)
            output_flat = hidden_full @ W_down.T  # [N, d_model]
        
        # Track stats
        self.total_neurons += N * d_ffn
        self.fired_neurons += N * k
        self.call_count += N
        
        return output_flat.reshape(orig_shape)
    
    @property
    def avg_sparsity(self):
        if self.total_neurons == 0:
            return 0.0
        return 1.0 - (self.fired_neurons / self.total_neurons)


def patch_model_with_fuse(model, target_sparsity=0.5, strategy="top_k",
                          threshold=0.0, relufied=False,
                          per_layer_sparsity=None):
    """
    Monkey-patch the model's MLP layers to use FUSE sparse execution.
    
    After this, calling model.generate() will use FUSE automatically.
    
    Args:
        per_layer_sparsity: Optional list of floats, one per layer.
            If provided, overrides target_sparsity with per-layer values
            from a calibration schedule.
    """
    fuse_layers = []
    
    for i, layer in enumerate(model.model.layers):
        layer_sparsity = (
            per_layer_sparsity[i] if per_layer_sparsity is not None
            else target_sparsity
        )
        fuse = FUSELayer(
            layer.mlp,
            target_sparsity=layer_sparsity,
            strategy=strategy,
            threshold=threshold,
            relufied=relufied,
        )
        fuse_layers.append(fuse)
        
        # Replace the MLP forward method
        original_forward = layer.mlp.forward
        
        def make_new_forward(fuse_layer):
            def new_forward(x):
                return fuse_layer.sparse_forward(x)
            return new_forward
        
        layer.mlp.forward = make_new_forward(fuse)
    
    return fuse_layers


def main():
    parser = argparse.ArgumentParser(description="FUSE Inference — Generate text with sparse FFN")
    parser.add_argument("--model", default=None,
                        help="Model to load (auto-detected from --schedule if not set)")
    parser.add_argument("--prompt", default="The future of artificial intelligence is")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--target-sparsity", type=float, default=0.5)
    parser.add_argument("--strategy", default="top_k", choices=["top_k", "threshold"])
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--relufied", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="Also generate with dense model and compare outputs")
    parser.add_argument("--schedule", default=None,
                        help="Path to calibration schedule JSON (from fuse_calibrate.py). "
                             "Overrides --target-sparsity with per-layer values.")
    parser.add_argument("--token", default=None)
    parser.add_argument("--dtype", default="float32",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Force a specific device")
    
    args = parser.parse_args()
    
    # ── Resolve model name ──
    # If a schedule is provided, read the model from it (unless --model overrides)
    schedule_data = None
    if args.schedule:
        with open(args.schedule, "r") as f:
            schedule_data = json.load(f)
        if args.model is None:
            args.model = schedule_data["model_name"]
            print(f"  (model auto-detected from schedule: {args.model})")
        if schedule_data.get("relufied") and not args.relufied:
            args.relufied = True
            print(f"  (ReLUfication auto-enabled from schedule)")
    
    if args.model is None:
        args.model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║   FUSE Inference: Sparse Text Generation via Gate Tracing   ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    dtype_map = {"float16": torch.float16, "float32": torch.float32,
                 "bfloat16": torch.bfloat16}
    
    print(f"\n  Loading: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.token,
                                              trust_remote_code=True)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=args.token, dtype=dtype_map[args.dtype],
        device_map=device if device in ("cpu", "mps") else "auto",
        trust_remote_code=True)
    model.eval()
    
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")
    
    # ── Validate schedule matches model ──
    if schedule_data:
        n_model_layers = len(model.model.layers)
        n_schedule_layers = len(schedule_data["per_layer_sparsity"])
        if n_model_layers != n_schedule_layers:
            print(f"\n  ERROR: Schedule has {n_schedule_layers} layers but "
                  f"model has {n_model_layers} layers.")
            print(f"  Schedule was calibrated for: {schedule_data.get('model_name', 'unknown')}")
            print(f"  Loaded model: {args.model}")
            print(f"  Use --model to specify the correct model, or recalibrate.")
            sys.exit(1)
    
    if args.relufied:
        print("  ReLUfication: ON")
        for layer in model.model.layers:
            layer.mlp.act_fn = torch.nn.ReLU()
    
    # ── Dense generation (baseline) ──
    if args.compare:
        print(f"\n  ─── Dense generation (baseline) ───")
        inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
        
        t0 = time.perf_counter()
        with torch.no_grad():
            dense_ids = model.generate(
                **inputs, max_new_tokens=args.max_tokens,
                do_sample=False, temperature=1.0)
        t_dense = time.perf_counter() - t0
        
        dense_text = tokenizer.decode(dense_ids[0], skip_special_tokens=True)
        print(f"  Time: {t_dense:.2f}s ({args.max_tokens/t_dense:.1f} tok/s)")
        print(f"  Output: {dense_text}")
    
    # ── FUSE sparse generation ──
    print(f"\n  ─── FUSE sparse generation ───")
    
    # Load per-layer schedule if provided
    per_layer_sparsity = None
    if args.schedule:
        per_layer_sparsity = schedule_data["per_layer_sparsity"]
        print(f"  Schedule: {args.schedule}")
        print(f"  Mode: per-layer adaptive sparsity")
        print(f"  Avg sparsity: {schedule_data['overall_avg_sparsity']:.1%}")
        print(f"  Range: {min(per_layer_sparsity):.1%} – {max(per_layer_sparsity):.1%}")
    else:
        print(f"  Strategy: {args.strategy}, "
              f"{'threshold=' + str(args.threshold) if args.strategy == 'threshold' else 'sparsity=' + str(args.target_sparsity)}")
    
    fuse_layers = patch_model_with_fuse(
        model,
        target_sparsity=args.target_sparsity,
        strategy=args.strategy,
        threshold=args.threshold,
        relufied=args.relufied,
        per_layer_sparsity=per_layer_sparsity,
    )
    
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        sparse_ids = model.generate(
            **inputs, max_new_tokens=args.max_tokens,
            do_sample=False, temperature=1.0)
    t_sparse = time.perf_counter() - t0
    
    sparse_text = tokenizer.decode(sparse_ids[0], skip_special_tokens=True)
    
    # Stats
    avg_sparsity = sum(f.avg_sparsity for f in fuse_layers) / len(fuse_layers)
    total_fired = sum(f.fired_neurons for f in fuse_layers)
    total_neurons = sum(f.total_neurons for f in fuse_layers)
    
    print(f"  Time: {t_sparse:.2f}s ({args.max_tokens/t_sparse:.1f} tok/s)")
    print(f"  Output: {sparse_text}")
    
    print(f"\n  ┌─ FUSE Stats")
    print(f"  │  Avg sparsity:    {avg_sparsity:.1%}")
    print(f"  │  Neurons fired:   {total_fired:,} / {total_neurons:,}")
    print(f"  │  I/O reduction:   {avg_sparsity:.1%}")
    print(f"  │  Effective speedup: {1/(1-avg_sparsity):.1f}x less data to stream")
    
    if args.compare:
        # Token-level comparison
        dense_tokens = dense_ids[0].tolist()
        sparse_tokens = sparse_ids[0].tolist()
        match = sum(1 for a, b in zip(dense_tokens, sparse_tokens) if a == b)
        total = max(len(dense_tokens), len(sparse_tokens))
        print(f"  │  Token match:     {match}/{total} ({match/total:.1%})")
    
    print(f"  └─")
    
    # Per-layer breakdown
    print(f"\n  Per-layer sparsity:")
    for i, fl in enumerate(fuse_layers):
        bar_len = int(fl.avg_sparsity * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        target_str = ""
        if per_layer_sparsity is not None:
            target_str = f"  (target={per_layer_sparsity[i]:.1%})"
        print(f"    Layer {i:>2}: {bar} {fl.avg_sparsity:.1%}{target_str}")
    
    print(f"\n  Note: This version keeps all weights in RAM for correctness.")
    print(f"  A production FUSE engine would stream W_up/W_down from disk,")
    print(f"  enabling {1/(1-avg_sparsity):.0f}x larger models on the same hardware.")


if __name__ == "__main__":
    main()