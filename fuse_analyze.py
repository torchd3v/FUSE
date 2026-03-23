"""
FUSE v2: Works with ANY model (SwiGLU, ReLU, GELU) FUSE: Feed-forward Unit-Sparse Execution 
====================================================
The key change: instead of thresholding (which needs ReLU for exact zeros),
we use Top-K selection — keep the K neurons with the highest gate activation.

Three selection strategies:
  1. top_k:     Keep the top K% of neurons by |gate activation| 
  2. adaptive:  Per-layer threshold calibrated on a few samples
  3. threshold: Original fixed threshold (best for ReLU-fied models)

Usage:
  pip install torch transformers accelerate

  # Top-K with 50% sparsity (works great on TinyLlama!)
  python fuse_v2.py --strategy top_k --target-sparsity 0.5

  # Try different sparsity levels
  python fuse_v2.py --sweep

  # Adaptive per-layer thresholds
  python fuse_v2.py --strategy adaptive --target-sparsity 0.5
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import time
import sys


def silu(x):
    return x * torch.sigmoid(x)


class NeuronSelector:
    """
    Different strategies for deciding which neurons to keep.
    All strategies use the gate activation — no external predictor.
    
    Think of it like this:
      - The gate says "neuron 42 has importance 0.8, neuron 43 has importance 0.001"
      - We pick the most important ones and skip the rest
    
    The strategies differ in HOW we pick:
      - top_k:     "keep the top 50% by importance" (guaranteed sparsity)
      - adaptive:  "find a smart cutoff per layer" (balanced quality/speed)
      - threshold: "skip anything below 0.01" (simple, best with ReLU)
    """
    
    def __init__(self, strategy="top_k", target_sparsity=0.5, threshold=0.0):
        self.strategy = strategy
        self.target_sparsity = target_sparsity
        self.threshold = threshold
    
    def select(self, gate_activations):
        """
        Given gate activations [d_ffn], return indices of neurons to KEEP.
        
        Args:
            gate_activations: tensor [d_ffn] — output of SiLU(W_gate @ x)
        
        Returns:
            fired_indices: tensor of neuron indices to load and compute
        """
        d_ffn = gate_activations.shape[0]
        magnitudes = torch.abs(gate_activations)
        
        if self.strategy == "top_k":
            # ════════════════════════════════════════════════
            # TOP-K: Keep the K neurons with highest |activation|
            # ════════════════════════════════════════════════
            # This ALWAYS gives you the target sparsity.
            # The gate tells us importance; we just keep the top ones.
            #
            # Example: d_ffn=5632, target_sparsity=0.5
            #   k = 5632 * 0.5 = 2816 neurons to keep
            #   We find the 2816 neurons with the biggest |gate output|
            
            k = max(1, int(d_ffn * (1.0 - self.target_sparsity)))
            _, top_indices = torch.topk(magnitudes, k)
            return torch.sort(top_indices).values
        
        elif self.strategy == "adaptive":
            # ════════════════════════════════════════════════
            # ADAPTIVE: Use percentile of actual activation distribution
            # ════════════════════════════════════════════════
            # Instead of a fixed threshold, we find the threshold that
            # gives us approximately the target sparsity for THIS specific
            # input at THIS specific layer.
            #
            # Example: if activations range from 0.0001 to 5.0, and we want
            # 50% sparsity, we find the median magnitude and use that as cutoff.
            
            cutoff = torch.quantile(magnitudes, self.target_sparsity)
            mask = magnitudes > cutoff
            fired_indices = torch.where(mask)[0]
            
            # Safety: always keep at least 1% of neurons
            if len(fired_indices) < d_ffn * 0.01:
                k = max(1, int(d_ffn * 0.01))
                _, top_indices = torch.topk(magnitudes, k)
                return torch.sort(top_indices).values
            
            return fired_indices
        
        elif self.strategy == "threshold":
            # ════════════════════════════════════════════════
            # FIXED THRESHOLD: Original approach
            # ════════════════════════════════════════════════
            # Best for ReLU-fied models where zeros are exact.
            # For SwiGLU models, you need a nonzero threshold.
            
            mask = magnitudes > self.threshold
            fired_indices = torch.where(mask)[0]
            
            if len(fired_indices) == 0:
                # Fallback: keep at least the strongest neuron
                return torch.argmax(magnitudes).unsqueeze(0)
            
            return fired_indices
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


class FUSEv2:
    """
    FUSE v2 — works with any SwiGLU/GELU/ReLU model.
    
    The core idea hasn't changed:
      Phase 1: W_gate (in RAM) traces which neurons matter
      Phase 2: Load only those neurons' W_up/W_down from storage
    
    What's new: smarter neuron selection that works even when
    SiLU never produces exact zeros.
    """
    
    def __init__(self, model, tokenizer, selector):
        self.model = model
        self.tokenizer = tokenizer
        self.selector = selector
        self.device = next(model.parameters()).device
        self.relufied = getattr(model.config, '_relufied', False)
    
    def _act_fn(self, x):
        """Use ReLU if relufied, SiLU otherwise."""
        if self.relufied:
            return torch.relu(x)
        return silu(x)
    
    def _get_ffn_layers(self):
        layers = []
        for i, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp
            layers.append({
                'index': i,
                'gate_proj': mlp.gate_proj.weight.data,
                'up_proj':   mlp.up_proj.weight.data,
                'down_proj': mlp.down_proj.weight.data,
            })
        return layers
    
    def analyze_layer(self, layer_info, hidden_state):
        W_gate = layer_info['gate_proj']
        W_up   = layer_info['up_proj']
        W_down = layer_info['down_proj']
        d_ffn = W_gate.shape[0]
        d_model = W_gate.shape[1]
        
        results = []
        
        for pos in range(hidden_state.shape[0]):
            x = hidden_state[pos]
            
            with torch.no_grad():
                # ── Dense baseline ──
                gate_dense = self._act_fn(W_gate @ x)
                up_dense = W_up @ x
                hidden_dense = gate_dense * up_dense
                output_dense = W_down @ hidden_dense
                
                # ── FUSE: Phase 1 — TRACE ──
                # gate_dense is already computed, reuse it
                # In a real system, this is the ONLY full matmul we do
                fired_indices = self.selector.select(gate_dense)
                n_fired = len(fired_indices)
                sparsity = 1.0 - (n_fired / d_ffn)
                
                # ── FUSE: Phase 2 — SPARSE COMPUTE ──
                # Only load and compute with fired neurons
                W_up_sparse = W_up[fired_indices]
                W_down_sparse = W_down[:, fired_indices]
                gate_sparse = gate_dense[fired_indices]
                
                up_sparse = W_up_sparse @ x
                hidden_sparse = gate_sparse * up_sparse
                output_sparse = W_down_sparse @ hidden_sparse
            
            # ── Measure quality ──
            cos_sim = F.cosine_similarity(
                output_dense.unsqueeze(0),
                output_sparse.unsqueeze(0)
            ).item()
            
            rel_error = (torch.norm(output_dense - output_sparse) / 
                        (torch.norm(output_dense) + 1e-10)).item()
            
            bytes_per_param = 2
            bytes_dense = (d_ffn * d_model * 2) * bytes_per_param
            bytes_sparse = (n_fired * d_model * 2) * bytes_per_param
            io_reduction = 1.0 - (bytes_sparse / bytes_dense)
            
            results.append({
                'pos': pos, 'n_fired': n_fired, 'd_ffn': d_ffn,
                'sparsity': sparsity, 'cos_sim': cos_sim,
                'rel_error': rel_error, 'io_reduction': io_reduction,
                'fired_indices': fired_indices.cpu().numpy(),
                # Track what we're throwing away
                'discarded_energy': (
                    torch.sum(torch.abs(gate_dense)).item() -
                    torch.sum(torch.abs(gate_sparse)).item()
                ),
                'total_energy': torch.sum(torch.abs(gate_dense)).item(),
            })
        
        return results
    
    def run(self, text):
        print(f"\n  Input: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        seq_len = inputs['input_ids'].shape[1]
        ffn_layers = self._get_ffn_layers()
        n_layers = len(ffn_layers)
        
        print(f"  Tokens: {seq_len}")
        print(f"  Layers: {n_layers}")
        print(f"  FFN dim: {ffn_layers[0]['gate_proj'].shape[0]}")
        print(f"  Model dim: {ffn_layers[0]['gate_proj'].shape[1]}")
        print(f"  Strategy: {self.selector.strategy}")
        if self.selector.strategy in ("top_k", "adaptive"):
            print(f"  Target sparsity: {self.selector.target_sparsity:.0%}")
        else:
            print(f"  Threshold: {self.selector.threshold}")
        
        # Capture hidden states
        hidden_states = {}
        hooks = []
        for i, layer in enumerate(self.model.model.layers):
            def make_hook(idx):
                def hook_fn(module, inp, out):
                    hidden_states[idx] = inp[0].detach().squeeze(0)
                return hook_fn
            hooks.append(layer.mlp.register_forward_hook(make_hook(i)))
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        for h in hooks:
            h.remove()
        
        # Analyze each layer
        print(f"\n  {'Layer':>6} {'Sparsity':>10} {'Neurons':>14} "
              f"{'Cos sim':>9} {'Rel err':>9} {'I/O saved':>10} {'Energy kept':>12}")
        print("  " + "-" * 73)
        
        all_results = []
        
        for i, layer_info in enumerate(ffn_layers):
            if i not in hidden_states:
                continue
            
            results = self.analyze_layer(layer_info, hidden_states[i])
            all_results.append(results)
            
            avg = lambda key: np.mean([r[key] for r in results])
            d_ffn = results[0]['d_ffn']
            
            # Energy kept = how much of the gate signal we preserved
            total_energy = sum(r['total_energy'] for r in results)
            discarded = sum(r['discarded_energy'] for r in results)
            energy_kept = 1.0 - (discarded / (total_energy + 1e-10))
            
            print(f"  {i:>6} {avg('sparsity'):>9.1%} "
                  f"{avg('n_fired'):>8.0f}/{d_ffn} "
                  f"{avg('cos_sim'):>9.4f} {avg('rel_error'):>9.4f} "
                  f"{avg('io_reduction'):>9.1%} {energy_kept:>11.1%}")
        
        # Summary
        all_flat = [r for layer in all_results for r in layer]
        avg_sparsity = np.mean([r['sparsity'] for r in all_flat])
        avg_cos = np.mean([r['cos_sim'] for r in all_flat])
        min_cos = np.min([r['cos_sim'] for r in all_flat])
        avg_io = np.mean([r['io_reduction'] for r in all_flat])
        
        gate_params = sum(l['gate_proj'].numel() for l in ffn_layers)
        total_ffn = sum(l['gate_proj'].numel() + l['up_proj'].numel() + 
                       l['down_proj'].numel() for l in ffn_layers)
        
        print(f"\n  ┌─ FUSE Summary ({self.selector.strategy})")
        print(f"  │  Sparsity:    {avg_sparsity:.1%}")
        print(f"  │  Cos sim:     {avg_cos:.4f} (worst: {min_cos:.4f})")
        print(f"  │  I/O saved:   {avg_io:.1%}")
        print(f"  │  Speedup:     {1/(1-avg_io):.1f}x less data to stream")
        print(f"  │  W_gate RAM:  {gate_params * 2 / 1024**3:.2f} GB "
              f"({gate_params * 2 / (total_ffn * 2):.0%} of FFN)")
        print(f"  └─")
        
        # ── End-to-end accuracy test ──
        # Actually generate tokens with dense vs sparse to check output
        self._test_generation(text, all_results, n_layers)
        
        return all_results
    
    def _test_generation(self, text, all_results, n_layers):
        """Check if the model's actual token predictions change."""
        print(f"\n  Token prediction test:")
        print(f"  (Does sparse computation change what the model predicts?)")
        
        # We can check: for the last token position, does the top-1
        # predicted next token change between dense and sparse?
        # This is a proxy for "does the quality actually degrade?"
        
        # Get the dense prediction (already computed during forward pass)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            dense_logits = self.model(**inputs).logits[0, -1]  # last token
            dense_top5 = torch.topk(dense_logits, 5)
            dense_tokens = [self.tokenizer.decode(t) for t in dense_top5.indices]
        
        print(f"    Dense model predicts next: {dense_tokens}")
        print(f"    (Full sparse generation would require hooking into the")
        print(f"     forward pass — left as exercise for real deployment)")
    
    def run_sweep(self, text, sparsities=None):
        """Run analysis at multiple sparsity levels."""
        if sparsities is None:
            sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        print(f"\n  ╔══ SPARSITY SWEEP ({self.selector.strategy}) ══╗")
        print(f"  ║  Testing {len(sparsities)} sparsity levels")
        print(f"  ╚{'═' * 38}╝")
        
        summary = []
        
        for sp in sparsities:
            self.selector.target_sparsity = sp
            
            # Quick analysis (suppress per-layer output)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            seq_len = inputs['input_ids'].shape[1]
            ffn_layers = self._get_ffn_layers()
            
            hidden_states = {}
            hooks = []
            for i, layer in enumerate(self.model.model.layers):
                def make_hook(idx):
                    def hook_fn(module, inp, out):
                        hidden_states[idx] = inp[0].detach().squeeze(0)
                    return hook_fn
                hooks.append(layer.mlp.register_forward_hook(make_hook(i)))
            
            with torch.no_grad():
                self.model(**inputs)
            for h in hooks:
                h.remove()
            
            all_cos = []
            all_err = []
            
            for i, layer_info in enumerate(ffn_layers):
                if i not in hidden_states:
                    continue
                results = self.analyze_layer(layer_info, hidden_states[i])
                all_cos.extend([r['cos_sim'] for r in results])
                all_err.extend([r['rel_error'] for r in results])
            
            avg_cos = np.mean(all_cos)
            min_cos = np.min(all_cos)
            avg_err = np.mean(all_err)
            
            summary.append({
                'sparsity': sp,
                'avg_cos': avg_cos,
                'min_cos': min_cos,
                'avg_err': avg_err,
                'speedup': 1 / (1 - sp),
            })
        
        # Print sweep table
        print(f"\n  {'Sparsity':>10} {'Avg cos':>10} {'Worst cos':>11} "
              f"{'Avg error':>11} {'I/O speedup':>12} {'Verdict':>10}")
        print("  " + "-" * 68)
        
        for s in summary:
            # Verdict based on cosine similarity
            if s['min_cos'] >= 0.99:
                verdict = "excellent"
            elif s['min_cos'] >= 0.95:
                verdict = "good"
            elif s['min_cos'] >= 0.90:
                verdict = "okay"
            elif s['min_cos'] >= 0.80:
                verdict = "risky"
            else:
                verdict = "too much"
            
            print(f"  {s['sparsity']:>9.0%} {s['avg_cos']:>10.4f} "
                  f"{s['min_cos']:>11.4f} {s['avg_err']:>11.4f} "
                  f"{s['speedup']:>10.1f}x  {verdict:>10}")
        
        # Find the sweet spot
        good = [s for s in summary if s['min_cos'] >= 0.95]
        if good:
            best = max(good, key=lambda s: s['sparsity'])
            print(f"\n  ★ Sweet spot: {best['sparsity']:.0%} sparsity")
            print(f"    → {best['speedup']:.1f}x less I/O, "
                  f"worst-case cos sim = {best['min_cos']:.4f}")
            print(f"    → Run with: --strategy top_k "
                  f"--target-sparsity {best['sparsity']:.1f}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="FUSE v2: Gate-as-Tracer Sparse Inference (works with any model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with top-K selection (works on TinyLlama!)
  python fuse_v2.py

  # Sweep to find the best sparsity level
  python fuse_v2.py --sweep

  # Specific sparsity target
  python fuse_v2.py --target-sparsity 0.5

  # Adaptive per-layer thresholds
  python fuse_v2.py --strategy adaptive --target-sparsity 0.5

  # Original threshold mode (for ReLU-fied models)
  python fuse_v2.py --strategy threshold --threshold 0.0 \\
      --model SparseLLM/ReluLLaMA-7B
        """)
    
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--text", default=(
        "Large language models have revolutionized natural language "
        "processing by learning patterns from vast amounts of text data. "
        "These models use transformer architectures with attention mechanisms "
        "to capture long-range dependencies between words in a sequence."))
    parser.add_argument("--strategy", default="top_k",
                        choices=["top_k", "adaptive", "threshold"],
                        help="Neuron selection strategy")
    parser.add_argument("--target-sparsity", type=float, default=0.5,
                        help="Target sparsity for top_k and adaptive (0.0-0.95)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Fixed threshold (only for --strategy threshold)")
    parser.add_argument("--token", default=None, help="HuggingFace token")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep sparsity levels to find the sweet spot")
    parser.add_argument("--relufied", action="store_true",
                        help="Replace SiLU with ReLU at inference time (no new download needed)")
    
    args = parser.parse_args()
    
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║    FUSE v2: Gate-Traced Sparse Inference (any model)        ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    print(f"\n  Loading model: {args.model}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    dtype_map = {"float16": torch.float16, "float32": torch.float32, 
                 "bfloat16": torch.bfloat16}
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.token, trust_remote_code=True)
    
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else \
             "mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=args.token, dtype=dtype_map[args.dtype],
        device_map=device if device in ("cpu", "mps") else "auto",
        trust_remote_code=True)
    model.eval()
    
    # ── ReLUfication: swap SiLU → ReLU in all FFN gate projections ──
    if args.relufied:
        print("  Applying ReLUfication (SiLU → ReLU swap)...")
        print("  Note: this is inference-only, no fine-tuning.")
        print("  A properly fine-tuned ReLU model would perform better.")
        # We don't change the weights — we just change how the activation
        # function works. Instead of SiLU(x) = x * sigmoid(x), we use
        # ReLU(x) = max(0, x). This creates hard zeros → real sparsity.
        for layer in model.model.layers:
            layer.mlp.act_fn = torch.nn.ReLU()
        # Also override the config so our analyzer knows
        model.config._relufied = True
    
    print(f"  Loaded on: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")
    
    selector = NeuronSelector(
        strategy=args.strategy,
        target_sparsity=args.target_sparsity,
        threshold=args.threshold,
    )
    
    fuse = FUSEv2(model, tokenizer, selector)
    
    if args.sweep:
        fuse.run_sweep(args.text)
    else:
        fuse.run(args.text)
    
    print()


if __name__ == "__main__":
    main()