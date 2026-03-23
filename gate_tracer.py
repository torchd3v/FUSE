"""
Gate-as-Tracer: Sparse Streaming LLM Inference
================================================
A prototype demonstrating two-phase FFN inference:
  Phase 1: Compute gate activations (W_gate always in RAM) → get sparse mask
  Phase 2: Load only fired neuron rows of W_up/W_down from disk → compute

This eliminates the need for trained predictors (used by PowerInfer, LLM in a Flash)
by using the model's own gate mechanism as ground-truth activation tracer.

Author: Brainstorm prototype
"""

import numpy as np
import time
import os
import json
import tempfile
from dataclasses import dataclass, field
from typing import Tuple, Dict, List


# ─── Model Configuration ────────────────────────────────────────────
@dataclass
class ModelConfig:
    """LLaMA-style model configuration."""
    name: str
    d_model: int        # hidden dimension
    d_ffn: int          # FFN intermediate dimension
    n_layers: int       # number of transformer layers
    dtype_bytes: int = 2  # FP16 = 2 bytes per param

    @property
    def params_per_layer_ffn(self) -> int:
        """Total FFN parameters per layer (3 matrices: gate, up, down)."""
        return 3 * self.d_model * self.d_ffn

    @property
    def bytes_per_layer_ffn(self) -> int:
        return self.params_per_layer_ffn * self.dtype_bytes

    @property
    def bytes_gate(self) -> int:
        """Size of W_gate per layer."""
        return self.d_model * self.d_ffn * self.dtype_bytes

    @property
    def bytes_up(self) -> int:
        return self.d_model * self.d_ffn * self.dtype_bytes

    @property
    def bytes_down(self) -> int:
        return self.d_model * self.d_ffn * self.dtype_bytes

    @property
    def total_ffn_bytes(self) -> int:
        return self.bytes_per_layer_ffn * self.n_layers

    @property
    def total_model_gb(self) -> float:
        # Rough: FFN is ~2/3 of total params
        return (self.total_ffn_bytes / 0.667) / (1024**3)


# Predefined model configs
MODELS = {
    "LLaMA-7B":  ModelConfig("LLaMA-7B",  d_model=4096,  d_ffn=11008, n_layers=32),
    "LLaMA-13B": ModelConfig("LLaMA-13B", d_model=5120,  d_ffn=13824, n_layers=40),
    "LLaMA-70B": ModelConfig("LLaMA-70B", d_model=8192,  d_ffn=28672, n_layers=80),
}


# ─── SwiGLU FFN Layer ───────────────────────────────────────────────
class SwiGLU_FFN:
    """
    A single SwiGLU FFN layer with gate-as-tracer capability.
    
    Standard SwiGLU: output = W_down @ (SiLU(W_gate @ x) * (W_up @ x))
    
    Two-phase:
      Phase 1: gate_act = SiLU(W_gate @ x), mask = |gate_act| > threshold
      Phase 2: load only W_up[mask] and W_down[:, mask], compute sparse output
    """
    
    def __init__(self, d_model: int, d_ffn: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / d_model)
        
        # Initialize weights (simulating a trained model)
        # In practice these would be loaded from a model file
        self.W_gate = (rng.randn(d_ffn, d_model) * scale).astype(np.float32)
        self.W_up   = (rng.randn(d_ffn, d_model) * scale).astype(np.float32)
        self.W_down = (rng.randn(d_model, d_ffn) * scale).astype(np.float32)
        
        self.d_model = d_model
        self.d_ffn = d_ffn
    
    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        """SiLU / Swish activation: x * sigmoid(x)"""
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))
    
    def forward_dense(self, x: np.ndarray) -> np.ndarray:
        """Standard dense forward pass (baseline)."""
        gate_out = self.silu(self.W_gate @ x)
        up_out = self.W_up @ x
        hidden = gate_out * up_out
        return self.W_down @ hidden
    
    def forward_sparse(self, x: np.ndarray, threshold: float = 0.1
                       ) -> Tuple[np.ndarray, Dict]:
        """
        Two-phase gate-as-tracer forward pass.
        
        Phase 1: Compute gate → identify fired neurons (TRACE)
        Phase 2: Load + compute only fired neurons (COMPUTE)
        
        Returns: (output, stats_dict)
        """
        # ── Phase 1: TRACE (W_gate is always in RAM) ──
        gate_pre = self.W_gate @ x           # gate pre-activation
        gate_act = self.silu(gate_pre)        # gate activation
        
        # Identify fired neurons
        mask = np.abs(gate_act) > threshold
        fired_indices = np.where(mask)[0]
        n_fired = len(fired_indices)
        sparsity = 1.0 - (n_fired / self.d_ffn)
        
        # ── Phase 2: COMPUTE (load only fired rows/cols) ──
        # In a real system, these would be loaded from SSD/NVMe
        W_up_sparse   = self.W_up[fired_indices]       # only fired rows
        W_down_sparse = self.W_down[:, fired_indices]   # only fired columns
        gate_sparse   = gate_act[fired_indices]
        
        # Sparse computation
        up_sparse = W_up_sparse @ x
        hidden_sparse = gate_sparse * up_sparse
        output = W_down_sparse @ hidden_sparse
        
        stats = {
            "n_fired": n_fired,
            "n_total": self.d_ffn,
            "sparsity": sparsity,
            "fired_indices": fired_indices,
        }
        return output, stats


# ─── Neuron-Level Storage Format ─────────────────────────────────────
class NeuronIndexedStore:
    """
    Custom storage format optimized for sparse neuron loading.
    
    Instead of storing weight matrices as flat arrays, we store
    per-neuron bundles: for neuron i, store W_up[i, :] and W_down[:, i]
    contiguously. This means activating neuron i requires ONE
    sequential read instead of two random reads.
    
    File layout:
      [Header: neuron_id → (offset, size)]
      [Neuron 0: W_up_row_0 | W_down_col_0]
      [Neuron 1: W_up_row_1 | W_down_col_1]
      ...
    """
    
    def __init__(self, path: str):
        self.path = path
        self.index = {}  # neuron_id -> (offset, bundle_size)
        self.d_model = 0
        self.d_ffn = 0
    
    def save(self, W_up: np.ndarray, W_down: np.ndarray):
        """Save weights in neuron-indexed format."""
        self.d_ffn, self.d_model = W_up.shape
        bundle_size = self.d_model * 2 * W_up.dtype.itemsize  # up_row + down_col
        
        with open(self.path, 'wb') as f:
            # Reserve header space
            header_size = 16  # d_model(4) + d_ffn(4) + dtype_size(4) + padding(4)
            f.write(np.array([self.d_model, self.d_ffn, W_up.dtype.itemsize, 0], 
                             dtype=np.int32).tobytes())
            
            data_offset = header_size
            for i in range(self.d_ffn):
                self.index[i] = (data_offset, bundle_size)
                # Bundle: W_up row i + W_down column i (contiguous!)
                f.write(W_up[i].tobytes())
                f.write(W_down[:, i].tobytes())
                data_offset += bundle_size
        
        # Save index separately
        index_path = self.path + '.idx'
        with open(index_path, 'w') as f:
            json.dump({
                'd_model': self.d_model,
                'd_ffn': self.d_ffn,
                'index': {str(k): list(v) for k, v in self.index.items()}
            }, f)
    
    def load_neurons(self, neuron_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load only the specified neurons from disk.
        Returns (W_up_sparse, W_down_sparse) with only the requested rows/cols.
        """
        n = len(neuron_ids)
        W_up_sparse = np.empty((n, self.d_model), dtype=np.float32)
        W_down_sparse = np.empty((self.d_model, n), dtype=np.float32)
        
        bytes_per_row = self.d_model * 4  # float32
        
        with open(self.path, 'rb') as f:
            for local_idx, neuron_id in enumerate(neuron_ids):
                offset, _ = self.index[int(neuron_id)]
                f.seek(offset)
                
                # Read W_up row
                raw = f.read(bytes_per_row)
                W_up_sparse[local_idx] = np.frombuffer(raw, dtype=np.float32)
                
                # Read W_down column (stored contiguously after W_up row)
                raw = f.read(bytes_per_row)
                W_down_sparse[:, local_idx] = np.frombuffer(raw, dtype=np.float32)
        
        return W_up_sparse, W_down_sparse
    
    def load_index(self):
        """Load the index from disk."""
        index_path = self.path + '.idx'
        with open(index_path, 'r') as f:
            data = json.load(f)
        self.d_model = data['d_model']
        self.d_ffn = data['d_ffn']
        self.index = {int(k): tuple(v) for k, v in data['index'].items()}


# ─── Full Sparse Streaming Engine ────────────────────────────────────
class GateTracerEngine:
    """
    Full gate-as-tracer inference engine.
    
    Architecture:
      - W_gate for ALL layers stays resident in RAM
      - W_up and W_down are stored on disk in neuron-indexed format
      - Per token, per layer:
          1. Compute gate activation (RAM)
          2. Identify fired neurons
          3. Load only fired W_up rows + W_down cols from disk
          4. Compute sparse FFN output
    """
    
    def __init__(self, config: ModelConfig, storage_dir: str):
        self.config = config
        self.storage_dir = storage_dir
        self.layers: List[Dict] = []
        self.stores: List[NeuronIndexedStore] = []
    
    def build_from_layers(self, ffn_layers: List[SwiGLU_FFN]):
        """Save model to disk in neuron-indexed format, keep gates in RAM."""
        os.makedirs(self.storage_dir, exist_ok=True)
        
        for i, layer in enumerate(ffn_layers):
            # Gate stays in RAM
            self.layers.append({
                'W_gate': layer.W_gate,  # resident!
                'd_model': layer.d_model,
                'd_ffn': layer.d_ffn,
            })
            
            # W_up and W_down go to disk in neuron-indexed format
            store = NeuronIndexedStore(
                os.path.join(self.storage_dir, f'layer_{i}.bin')
            )
            store.save(layer.W_up, layer.W_down)
            self.stores.append(store)
    
    def forward_one_layer(self, x: np.ndarray, layer_idx: int,
                          threshold: float = 0.1) -> Tuple[np.ndarray, Dict]:
        """Run one layer with gate-as-tracer sparse streaming."""
        layer = self.layers[layer_idx]
        store = self.stores[layer_idx]
        
        # ── Phase 1: TRACE ──
        t0 = time.perf_counter()
        gate_pre = layer['W_gate'] @ x
        gate_act = SwiGLU_FFN.silu(gate_pre)
        mask = np.abs(gate_act) > threshold
        fired_indices = np.where(mask)[0]
        t_trace = time.perf_counter() - t0
        
        # ── Phase 2: STREAM + COMPUTE ──
        t0 = time.perf_counter()
        W_up_sparse, W_down_sparse = store.load_neurons(fired_indices)
        t_load = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        gate_sparse = gate_act[fired_indices]
        up_sparse = W_up_sparse @ x
        hidden_sparse = gate_sparse * up_sparse
        output = W_down_sparse @ hidden_sparse
        t_compute = time.perf_counter() - t0
        
        n_fired = len(fired_indices)
        sparsity = 1.0 - (n_fired / layer['d_ffn'])
        
        # I/O stats
        bytes_loaded = n_fired * layer['d_model'] * 4 * 2  # up_row + down_col
        bytes_dense  = layer['d_ffn'] * layer['d_model'] * 4 * 2
        
        stats = {
            'n_fired': n_fired,
            'n_total': layer['d_ffn'],
            'sparsity': sparsity,
            'time_trace_ms': t_trace * 1000,
            'time_load_ms': t_load * 1000,
            'time_compute_ms': t_compute * 1000,
            'bytes_loaded': bytes_loaded,
            'bytes_dense': bytes_dense,
            'io_reduction': 1.0 - (bytes_loaded / bytes_dense),
        }
        return output, stats


# ─── Benchmarking ────────────────────────────────────────────────────
def benchmark_accuracy(d_model=512, d_ffn=2048, n_tests=20):
    """Compare dense vs sparse output to verify correctness."""
    print("=" * 65)
    print("  ACCURACY TEST: Dense vs Gate-Traced Sparse")
    print("=" * 65)
    
    layer = SwiGLU_FFN(d_model, d_ffn, seed=42)
    
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    print(f"\n  d_model={d_model}, d_ffn={d_ffn}, n_tests={n_tests}")
    print(f"  {'Threshold':>10} {'Sparsity':>10} {'Cos Sim':>10} {'Rel Error':>12} {'Max Err':>10}")
    print("  " + "-" * 56)
    
    rng = np.random.RandomState(123)
    results = []
    
    for thresh in thresholds:
        cos_sims, rel_errors, max_errors, sparsities = [], [], [], []
        
        for _ in range(n_tests):
            x = rng.randn(d_model).astype(np.float32) * 0.1
            
            out_dense = layer.forward_dense(x)
            out_sparse, stats = layer.forward_sparse(x, threshold=thresh)
            
            # Cosine similarity
            cos_sim = np.dot(out_dense, out_sparse) / (
                np.linalg.norm(out_dense) * np.linalg.norm(out_sparse) + 1e-10)
            
            # Relative error
            rel_err = np.linalg.norm(out_dense - out_sparse) / (
                np.linalg.norm(out_dense) + 1e-10)
            
            max_err = np.max(np.abs(out_dense - out_sparse))
            
            cos_sims.append(cos_sim)
            rel_errors.append(rel_err)
            max_errors.append(max_err)
            sparsities.append(stats['sparsity'])
        
        avg_cos = np.mean(cos_sims)
        avg_rel = np.mean(rel_errors)
        avg_max = np.mean(max_errors)
        avg_spar = np.mean(sparsities)
        
        results.append({
            'threshold': thresh,
            'sparsity': avg_spar,
            'cos_sim': avg_cos,
            'rel_error': avg_rel,
            'max_error': avg_max,
        })
        
        print(f"  {thresh:>10.2f} {avg_spar:>9.1%} {avg_cos:>10.6f} {avg_rel:>11.6f} {avg_max:>10.6f}")
    
    return results


def benchmark_io_savings(d_model=512, d_ffn=2048, n_tokens=50):
    """Measure I/O reduction from gate-as-tracer approach."""
    print("\n" + "=" * 65)
    print("  I/O SAVINGS BENCHMARK")
    print("=" * 65)
    
    layer = SwiGLU_FFN(d_model, d_ffn, seed=42)
    rng = np.random.RandomState(456)
    
    threshold = 0.1
    total_bytes_sparse = 0
    total_bytes_dense = 0
    sparsities = []
    
    for _ in range(n_tokens):
        x = rng.randn(d_model).astype(np.float32) * 0.1
        _, stats = layer.forward_sparse(x, threshold=threshold)
        
        n_fired = stats['n_fired']
        # Bytes that would be loaded: W_up rows + W_down cols for fired neurons
        bytes_sparse = n_fired * d_model * 4 * 2  # float32, both matrices
        bytes_dense = d_ffn * d_model * 4 * 2
        
        total_bytes_sparse += bytes_sparse
        total_bytes_dense += bytes_dense
        sparsities.append(stats['sparsity'])
    
    avg_sparsity = np.mean(sparsities)
    reduction = 1.0 - (total_bytes_sparse / total_bytes_dense)
    
    print(f"\n  Config: d_model={d_model}, d_ffn={d_ffn}, threshold={threshold}")
    print(f"  Tokens processed: {n_tokens}")
    print(f"  Average sparsity: {avg_sparsity:.1%}")
    print(f"  Dense I/O per token: {total_bytes_dense/n_tokens/1024:.1f} KB")
    print(f"  Sparse I/O per token: {total_bytes_sparse/n_tokens/1024:.1f} KB")
    print(f"  I/O reduction: {reduction:.1%}")
    print(f"  Effective speedup: {1/(1-reduction):.1f}x less data to transfer")
    
    return {
        'avg_sparsity': avg_sparsity,
        'reduction': reduction,
        'dense_bytes_per_token': total_bytes_dense / n_tokens,
        'sparse_bytes_per_token': total_bytes_sparse / n_tokens,
    }


def benchmark_disk_streaming(d_model=512, d_ffn=2048, n_layers=4, n_tokens=10):
    """End-to-end benchmark with actual disk I/O using neuron-indexed storage."""
    print("\n" + "=" * 65)
    print("  DISK STREAMING BENCHMARK (Neuron-Indexed Storage)")
    print("=" * 65)
    
    # Build layers
    layers = [SwiGLU_FFN(d_model, d_ffn, seed=i) for i in range(n_layers)]
    
    # Build engine (saves to disk)
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = GateTracerEngine(
            ModelConfig("test", d_model, d_ffn, n_layers),
            tmpdir
        )
        engine.build_from_layers(layers)
        
        # Check disk usage
        total_disk = sum(
            os.path.getsize(os.path.join(tmpdir, f))
            for f in os.listdir(tmpdir) if not f.endswith('.idx')
        )
        
        print(f"\n  Config: {n_layers} layers, d_model={d_model}, d_ffn={d_ffn}")
        print(f"  Storage on disk: {total_disk/1024/1024:.1f} MB")
        print(f"  W_gate in RAM: {n_layers * d_model * d_ffn * 4 / 1024/1024:.1f} MB")
        
        # Run inference
        rng = np.random.RandomState(789)
        threshold = 0.1
        
        all_stats = []
        
        for t in range(n_tokens):
            x = rng.randn(d_model).astype(np.float32) * 0.1
            token_stats = []
            
            for l in range(n_layers):
                x_out, stats = engine.forward_one_layer(x, l, threshold)
                token_stats.append(stats)
                # In a real model, there'd be attention + residual here
                # We just chain FFN outputs for demo
                x = x + x_out  # residual
            
            all_stats.append(token_stats)
        
        # Aggregate results
        print(f"\n  Per-layer averages over {n_tokens} tokens:")
        print(f"  {'Layer':>6} {'Sparsity':>10} {'Fired':>8} {'Trace ms':>10} {'Load ms':>10} {'Compute ms':>11} {'I/O Red.':>10}")
        print("  " + "-" * 68)
        
        for l in range(n_layers):
            layer_stats = [all_stats[t][l] for t in range(n_tokens)]
            avg = lambda key: np.mean([s[key] for s in layer_stats])
            
            print(f"  {l:>6} {avg('sparsity'):>9.1%} "
                  f"{avg('n_fired'):>7.0f} "
                  f"{avg('time_trace_ms'):>9.2f} "
                  f"{avg('time_load_ms'):>9.2f} "
                  f"{avg('time_compute_ms'):>10.2f} "
                  f"{avg('io_reduction'):>9.1%}")
        
        # Total summary
        total_trace = sum(np.mean([all_stats[t][l]['time_trace_ms'] 
                                   for t in range(n_tokens)]) for l in range(n_layers))
        total_load = sum(np.mean([all_stats[t][l]['time_load_ms'] 
                                  for t in range(n_tokens)]) for l in range(n_layers))
        total_compute = sum(np.mean([all_stats[t][l]['time_compute_ms'] 
                                     for t in range(n_tokens)]) for l in range(n_layers))
        avg_io_red = np.mean([all_stats[t][l]['io_reduction'] 
                              for t in range(n_tokens) for l in range(n_layers)])
        
        print(f"\n  Total per token ({n_layers} layers):")
        print(f"    Trace:   {total_trace:.2f} ms")
        print(f"    Load:    {total_load:.2f} ms")
        print(f"    Compute: {total_compute:.2f} ms")
        print(f"    Total:   {total_trace + total_load + total_compute:.2f} ms")
        print(f"    Avg I/O reduction: {avg_io_red:.1%}")
        
        return all_stats


def project_real_models():
    """Project savings for real model configurations."""
    print("\n" + "=" * 65)
    print("  PROJECTED SAVINGS FOR REAL MODELS")
    print("=" * 65)
    
    # Measure sparsity on a representative layer
    layer = SwiGLU_FFN(512, 2048, seed=42)
    rng = np.random.RandomState(999)
    sparsities = []
    for _ in range(100):
        x = rng.randn(512).astype(np.float32) * 0.1
        _, stats = layer.forward_sparse(x, threshold=0.1)
        sparsities.append(stats['sparsity'])
    measured_sparsity = np.mean(sparsities)
    
    # Use literature values for ReLU-fied models
    sparsity_scenarios = {
        "SwiGLU (threshold=0.1)": 0.50,  # CATS-like
        "ReLU-fied (ProSparse)": 0.90,
        "dReLU (TurboSparse)": 0.95,
    }
    
    storage_speeds = {
        "NVMe Gen4": 7.0,     # GB/s
        "NVMe Gen5": 14.0,
        "CPU RAM (DDR5)": 50.0,
    }
    
    for model_name, config in MODELS.items():
        print(f"\n  ┌─ {model_name} ({config.total_model_gb:.0f}GB total, "
              f"FFN={config.total_ffn_bytes/1024**3:.1f}GB)")
        print(f"  │  W_gate resident: {config.bytes_gate * config.n_layers / 1024**3:.1f}GB "
              f"({config.bytes_gate * config.n_layers / config.total_ffn_bytes:.0%} of FFN)")
        
        for scenario, sparsity in sparsity_scenarios.items():
            active_fraction = 1.0 - sparsity
            io_per_token = (config.bytes_up + config.bytes_down) * active_fraction * config.n_layers
            io_dense = (config.bytes_up + config.bytes_down) * config.n_layers
            
            print(f"  │")
            print(f"  ├─ {scenario} (sparsity={sparsity:.0%})")
            print(f"  │    I/O per token: {io_per_token/1024**2:.1f} MB "
                  f"(vs {io_dense/1024**2:.0f} MB dense)")
            
            for storage, speed_gb in storage_speeds.items():
                speed_bytes = speed_gb * 1024**3
                time_sparse = io_per_token / speed_bytes * 1000  # ms
                time_dense = io_dense / speed_bytes * 1000
                tokens_per_sec = 1000.0 / time_sparse if time_sparse > 0 else float('inf')
                
                print(f"  │    {storage}: {time_sparse:.1f}ms/token "
                      f"({tokens_per_sec:.1f} tok/s) "
                      f"vs dense {time_dense:.0f}ms")
        
        print(f"  └─")


# ─── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  GATE-AS-TRACER: Sparse Streaming LLM Inference Prototype   ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    # 1. Accuracy test
    accuracy_results = benchmark_accuracy(d_model=512, d_ffn=2048, n_tests=50)
    
    # 2. I/O savings
    io_results = benchmark_io_savings(d_model=512, d_ffn=2048, n_tokens=100)
    
    # 3. Disk streaming
    disk_results = benchmark_disk_streaming(d_model=512, d_ffn=2048, n_layers=4, n_tokens=20)
    
    # 4. Projections for real models
    project_real_models()
    
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print("""
  Key findings:
  
  1. ZERO predictor needed — the gate IS the tracer
  2. W_gate stays in RAM (only 1/3 of FFN parameters)
  3. Full-precision computation — no quantization
  4. I/O reduction scales with sparsity:
     - 50% sparsity (SwiGLU threshold) → 2x less I/O
     - 90% sparsity (ReLU-fied)       → 10x less I/O  
     - 95% sparsity (dReLU)           → 20x less I/O
  5. Neuron-indexed storage enables sequential reads
     (row-column bundling per neuron)
  
  Novel contribution vs prior work:
  - PowerInfer, LLM in a Flash: use TRAINED PREDICTORS → ours uses gate
  - CATS, MoC: use gate for COMPUTE savings → ours uses gate for I/O
  - This combines both insights: gate as tracer + disk streaming
""")
