# zeroclawgpt 🦀⚡

**Zero dependencies. Single file. Pure Rust. ~4,500x faster than Python.**

A faithful port of [Karpathy's microGPT.py](https://github.com/karpathy/microgpt) — a tiny GPT that trains on baby names and generates new ones. No `ndarray`, no `rand`, no crates at all. Just `std`.

## Results

| Metric | Karpathy Python | zeroclawgpt |
|---|---|---|
| Training time (1000 steps) | 297.7s | **0.065s** |
| **Speedup** | 1x | **~4,580x** |
| Parameters | 3,632 | 3,632 |
| Final loss (5000 steps) | ~2.4 | **0.69** |
| Generated samples | `cagaan, lavoen` | **`ryan, ava, naomi, luke`** |
| Dependencies | numpy-ish autograd | **zero** |

## Quick Start

```bash
cargo build --release
./target/release/zeroclawgpt
```

Output:
```
zeroclawgpt v2  vocab=27  params=3632  layers=1  embd=16  heads=4
Fixes: KV-cache causal attn | LR linear decay | beta2=0.95 | zero-init wo/fc2
Training 5000 steps

step     0  loss=3.2943  t=0.00s  | sample 0: <random>
step  2500  loss=2.1965  t=0.16s  | sample 1: logan  ...
step  4000  loss=1.2236  t=0.25s  | sample 0: jaden  sample 1: caleb  ...
step  4999  loss=0.6869  t=0.32s  | sample 0: naomi  sample 2: eleanora  ...

Done in 0.319s
```

## What's Inside

A complete GPT implementation in ~475 lines of Rust:

- **Transformer** — single-layer, 4-head causal self-attention with KV cache
- **Training** — analytical gradients (no autograd), Adam optimizer with linear LR decay
- **PRNG** — xoshiro128+ with Box-Muller gaussian sampling
- **Inference** — autoregressive generation with categorical sampling

### Architecture

```
token + position embeddings
        │
  ┌─────▼──────────────────────┐
  │  Transformer Block ×1      │
  │  RMSNorm → Multi-Head Attn │  ← causal KV cache
  │  + residual                │
  │  RMSNorm → MLP (SqReLU)   │
  │  + residual                │
  └────────────────────────────┘
        │
  weight-tied logits → softmax
```

### Hyperparameters

| Param | Value |
|---|---|
| `N_EMBD` | 16 |
| `N_HEAD` | 4 |
| `N_LAYER` | 1 |
| `BLOCK_SIZE` | 8 |
| `N_STEPS` | 5000 |
| `LR` | 1e-2 (linear decay) |
| `BETA1 / BETA2` | 0.9 / 0.95 |

## 5 Bugs Fixed vs Naive Port

Discovered by reading Karpathy's actual source:

1. **🔴 KV Cache** — v1 had no causal attention (each token only saw itself). Fixed: real KV cache that grows per position.
2. **🟡 Adam beta2** — 0.999 → 0.95 (matches Karpathy, faster convergence on tiny data)
3. **🟡 LR Decay** — constant → linear decay to 0 (prevents late-stage overshooting)
4. **🟡 Zero Init** — output projections (Wo, Wfc2) zero-initialized (GPT-2 scaled init)
5. **🟢 Loss Normalization** — gradients now scaled by 1/(seq_len-1) before backward pass

These fixes are the difference between generating `ioeanaa` and generating `naomi`.

## Why Zero Dependencies?

Karpathy's Python version uses a custom scalar autograd — every float is a heap-allocated `Value` object with a backward closure. At 3,632 parameters, one forward pass creates tens of thousands of nodes.

We implement analytical matrix-level gradients instead. Same math, zero allocations, and it turns out to be simpler to read once you see the pattern.

The PRNG (xoshiro128+) is 15 lines. Python's `random` module is ~2,000 lines of C that nobody counts.

## License

MIT
