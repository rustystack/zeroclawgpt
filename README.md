# zeroclawgpt 🦀⚡

> Zero-dependency GPT in pure Rust — a faithful port of Karpathy's microGPT.py that trains ~4,500x faster.

A complete, from-scratch GPT implementation in a single Rust file. No crates. No `ndarray`. No `rand`. No `serde`. Just `std` — and it generates real human names after training for less than a second.

```
step     0  loss=3.2943  | m<BOS>kv<BOS>tpl  kygocwfv  vydcdhlm
step  2500  loss=2.1965  | logan  leo  lagan
step  4999  loss=0.6869  | naomi  eleanora  ryan

Done in 0.689s
```

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Hyperparameters](#hyperparameters)
- [Training Output](#training-output)
- [Performance](#performance)
- [5 Bugs We Fixed](#5-bugs-we-fixed)
- [Why Zero Dependencies](#why-zero-dependencies)
- [Code Tour](#code-tour)
- [Extending It](#extending-it)
- [License](#license)

---

## Why This Exists

Andrej Karpathy released [microGPT.py](https://github.com/karpathy/microgpt) — a minimal GPT that fits in ~200 lines of Python with a custom scalar autograd engine. It trains on a list of baby names and learns to generate new ones.

We ported it to Rust. Faithfully. Then we read his actual source code carefully and found 5 meaningful differences in our implementation. Fixing them took us from generating `ioeanaa` to generating `naomi`.

The result: **474 lines of Rust, zero dependencies, 3,632 parameters, and a ~4,500x speedup over Python.**

---

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) (1.56+ for edition 2021)

### Build & Run

```bash
git clone https://github.com/rustystack/zeroclawgpt.git
cd zeroclawgpt
cargo build --release
./target/release/zeroclawgpt
```

That's it. No data files to download — the training data (92 baby names) is embedded in the source.

### What You'll See

```
zeroclawgpt v2  vocab=27  params=3632  layers=1  embd=16  heads=4
Fixes: KV-cache causal attn | LR linear decay | beta2=0.95 | zero-init wo/fc2
Training 5000 steps

step     0  loss=3.2943  t=0.00s  | sample 0: m<BOS>kv<BOS>tpl  sample 1: kygocwfv  ...
step   500  loss=2.3344  t=0.07s  | sample 0: eyy  sample 1: iyne  ...
step  1000  loss=2.3421  t=0.14s  | sample 0: rarloeba  sample 1: alievy  ...
step  2000  loss=1.6842  t=0.27s  | sample 0: lmye  sample 1: rync  sample 2: luke  ...
step  3000  loss=1.0547  t=0.41s  | sample 0: carey  sample 1: gadrel  ...
step  4000  loss=1.2236  t=0.55s  | sample 0: jaden  sample 1: caleb  sample 2: axel  ...
step  4999  loss=0.6869  t=0.69s  | sample 0: naomi  sample 1: eleanora  sample 2: ryan  ...

Done in 0.689s
```

Watch the samples evolve from random noise → plausible letter combos → real names.

---

## How It Works

### The Task

Given a dataset of 92 names (`emma`, `oliver`, `luna`, `axel`, ...), train a tiny transformer to predict the next character. At inference time, feed it a `<BOS>` (beginning of sequence) token and let it generate characters one by one until it produces `<EOS>` (end of sequence).

### The Training Loop

Each of the 5,000 training steps:

1. **Pick a name** — cycle through the 92 names round-robin
2. **Tokenize** — convert to `[<BOS>, c1, c2, ..., <EOS>]` (character-level, max 8 tokens)
3. **Forward** — process each token position through the transformer, building up a KV cache so each position attends to all previous positions (causal attention)
4. **Loss** — cross-entropy: for each position, how surprised was the model by the actual next character?
5. **Backward** — compute analytical gradients for every parameter (no autograd graph)
6. **Update** — Adam optimizer with linear learning rate decay

### Inference

Generation is autoregressive:

```
Input:  <BOS>
Step 1: <BOS> → predict 'n' → "n"
Step 2: <BOS>, n → predict 'a' → "na"
Step 3: <BOS>, n, a → predict 'o' → "nao"
Step 4: <BOS>, n, a, o → predict 'm' → "naom"
Step 5: <BOS>, n, a, o, m → predict 'i' → "naomi"
Step 6: <BOS>, n, a, o, m, i → predict <EOS> → done!
```

Each step, the model sees all previous characters via the KV cache and predicts the next one.

---

## Architecture

```
token_id ──→ wte[tok]  ─┐
                         ├──→ x = tok_emb + pos_emb
pos_id   ──→ wpe[pos]  ─┘
                │
      ┌────────▼─────────────────────────────┐
      │       Transformer Block ×1           │
      │                                      │
      │  x_res = x                           │
      │  x = RMSNorm(x)                      │
      │                                      │
      │  Q = x @ Wq                          │
      │  K = x @ Wk  ──→ append to KV cache  │
      │  V = x @ Wv  ──→ append to KV cache  │
      │                                      │
      │  ┌─ For each of 4 heads: ──────────┐ │
      │  │  scores = Q·K^T / √d_head       │ │
      │  │  weights = softmax(scores)       │ │
      │  │  head_out = weights · V          │ │
      │  └─────────────────────────────────┘ │
      │                                      │
      │  attn_out = concat(heads) @ Wo       │
      │  x = x_res + attn_out               │
      │                                      │
      │  x_res = x                           │
      │  x = RMSNorm(x)                      │
      │  x = x @ Wfc1                        │
      │  x = squared_relu(x)                 │
      │  x = x @ Wfc2                        │
      │  x = x_res + x                       │
      └──────────────────────────────────────┘
                │
      logits = x @ Wte^T        (weight-tied with token embeddings)
                │
      probs  = softmax(logits)
                │
      loss   = -log(probs[target]) / seq_len
```

Key design choices (matching Karpathy's implementation):
- **RMSNorm** instead of LayerNorm (no bias, no learnable scale)
- **Squared ReLU** activation in the MLP (`max(0, x)²`)
- **Weight tying** — the output projection reuses the token embedding matrix
- **Zero-initialized** output projections (Wo, Wfc2) — residual stream starts as identity

---

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `N_EMBD` | 16 | Embedding dimension |
| `N_HEAD` | 4 | Number of attention heads |
| `HEAD_DIM` | 4 | Per-head dimension (`N_EMBD / N_HEAD`) |
| `N_LAYER` | 1 | Number of transformer blocks |
| `BLOCK_SIZE` | 8 | Maximum sequence length |
| `N_STEPS` | 5000 | Training iterations |
| `LR` | 1e-2 | Initial learning rate |
| `BETA1` | 0.9 | Adam first moment decay |
| `BETA2` | 0.95 | Adam second moment decay |
| LR schedule | linear decay | `LR × (1 - step/N_STEPS)` → decays to 0 |

All values match Karpathy's defaults exactly.

---

## Training Output

The model generates 5 samples every 500 steps. Here's what the learning progression looks like:

| Step | Loss | Sample Names | What's Happening |
|---|---|---|---|
| 0 | 3.29 | `m<BOS>kv<BOS>tpl` | Random noise — loss ≈ -log(1/27) as expected |
| 500 | 2.33 | `eyy`, `iyne`, `adee` | Learning vowel/consonant patterns |
| 1000 | 2.34 | `rarloeba`, `alievy` | Longer sequences, still garbled |
| 2000 | 1.68 | `luke`, `rync` | First real name appears! |
| 2500 | 2.20 | `logan`, `leo` | More real names emerging |
| 3500 | 1.24 | `cole`, `kole` | Variations on learned patterns |
| 4000 | 1.22 | `jaden`, `caleb`, `axel` | Consistent real names |
| 4999 | 0.69 | `naomi`, `eleanora`, `ryan` | Strong generation quality |

---

## Performance

| Metric | Karpathy Python | zeroclawgpt |
|---|---|---|
| 1000-step training time | 297.7s | **0.065s** |
| Full 5000-step run | ~25 min | **< 1s** |
| Speedup | 1× | **~4,580×** |
| Parameters | 3,632 | 3,632 |
| Final loss | ~2.4 (1000 steps) | **0.69** (5000 steps) |
| Memory allocations per step | Tens of thousands | Minimal |

### Why So Fast?

Karpathy's Python builds a **dynamic computation graph** — every scalar `float` is wrapped in a `Value` object with a `_backward` closure for autograd. At 3,632 parameters, one forward pass creates tens of thousands of heap-allocated nodes that must be topologically sorted and traversed for backprop.

We implement **analytical matrix-level gradients** — the same math, computed directly:

| Operation | Python (autograd) | Rust (analytical) |
|---|---|---|
| `c = a + b` | Allocate node + closure | `dc = 1`, applied inline |
| `softmax → CE` | Chain of exp/sum/log nodes | `d_logits = probs - one_hot` |
| RMSNorm backward | Graph traversal | 4-line function |
| Full attention backward | Thousands of nodes | Direct matrix ops |

---

## 5 Bugs We Fixed

We started with a naive port, then read Karpathy's actual source line by line. Five differences emerged:

### 1. 🔴 KV Cache — Real Causal Attention (Critical)

**The bug:** Our v1 processed each position independently. Token at position 3 could only attend to *itself* — not tokens 0, 1, 2. This is fundamentally not a language model.

**The fix:** Accumulate keys and values into a growing cache. At position `t`, the model attends over all positions `[0..t]`, exactly like Karpathy's implementation.

**Impact:** This is the difference between `ioeanaa` and `naomi`.

### 2. 🟡 Adam beta2: 0.999 → 0.95

Lower `beta2` means the optimizer's second moment estimate adapts faster — it forgets old gradient magnitudes more quickly. On a tiny dataset with few training steps, this converges noticeably faster.

### 3. 🟡 Linear LR Decay

**The bug:** Constant learning rate throughout training.

**The fix:** `lr = LR × (1 - step/N_STEPS)`, decaying linearly to zero. Prevents overshooting near convergence.

### 4. 🟡 Zero-Init Output Projections

**The bug:** `Wo` and `Wfc2` initialized with `std=0.02` like other weights.

**The fix:** Initialize to zero. This means at step 0, both the attention and MLP blocks contribute *nothing* — the residual stream is pure identity. The model only starts deviating as gradients flow in. This is the GPT-2 "scaled initialization" technique.

### 5. 🟢 Loss Normalization

**The bug:** Gradients were `(seq_len-1)×` too large — we normalized loss for display but not before the backward pass.

**The fix:** Scale `d_logits` by `1/(seq_len-1)` before backpropagation, matching Karpathy's normalization.

---

## Why Zero Dependencies

This isn't just a flex. It's the point.

Karpathy's microGPT uses no ML frameworks — no PyTorch, no JAX. Just a tiny autograd engine that fits in the same file. The beauty is seeing every piece of a GPT laid bare with nothing hidden.

We carry that philosophy to Rust:

- **PRNG** — xoshiro128+ in 15 lines, with Box-Muller gaussian sampling
- **Linear algebra** — row-major matrix multiply, element-wise ops
- **Optimizer** — Adam with bias correction, implemented directly
- **Gradients** — analytical, not autograd. Every backward function is hand-derived

Python's `random` module is ~2,000 lines of C that Karpathy doesn't count. We don't count our 15-line PRNG either. Fair's fair.

**The entire model — forward pass, backward pass, optimizer, data loading, inference — is one file you can read top to bottom in 20 minutes.**

---

## Code Tour

The source (`src/main.rs`) is organized in sections:

| Lines | Section | What It Does |
|---|---|---|
| 1–10 | Header | Constants, imports |
| 11–18 | Hyperparameters | All tuneable values in one place |
| 20–50 | PRNG | xoshiro128+ RNG, gaussian sampling, categorical sampler |
| 52–85 | Matrix ops | `linear()`, `softmax()`, `rmsnorm()` and their backward passes |
| 87–140 | Model struct | Parameter storage, gradient buffers, Adam optimizer |
| 142–180 | Activation cache | Per-position saved state for backward pass |
| 182–260 | Forward pass | Embeddings → attention with KV cache → MLP → logits |
| 262–360 | Backward pass | Analytical gradients through every operation |
| 362–395 | Data & vocab | Baby names dataset, character-level tokenizer |
| 397–420 | Inference | Autoregressive generation with KV cache |
| 422–475 | Main | Training loop with logging |

### Key functions

- **`forward()`** — processes one token position, appends to KV cache, returns cached activations
- **`backward()`** — computes gradients for one position, accumulates cross-position KV gradients via `d_kv_cache`
- **`rmsnorm()` / `rmsnorm_bwd()`** — forward and backward for RMS normalization
- **`linear()` / `linear_bwd_w()` / `linear_bwd_x()`** — matrix multiply and its two gradient components
- **`generate()`** — autoregressive sampling with fresh KV cache per name

---

## Extending It

Ideas for building on this (roughly ordered by complexity):

| Extension | Difficulty | Description |
|---|---|---|
| CLI arguments | Easy | Add `--steps`, `--lr`, `--layers` via `std::env::args()` |
| Fetch `names.txt` | Easy | Download Karpathy's full dataset via `std::net::TcpStream` (still zero deps) |
| Checkpoint save/load | Easy | Write/read raw `f32` bytes via `std::fs` |
| Batched training | Medium | `[B, seq, embd]` tensors to amortize fixed overhead |
| SIMD matmul | Medium | `#[target_feature(enable = "avx2")]` on `linear()`, expect 4-8× speedup |
| Gradient checkpointing | Medium | Recompute activations in backward for larger `N_LAYER` |
| Multi-layer scaling | Hard | Test with `N_EMBD=64`, `N_LAYER=4` — verify correctness at scale |

---

## License

[MIT](LICENSE)

---

<p align="center">
  Built by <a href="https://github.com/rustystack">rustystack</a> 🦀
</p>
