// zeroclawgpt in Rust — zero dependencies, single file
// Faithful port of Karpathy's zeroclawgpt.py (Feb 11 2026)
// Fixes vs v1: KV cache (real causal attention), LR linear decay,
//              beta2=0.95, zero-init attn_wo+mlp_fc2, correct loss normalization

use std::collections::HashMap;
use std::io::{self, Write};
use std::time::Instant;

// ─── Hyperparameters (mirrors Karpathy's defaults) ────────────────────────────
const N_EMBD:    usize = 16;
const N_HEAD:    usize = 4;
const N_LAYER:   usize = 1;
const BLOCK_SIZE:usize = 8;
const N_STEPS:   usize = 5000;
const LR:        f32   = 1e-2;
const BETA1:     f32   = 0.9;
const BETA2:     f32   = 0.95;   // matches Karpathy (was 0.999)
const EPS_ADAM:  f32   = 1e-8;
const HEAD_DIM:  usize = N_EMBD / N_HEAD;

// ─── Tiny PRNG (xoshiro128+, zero dep) ───────────────────────────────────────
struct Rng { s: [u32; 4] }

impl Rng {
    fn new(seed: u64) -> Self {
        let lo = seed as u32;
        let hi = (seed >> 32) as u32;
        Self { s: [lo ^ 0xdeadbeef, hi ^ 0xcafebabe, lo.wrapping_add(1), hi.wrapping_add(1)] }
    }
    fn next_u32(&mut self) -> u32 {
        let r = self.s[0].wrapping_add(self.s[3]);
        let t = self.s[1] << 9;
        self.s[2] ^= self.s[0]; self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2]; self.s[0] ^= self.s[3];
        self.s[2] ^= t; self.s[3] = self.s[3].rotate_left(11);
        r
    }
    fn gauss(&mut self, std: f32) -> f32 {
        let u1 = (self.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0);
        let u2 = self.next_u32() as f32 / u32::MAX as f32;
        std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    fn categorical(&mut self, probs: &[f32]) -> usize {
        let mut dart = (self.next_u32() as f32 / u32::MAX as f32) * probs.iter().sum::<f32>();
        for (i, &p) in probs.iter().enumerate() {
            dart -= p;
            if dart <= 0.0 { return i; }
        }
        probs.len() - 1
    }
}

// ─── Matrix / vector helpers (row-major, [rows × cols]) ──────────────────────
fn mat_rand(rows: usize, cols: usize, rng: &mut Rng, std: f32) -> Vec<f32> {
    (0..rows * cols).map(|_| rng.gauss(std)).collect()
}
fn zeros(n: usize) -> Vec<f32> { vec![0.0f32; n] }

/// y = x @ W^T   x:[in]  W:[out,in]  → y:[out]
fn linear(x: &[f32], w: &[f32], out: usize, inp: usize) -> Vec<f32> {
    (0..out).map(|o| x.iter().zip(&w[o*inp..(o+1)*inp]).map(|(a,b)| a*b).sum()).collect()
}
fn linear_bwd_w(dw: &mut [f32], dy: &[f32], x: &[f32], out: usize, inp: usize) {
    for o in 0..out { for i in 0..inp { dw[o*inp+i] += dy[o]*x[i]; } }
}
fn linear_bwd_x(dx: &mut [f32], dy: &[f32], w: &[f32], out: usize, inp: usize) {
    for i in 0..inp { dx[i] += (0..out).map(|o| dy[o]*w[o*inp+i]).sum::<f32>(); }
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ex: Vec<f32> = x.iter().map(|v| (v-m).exp()).collect();
    let s: f32 = ex.iter().sum();
    ex.iter().map(|v| v/s).collect()
}

/// Returns (normed_x, rms_inv)
fn rmsnorm(x: &[f32]) -> (Vec<f32>, f32) {
    let ms: f32 = x.iter().map(|v| v*v).sum::<f32>() / x.len() as f32;
    let ri = (ms + 1e-5_f32).sqrt().recip();
    (x.iter().map(|v| v*ri).collect(), ri)
}
fn rmsnorm_bwd(dy: &[f32], x: &[f32], ri: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let dot: f32 = dy.iter().zip(x).map(|(a,b)| a*b).sum();
    dy.iter().zip(x).map(|(dy_i, x_i)| ri*dy_i - (ri*ri*ri/n)*dot*x_i).collect()
}

// ─── Model ───────────────────────────────────────────────────────────────────
struct Model {
    w: HashMap<String, Vec<f32>>,
    g: HashMap<String, Vec<f32>>,
    m: HashMap<String, Vec<f32>>,
    v: HashMap<String, Vec<f32>>,
    vocab_size: usize,
}

impl Model {
    fn new(vs: usize, rng: &mut Rng) -> Self {
        let mut w: HashMap<String,Vec<f32>> = HashMap::new();
        let mut g = HashMap::new(); let mut m = HashMap::new(); let mut v = HashMap::new();
        macro_rules! p {
            ($n:expr, $r:expr, $c:expr, $s:expr) => {{
                let sz = $r * $c;
                w.insert($n.to_string(), mat_rand($r,$c,rng,$s));
                g.insert($n.to_string(), zeros(sz));
                m.insert($n.to_string(), zeros(sz));
                v.insert($n.to_string(), zeros(sz));
            }};
        }
        p!("wte",  vs,          N_EMBD,    0.02);
        p!("wpe",  BLOCK_SIZE,  N_EMBD,    0.02);
        for l in 0..N_LAYER {
            p!(format!("l{l}.wq"),  N_EMBD,     N_EMBD,    0.02);
            p!(format!("l{l}.wk"),  N_EMBD,     N_EMBD,    0.02);
            p!(format!("l{l}.wv"),  N_EMBD,     N_EMBD,    0.02);
            p!(format!("l{l}.wo"),  N_EMBD,     N_EMBD,    0.0 ); // zero init (matches Karpathy)
            p!(format!("l{l}.fc1"), 4*N_EMBD,   N_EMBD,    0.02);
            p!(format!("l{l}.fc2"), N_EMBD,     4*N_EMBD,  0.0 ); // zero init (matches Karpathy)
        }
        Self { w, g, m, v, vocab_size: vs }
    }
    fn zero_grad(&mut self) { self.g.values_mut().for_each(|g| g.iter_mut().for_each(|v| *v=0.0)); }
    fn param_count(&self) -> usize { self.w.values().map(|v| v.len()).sum() }

    /// Adam with linear LR decay: lr_t = LR * (1 - step/N_STEPS)
    fn adam_step(&mut self, step: usize) {
        let decay  = 1.0 - step as f32 / N_STEPS as f32;
        let lr_t   = LR * decay * (1.0-BETA2.powi(step as i32)).sqrt() / (1.0-BETA1.powi(step as i32));
        for key in self.w.keys().cloned().collect::<Vec<_>>() {
            let (w,g,m,v) = (
                self.w.get_mut(&key).unwrap(), self.g.get(&key).unwrap(),
                self.m.get_mut(&key).unwrap(), self.v.get_mut(&key).unwrap(),
            );
            for i in 0..w.len() {
                m[i] = BETA1*m[i] + (1.0-BETA1)*g[i];
                v[i] = BETA2*v[i] + (1.0-BETA2)*g[i]*g[i];
                w[i] -= lr_t * m[i] / (v[i].sqrt() + EPS_ADAM);
            }
        }
    }
}

// ─── Per-position activation cache ───────────────────────────────────────────
// We store activations for every position in the sequence so backward can
// propagate through the full causal attention over all prior K/V entries.
struct PosCache {
    tok_id: usize,
    pos_id: usize,
    x0:     Vec<f32>,          // embedding output
    layers: Vec<LCache>,
    probs:  Vec<f32>,
    // causal attn context saved per position for backward
    // attn_ctx[l] = (all_k up to this pos, all_v up to this pos, attn_weights)
    attn_ctx: Vec<AttnCtx>,
}

struct AttnCtx {
    // keys and values for ALL positions up to and including this one: [T, N_EMBD]
    all_k: Vec<Vec<f32>>,
    all_v: Vec<Vec<f32>>,
    // attention weights per head: [N_HEAD, T]
    aw: Vec<Vec<f32>>,
    // concatenated head outputs (before wo): [N_EMBD]
    ho: Vec<f32>,
}

struct LCache {
    x_pre_attn:  Vec<f32>, xn_attn: Vec<f32>, rms_a: f32,
    q:           Vec<f32>,
    attn_out:    Vec<f32>, x_post_attn: Vec<f32>,
    xn_mlp:      Vec<f32>, rms_m: f32,
    h1:          Vec<f32>, h1a: Vec<f32>,
    mlp_out:     Vec<f32>, x_post_mlp: Vec<f32>,
}

// ─── Forward pass (single position, with KV cache accumulation) ──────────────
// kv_cache[layer] = (accumulated_keys, accumulated_values)
// Each entry is Vec<[N_EMBD]> growing with each position processed.
fn forward(
    model: &Model,
    tok: usize, pos: usize,
    kv_cache: &mut Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)>,
) -> PosCache {
    let vs = model.vocab_size;
    let w  = &model.w;

    // Embeddings
    let te = &w["wte"][tok*N_EMBD..(tok+1)*N_EMBD];
    let pe = &w["wpe"][(pos%BLOCK_SIZE)*N_EMBD..(pos%BLOCK_SIZE+1)*N_EMBD];
    let mut x: Vec<f32> = te.iter().zip(pe).map(|(a,b)| a+b).collect();
    let x0 = x.clone();
    let mut layers  = Vec::with_capacity(N_LAYER);
    let mut attn_ctx= Vec::with_capacity(N_LAYER);

    for l in 0..N_LAYER {
        let pf = format!("l{l}");
        let x_pre_attn = x.clone();
        let (xn_attn, rms_a) = rmsnorm(&x);

        let q   = linear(&xn_attn, &w[&format!("{pf}.wq")], N_EMBD, N_EMBD);
        let k   = linear(&xn_attn, &w[&format!("{pf}.wk")], N_EMBD, N_EMBD);
        let v   = linear(&xn_attn, &w[&format!("{pf}.wv")], N_EMBD, N_EMBD);

        // Accumulate into KV cache (real causal attention — matches Karpathy)
        kv_cache[l].0.push(k);
        kv_cache[l].1.push(v);
        let all_k = &kv_cache[l].0;   // [T, N_EMBD]
        let all_v = &kv_cache[l].1;
        let t_len = all_k.len();
        let scale = (HEAD_DIM as f32).sqrt();

        let mut aw_all: Vec<Vec<f32>> = Vec::with_capacity(N_HEAD); // [N_HEAD, T]
        let mut ho = zeros(N_EMBD);

        for h in 0..N_HEAD {
            let qs = &q[h*HEAD_DIM..(h+1)*HEAD_DIM];
            // dot product with every past key
            let logits: Vec<f32> = (0..t_len).map(|t| {
                let ks = &all_k[t][h*HEAD_DIM..(h+1)*HEAD_DIM];
                qs.iter().zip(ks).map(|(a,b)| a*b).sum::<f32>() / scale
            }).collect();
            let aw_h = softmax(&logits);
            // weighted sum of values
            for i in 0..HEAD_DIM {
                let out_i: f32 = (0..t_len).map(|t| aw_h[t] * all_v[t][h*HEAD_DIM+i]).sum();
                ho[h*HEAD_DIM+i] = out_i;
            }
            aw_all.push(aw_h);
        }

        let attn_out = linear(&ho, &w[&format!("{pf}.wo")], N_EMBD, N_EMBD);
        let x_post_attn: Vec<f32> = x_pre_attn.iter().zip(&attn_out).map(|(a,b)| a+b).collect();
        x = x_post_attn.clone();

        let (xn_mlp, rms_m) = rmsnorm(&x);
        let h1: Vec<f32>  = linear(&xn_mlp, &w[&format!("{pf}.fc1")], 4*N_EMBD, N_EMBD);
        let h1a: Vec<f32> = h1.iter().map(|v| if *v>0.0 {v*v} else {0.0}).collect();
        let mlp_out       = linear(&h1a,   &w[&format!("{pf}.fc2")], N_EMBD, 4*N_EMBD);
        let x_post_mlp: Vec<f32> = x.iter().zip(&mlp_out).map(|(a,b)| a+b).collect();
        x = x_post_mlp.clone();

        attn_ctx.push(AttnCtx {
            all_k: all_k.clone(), all_v: all_v.clone(), aw: aw_all, ho,
        });
        layers.push(LCache {
            x_pre_attn, xn_attn, rms_a, q, attn_out, x_post_attn,
            xn_mlp, rms_m, h1, h1a, mlp_out, x_post_mlp,
        });
    }

    let logits = linear(&x, &w["wte"], vs, N_EMBD);
    let probs  = softmax(&logits);
    PosCache { tok_id: tok, pos_id: pos, x0, layers, probs, attn_ctx }
}

// ─── Backward (single position, causal attention) ─────────────────────────────
// d_kv_cache[l] = accumulated d_k and d_v errors from future positions
// that attended back to this position's keys/values.
fn backward(
    model: &mut Model,
    c: &PosCache,
    target: usize,
    seq_len: usize,                          // for loss normalization
    d_kv_cache: &mut Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)>, // [L][(dk_list, dv_list)]
) {
    let vs = model.vocab_size;
    let g  = &mut model.g;
    let norm = 1.0 / (seq_len - 1) as f32;  // matches Karpathy's per-step normalization

    // Cross-entropy gradient (scaled by normalization)
    let mut d_logits = c.probs.clone();
    d_logits[target] -= 1.0;
    d_logits.iter_mut().for_each(|v| *v *= norm);

    // lm_head (weight-tied wte)
    let last_x = if N_LAYER>0 { &c.layers[N_LAYER-1].x_post_mlp } else { &c.x0 };
    linear_bwd_w(g.get_mut("wte").unwrap(), &d_logits, last_x, vs, N_EMBD);
    let mut dx = zeros(N_EMBD);
    linear_bwd_x(&mut dx, &d_logits, &model.w["wte"], vs, N_EMBD);

    for l in (0..N_LAYER).rev() {
        let pf = format!("l{l}");
        let lc = &c.layers[l];
        let ac = &c.attn_ctx[l];
        let t_len = ac.all_k.len();      // number of positions processed so far
        let cur_t = t_len - 1;           // index of THIS position in the kv sequence

        // ── MLP residual ─────────────────────────────────────────────────────
        let dx_s = dx.clone();
        let mut d_h1a = zeros(4*N_EMBD);
        linear_bwd_w(g.get_mut(&format!("{pf}.fc2")).unwrap(), &dx, &lc.h1a, N_EMBD, 4*N_EMBD);
        linear_bwd_x(&mut d_h1a, &dx, &model.w[&format!("{pf}.fc2")], N_EMBD, 4*N_EMBD);
        let d_h1: Vec<f32> = d_h1a.iter().zip(&lc.h1).map(|(da,h)| if *h>0.0 {da*2.0*h} else {0.0}).collect();
        let mut d_xn_mlp = zeros(N_EMBD);
        linear_bwd_w(g.get_mut(&format!("{pf}.fc1")).unwrap(), &d_h1, &lc.xn_mlp, 4*N_EMBD, N_EMBD);
        linear_bwd_x(&mut d_xn_mlp, &d_h1, &model.w[&format!("{pf}.fc1")], 4*N_EMBD, N_EMBD);
        let d_rn_m = rmsnorm_bwd(&d_xn_mlp, &lc.x_post_attn, lc.rms_m);
        dx = dx_s.iter().zip(&d_rn_m).map(|(a,b)| a+b).collect();

        // ── Attention residual ────────────────────────────────────────────────
        let dx_s = dx.clone();
        let mut d_ho = zeros(N_EMBD);
        linear_bwd_w(g.get_mut(&format!("{pf}.wo")).unwrap(), &dx, &ac.ho, N_EMBD, N_EMBD);
        linear_bwd_x(&mut d_ho, &dx, &model.w[&format!("{pf}.wo")], N_EMBD, N_EMBD);

        // Per-head backward through weighted-sum and softmax
        // d_k and d_v are distributed back to ALL positions they came from
        let mut d_q = zeros(N_EMBD);
        for h in 0..N_HEAD {
            let aw_h  = &ac.aw[h];           // [T]
            let scale = (HEAD_DIM as f32).sqrt();
            // gradient into value vectors: d_v[t][i] += aw[t] * d_ho[h*HD+i]
            for t in 0..t_len {
                let dv = &mut d_kv_cache[l].1[t];
                for i in 0..HEAD_DIM { dv[h*HEAD_DIM+i] += aw_h[t] * d_ho[h*HEAD_DIM+i]; }
            }
            // gradient into attention weights: d_aw[t] = sum_i d_ho[h*HD+i] * v[t][i]
            let d_aw_logits_raw: Vec<f32> = (0..t_len).map(|t| {
                (0..HEAD_DIM).map(|i| d_ho[h*HEAD_DIM+i] * ac.all_v[t][h*HEAD_DIM+i]).sum::<f32>()
            }).collect();
            // softmax backward: d_logit[t] = aw[t]*(d_aw_raw[t] - sum_j aw[j]*d_aw_raw[j])
            let dot_aw: f32 = aw_h.iter().zip(&d_aw_logits_raw).map(|(a,b)| a*b).sum();
            let d_attn_logits: Vec<f32> = aw_h.iter().zip(&d_aw_logits_raw)
                .map(|(a, d)| a*(d - dot_aw)).collect();
            // d_q[h] += sum_t d_attn_logits[t] * k[t][h*HD..] / scale
            for t in 0..t_len {
                let ks = &ac.all_k[t][h*HEAD_DIM..(h+1)*HEAD_DIM];
                for i in 0..HEAD_DIM {
                    d_q[h*HEAD_DIM+i] += d_attn_logits[t] * ks[i] / scale;
                }
            }
            // d_k[t][h] += d_attn_logits[t] * q[h*HD..] / scale
            for t in 0..t_len {
                let dk = &mut d_kv_cache[l].0[t];
                let qs = &c.layers[l].q[h*HEAD_DIM..(h+1)*HEAD_DIM];
                for i in 0..HEAD_DIM {
                    dk[h*HEAD_DIM+i] += d_attn_logits[t] * qs[i] / scale;
                }
            }
        }

        // Project d_q, d_k_cur, d_v_cur back through wq/wk/wv
        let dk_cur = &d_kv_cache[l].0[cur_t];
        let dv_cur = &d_kv_cache[l].1[cur_t];
        let mut d_xn_attn = zeros(N_EMBD);
        linear_bwd_w(g.get_mut(&format!("{pf}.wq")).unwrap(), &d_q,   &lc.xn_attn, N_EMBD, N_EMBD);
        linear_bwd_x(&mut d_xn_attn, &d_q, &model.w[&format!("{pf}.wq")], N_EMBD, N_EMBD);
        linear_bwd_w(g.get_mut(&format!("{pf}.wk")).unwrap(), dk_cur, &lc.xn_attn, N_EMBD, N_EMBD);
        linear_bwd_x(&mut d_xn_attn, dk_cur, &model.w[&format!("{pf}.wk")], N_EMBD, N_EMBD);
        linear_bwd_w(g.get_mut(&format!("{pf}.wv")).unwrap(), dv_cur, &lc.xn_attn, N_EMBD, N_EMBD);
        linear_bwd_x(&mut d_xn_attn, dv_cur, &model.w[&format!("{pf}.wv")], N_EMBD, N_EMBD);

        let d_rn_a = rmsnorm_bwd(&d_xn_attn, &lc.x_pre_attn, lc.rms_a);
        dx = dx_s.iter().zip(&d_rn_a).map(|(a,b)| a+b).collect();
    }

    // Embedding gradients
    let wte_g = g.get_mut("wte").unwrap();
    for i in 0..N_EMBD { wte_g[c.tok_id*N_EMBD+i] += dx[i]; }
    let wpe_g = g.get_mut("wpe").unwrap();
    for i in 0..N_EMBD { wpe_g[(c.pos_id%BLOCK_SIZE)*N_EMBD+i] += dx[i]; }
}

// ─── Data ─────────────────────────────────────────────────────────────────────
fn names() -> Vec<&'static str> {
    vec!["emma","olivia","ava","isabella","sophia","mia","charlotte","amelia",
         "harper","evelyn","liam","noah","oliver","elijah","william","james",
         "benjamin","lucas","mason","ethan","aiden","logan","jackson","sebastian",
         "mateo","jack","owen","theodore","samuel","henry","leo","luke","jayden",
         "gabriel","landon","anthony","dylan","carter","julian","layla","zoe",
         "penelope","lily","eleanor","nora","luna","hazel","aurora","chloe",
         "aria","grace","zoey","riley","violet","nova","camille","claire","isla",
         "sofia","scarlett","elena","alice","savannah","daisy","audrey","ruby",
         "stella","naomi","adeline","ryan","caleb","eli","christian","josiah",
         "nathan","wyatt","andrew","joshua","christopher","lincoln","thomas",
         "ezra","hudson","daniel","nicholas","peter","john","levi","ian","axel",
         "cole","beau","felix","maya","nadia","iris","june","vera"]
}

fn build_vocab(docs: &[&str]) -> (Vec<String>, HashMap<String, usize>) {
    let mut chars: std::collections::BTreeSet<char> = std::collections::BTreeSet::new();
    for d in docs { for c in d.chars() { chars.insert(c); } }
    let mut vocab = vec!["<BOS>".to_string(), "<EOS>".to_string()];
    vocab.extend(chars.iter().map(|c| c.to_string()));
    let stoi: HashMap<String,usize> = vocab.iter().enumerate().map(|(i,s)| (s.clone(),i)).collect();
    (vocab, stoi)
}

// ─── Inference ────────────────────────────────────────────────────────────────
fn generate(model: &Model, vocab: &[String], stoi: &HashMap<String,usize>, rng: &mut Rng, n: usize) {
    let bos = stoi["<BOS>"];
    let eos = stoi["<EOS>"];
    for i in 0..n {
        let mut kv: Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)> = vec![(vec![], vec![]); N_LAYER];
        let mut tok = bos;
        let mut name = String::new();
        for pos in 0..BLOCK_SIZE {
            let c = forward(model, tok, pos, &mut kv);
            tok = rng.categorical(&c.probs);
            if tok == eos { break; }
            name.push_str(&vocab[tok]);
        }
        print!("sample {i}: {name}");
        if i < n-1 { print!("  "); }
    }
    println!();
}

// ─── Main ─────────────────────────────────────────────────────────────────────
fn main() {
    let mut rng = Rng::new(42);
    let docs    = names();
    let (vocab, stoi) = build_vocab(&docs);
    let vs  = vocab.len();
    let bos = stoi["<BOS>"];
    let eos = stoi["<EOS>"];
    let mut model = Model::new(vs, &mut rng);

    println!("zeroclawgpt v2  vocab={}  params={}  layers={}  embd={}  heads={}",
             vs, model.param_count(), N_LAYER, N_EMBD, N_HEAD);
    println!("Fixes: KV-cache causal attn | LR linear decay | beta2=0.95 | zero-init wo/fc2");
    println!("Training {} steps\n", N_STEPS);

    let t0 = Instant::now();

    for step in 0..N_STEPS {
        let doc = docs[step % docs.len()];
        let tokens: Vec<usize> = std::iter::once(bos)
            .chain(doc.chars().map(|c| stoi[&c.to_string()]))
            .chain(std::iter::once(eos))
            .take(BLOCK_SIZE)
            .collect();
        let seq_len = tokens.len();
        let n_pred  = seq_len.saturating_sub(1);

        model.zero_grad();

        // Forward ALL positions first (building KV cache), then backward
        let mut kv: Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)> = vec![(vec![], vec![]); N_LAYER];
        let mut caches: Vec<PosCache> = Vec::with_capacity(n_pred);
        let mut loss_sum = 0.0f32;

        for pos in 0..n_pred {
            let cache = forward(&model, tokens[pos], pos, &mut kv);
            loss_sum -= cache.probs[tokens[pos+1]].ln();
            caches.push(cache);
        }

        // Backward in reverse — d_kv_cache accumulates cross-position gradients
        // Initialize with zeros for each position's k and v
        let mut d_kv: Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)> = (0..N_LAYER).map(|_| {
            let dk: Vec<Vec<f32>> = (0..n_pred).map(|_| zeros(N_EMBD)).collect();
            let dv: Vec<Vec<f32>> = (0..n_pred).map(|_| zeros(N_EMBD)).collect();
            (dk, dv)
        }).collect();

        for pos in (0..n_pred).rev() {
            backward(&mut model, &caches[pos], tokens[pos+1], seq_len, &mut d_kv);
        }

        model.adam_step(step + 1);

        if step % 500 == 0 || step == N_STEPS-1 {
            print!("step {:>5}  loss={:.4}  t={:.2}s  | ",
                   step, loss_sum / n_pred as f32, t0.elapsed().as_secs_f32());
            generate(&model, &vocab, &stoi, &mut rng, 5);
            io::stdout().flush().unwrap();
        }
    }

    println!("\nDone in {:.3}s", t0.elapsed().as_secs_f32());
}
