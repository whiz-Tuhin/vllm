# AFD Optimizations

How to make the `MaNbF` bipartite actually pay off, given that today the
simplest case (`1A1F`) is the fastest. Two redesign moves carry most of the
benefit: **pre-route on the attention side** and **move shared experts off
the FFN side**. Section 5 (Design Decisions) walks through the per-config
tradeoffs, especially for the shared-expert relocation. Section 6 is the
probability math that quantifies the bandwidth win.

> **Companion docs**
>
> - `AFD-communication-structure.md` — the current `afd_pg` / Gloo / NCCL /
>   pair-group layout.
> - `MoE-expert-split.md` — TP vs EP inside the FFN side, expert placement.

---

## 1. Why `1A1F` currently wins

`1A1F` is the cheapest configuration to run because it has **no internal
collectives**:

- 1 attention GPU, 1 FFN GPU, 1 pair group.
- No FFN-side TP all-reduce (`tensor_parallel_size=1`).
- No FFN-side EP all-to-all (only 1 EP rank, all 64 experts local).
- No attention-side DP coordinator (only 1 attn rank).
- Every layer the cross-GPU traffic is just one `send(hidden_states)` plus
  one `recv(hidden_states)` on a single pair NCCL stream.

Every other config adds at least one of: TP all-reduce, EP all-to-all, DP
coordinator chatter, multi-pair stream contention. For DeepSeek V2-Lite
(small model, ~µs of compute per layer), those fixed coordination costs eat
the parallelism gain. The forward pass is comm-bound, not compute-bound, and
1A1F has the least comm.

The way to make `MaNbF` win is to **remove the per-layer coordination
overhead, not to add more parallelism on top of it.**

---

## 2. DBO (Dual Batch Overlap) and why it isn't on

The AFD config logs all show `num_afd_stages=1`. That means a single
attention→FFN→attention pipeline stage with **no microbatch overlap**. Per
layer, the critical path is strictly:

```text
  attn compute ─► send ─► FFN compute (incl. all-to-all) ─► recv ─► attn next layer
        ▲                                                                  │
        └──────────────────  serial dependency  ──────────────────────────┘
```

DBO splits the in-flight batch into halves and pipelines them so that
attention and FFN are doing *useful work simultaneously*:

```text
With DBO (conceptual):
  half-A:  [attn]──send─►       ◄──recv──[attn next]──send─► …
  half-B:        [attn]──send─►       ◄──recv──[attn next] …
                       ┊       ┊       ┊
                  FFN A ┊  FFN B ┊  FFN A ┊
critical path ≈ max(attn_time, comm + ffn_time)   instead of their sum
```

With `num_afd_stages>1` (or DBO turned on at the connector level) the per-layer
sync barrier between attn and ffn dissolves into pipeline bubbles only at the
edges. **Without DBO and with `num_afd_stages=1`, every layer is a hard
barrier**, which is the structural reason 1A1F's lower fixed overhead
dominates the parallelism advantage of bigger configs.

DBO is orthogonal to the routing redesign below: do both for compounding
wins.

---

## 3. Optimization #1: Pre-route on the attention side

Today the router runs on the FFN side, so attention has no idea which tokens
will need which experts. It must therefore either broadcast the full
hidden state to every FFN partner, or send to one partner and let the FFN
side EP-all-to-all redistribute. Either way, the FFN side runs the EP
all-to-all and a TP all-reduce on every layer.

If the router moves to the attention side:

```text
ATTENTION GPU                                     FFN GPU j
─────────────                                     ─────────
hidden = attn(...)
top_k, gate_w = router(hidden)         ← LOCAL
segment_j   = pack(hidden, tokens whose top_k ∩ ffn_j.experts ≠ ∅)
                              send segment_j ─►
                                                  recv segment_j   (only T_j tokens)
                                                  run owned experts on those tokens
                                                  weight by gate_w
                                                  pack partial_j
                              ◄── recv partial_j
combined = Σ_j partial_j  (weighted sum, scatter back into [T, hidden])
output = hidden + combined
```

Per layer, NCCL rounds drop from roughly:

| Step                                | Today  | Pre-routed |
| ----------------------------------- | ------ | ---------- |
| attn↔ffn pair send + recv           | 2      | 2          |
| EP all-to-all (dispatch + combine)  | 2      | 0          |
| TP all-reduce on dense path         | 1      | 0 (see §4) |
| **Total per layer**                 | **5**  | **2**      |

The two saved rounds are the EP all-to-all, whose latency *grows with `N`*.
Removing it is what makes `N` start paying for itself instead of degrading
into overhead.

---

## 4. Optimization #2: Shared experts move to the attention side

DeepSeek V2-Lite has 2 always-on **shared experts** per MoE layer plus the
64 routed experts. Today the shared experts live on the FFN side and are
TP-sliced; combining their output with routed-expert output requires a TP
all-reduce.

If pre-routing moves the *router* to attention, then attention is also the
natural place to run shared experts: they're dense, always-on, and need to
see the full `hidden_states` (which attention already has, before
segmentation).

```text
ATTENTION GPU                                     FFN GPU j
─────────────                                     ─────────
hidden = attn(...)
shared_out = shared_expert(hidden)     ← LOCAL, always-on dense
top_k, gate_w = router(hidden)         ← LOCAL
segment_j  = pack(...)
                              send segment_j ─►
                                                  routed experts
                              ◄── recv partial_j
combined = shared_out + Σ_j partial_j
output = hidden + combined
```

**Effects**

- FFN side's TP all-reduce on the routed path disappears too — partials are
  already token-disjoint after pre-routing, so each FFN's contribution can be
  combined additively on the attention side.
- FFN side becomes a *pure routed-expert engine*: receive a segment, run owned
  experts, send a partial. No internal collectives at all on the routed path.
- Shared-expert weights duplicate `M` times. The exact size and the
  per-config tradeoff are worked out in §5 (Design Decisions).

**Architectural rule** the redesign proposes:

> **Attention side owns everything dense and replicated. FFN side owns the
> sparse routed experts and only the sparse routed experts.**

The rule is conditional, not universal — see §5 for when to keep shared
experts on the FFN side instead.

---

## 5. Design decisions

### 5.1 The question

> *If shared experts run on the attention side, what's the downside, and
> how big are the shared-expert weights?*

The architectural rule in §4 sounds clean, but moving shared experts to the
attention side has real costs that depend on the `M / N` ratio. This section
quantifies them so you can decide per-config whether to actually do the move,
or pick a softer compromise.

### 5.2 Concrete size of shared-expert weights

The relevant DeepSeek V2-Lite values (visible in the FFN log header and the
V2-Lite model config):

| Parameter                | Value |
| ------------------------ | ----- |
| `hidden_size`            | 2048  |
| `moe_intermediate_size`  | 1408  |
| `n_shared_experts`       | 2     |
| `num_hidden_layers`      | 27    |
| `first_k_dense_replace`  | 1     |

So MoE layers = `27 − 1 = 26`. Each shared expert is a SwiGLU FFN with three
projections:

```text
gate_proj : [2048, 1408]   = 2,883,584 params
up_proj   : [2048, 1408]   = 2,883,584 params
down_proj : [1408, 2048]   = 2,883,584 params
                            ─────────────
                             8,650,752  per shared expert
```

(In implementation the 2 shared experts are typically fused into a single MLP
with `intermediate = 2 × 1408 = 2816`. Same total parameter count.)

Per MoE layer:

```text
2 shared experts × 8,650,752 = 17,301,504 params  ≈ 17.3 M
```

Across all 26 MoE layers:

```text
26 × 17,301,504 = 449,839,104 params  ≈ 450 M
```

In fp16 (the logs show `dtype=torch.float16`):

```text
450 M × 2 bytes ≈ 900 MB per replica
```

**That is per attention GPU**, if we move shared experts there in full
replication. About 4× larger than a quick estimate of "a few hundred MB" —
small in absolute terms, but worth tracking.

### 5.3 Memory accounting before vs after

Per-GPU view:

|                        | Today (FFN side, TP-sliced) | After (attn side, replicated) |
| ---------------------- | --------------------------- | ----------------------------- |
| Per FFN GPU            | `900 / N` MB                | 0                             |
| Per attention GPU      | 0                           | 900 MB                        |
| Cluster-wide total     | 900 MB                      | `M × 900` MB                  |

Per-config:

| Config | FFN-side today | Attn-side after | Cluster delta |
| ------ | -------------- | --------------- | ------------- |
| 1A1F   | 900 MB × 1     | 900 MB × 1      | 0             |
| 1A4F   | 225 MB × 4     | 900 MB × 1      | 0             |
| 2A4F   | 225 MB × 4     | 900 MB × 2      | +900 MB       |
| 4A4F   | 225 MB × 4     | 900 MB × 4      | +2.7 GB       |
| 7A1F   | 900 MB × 1     | 900 MB × 7      | +5.4 GB       |

The cluster-wide cost grows linearly with `M`. On an H200 (141 GB HBM) one
extra 900 MB per attention GPU is **~0.6% of memory** — fine for V2-Lite.
The same redesign applied to a larger MoE (DeepSeek-V3 has shared expert
intermediate=18432) would cost 10–15 GB per attention GPU and is **not**
free at that scale.

KV-cache impact for V2-Lite, using the 1A1F log as reference (KV cache
sized to 111.3 GiB ≈ 3.84 M tokens): losing 900 MB drops capacity to ~3.81 M
tokens — about 0.8% fewer concurrent sequences. Negligible here.

### 5.4 The five real downsides, ranked

#### Downside 1: Loss of TP scaling on shared-expert compute

Today the shared-expert FLOPs are TP-sliced across `N` FFN GPUs, so each
FFN GPU does `1/N` of the matmul. After the move each attention GPU does
the *full* shared-expert matmul locally, with no slicing.

Per-token shared-expert work ≈ `2 × params per token = 34.6 MFLOPs`.

Per-GPU shared-expert compute, comparing today (per FFN GPU, sees all `T`
tokens but does `1/N` of the work) vs after (per attention GPU, sees `T/M`
tokens and does the full work):

| Config | Per FFN GPU today (`T · f / N`) | Per attn GPU after (`T · f / M`) | Ratio (after / today) |
| ------ | ------------------------------- | -------------------------------- | --------------------- |
| 1A1F   | `T · f`                        | `T · f`                          | 1×                    |
| 1A4F   | `T · f / 4`                    | `T · f`                          | **4× more**           |
| 1A2F   | `T · f / 2`                    | `T · f`                          | 2× more               |
| 2A4F   | `T · f / 4`                    | `T · f / 2`                      | 2× more               |
| 4A4F   | `T · f / 4`                    | `T · f / 4`                      | 1×                    |
| 4A2F   | `T · f / 2`                    | `T · f / 4`                      | 0.5×                  |
| 7A1F   | `T · f`                        | `T · f / 7`                      | **0.14×**             |

`f` = shared-expert FLOPs per token.

The pattern: **the move helps when `M ≥ N` and hurts when `N > M`.**

- Configs where the move is a clear win on shared-expert compute: `7A1F`,
  `6A2F`, `4A2F`, `4A4F`, `3A1F`, `3A2F`, `2A1F` — attention side has more
  GPUs to spread the dense work.
- Configs where the move is a clear loss on shared-expert compute: `1A4F`,
  `1A2F`, `2A4F` — FFN side had more parallelism than attention does, and
  you give that up.

#### Downside 2: Longer attention critical path

Today the attention GPU's per-layer work is: attention → small post-attn →
send. After the move it becomes: attention → router → **shared expert
(~1 ms/layer on H200)** → pack → send. That's ~26 ms additional attention
compute per forward pass across all 26 MoE layers.

If the FFN side becomes faster than attention (likely once you remove the
EP all-to-all), this just moves the bottleneck back to attention. With DBO
that gets hidden; without DBO it's pure overhead on the critical path.

#### Downside 3: Higher per-process working set on attention

Loading the full shared-expert weights on every attention process changes
the model-loading code path: today the attention process only loads the
embedding, attention layers, and norms; FFN weights live on the FFN
process. After the move the attention process must also load shared-expert
weights for all 26 MoE layers. It's a real code change, not a config flag.

#### Downside 4: Combine arithmetic moves to attention

The weighted sum `Σ_j partial_j + shared_out` was being computed on the FFN
side; now it's on the attention GPU. Couple of additions per token —
small extra compute on attention. Not significant on its own, but adds to
Downside 2.

#### Downside 5: Numerical-consistency risk

When the math is split across two roles you have to ensure reduction order
matches between roles. The current FFN-side combine has a fixed order; an
attention-side combine over `N` partials may be reordered by NCCL recv
arrival sequence, producing tiny fp16 reduction differences. Usually fine,
but worth a correctness test gate.

### 5.5 Compromises if the downsides bite

When a config falls in the "loss" column (low `M`, high `N`), three softer
designs:

#### Option A: Keep shared experts on FFN side, replicate (don't TP-slice)

Each FFN GPU holds full shared weights and runs them on the *segment* it
received from pre-routing. No TP all-reduce needed because each FFN's
segment is token-disjoint.

- ✓ Preserves pre-routing's elimination of FFN-internal all-reduce.
- ✗ Duplicates `900 MB × N` on the FFN side.
- ✗ You have to ship full `hidden_states` to every FFN partner for the
  shared path, which kills most of pre-routing's bandwidth savings.

#### Option B: TP-slice shared experts across the *attention* DP group

The `M` attention GPUs jointly TP-slice the shared experts (each holds
`900/M` MB of shared weights). Per layer this requires one all-reduce among
the `M` attention GPUs — and the DP NCCL group already exists with exactly
those ranks, so you reuse it.

| Config | Per-attn shared weight | New collective on attn side |
| ------ | ---------------------- | --------------------------- |
| 4A4F   | 225 MB                 | all_reduce(4)               |
| 7A1F   | 128 MB                 | all_reduce(7)               |
| 1A4F   | 900 MB                 | none (`M = 1`)              |

Trade: 1 NCCL all-reduce on the attention side for the memory savings.
Net latency probably worse than Options A or full-replication for V2-Lite;
this option only really earns its keep at DeepSeek-V3 scale where the
shared-expert weights themselves are 10+ GB.

#### Option C: Move the *router* to attention but keep shared experts on FFN

Pre-route on attention. Send full hidden states to one FFN partner *and*
segments to the others. The "primary" FFN runs both shared (TP-sliced) and
routed paths; secondary FFNs run only routed.

- ✓ Preserves shared-expert TP scaling for high-`N` configs.
- ✗ Asymmetric pair links — primary FFN gets more bytes than the others.
- ✗ Most complex implementation of the three.

Probably not worth the complexity for V2-Lite. Useful in scenarios where
the shared expert is large enough that TP scaling on FFN side is essential.

### 5.6 Per-config recommendation

Combining the bandwidth math (§6) with the compute math (§5.4):

| Config | Recommended placement                   | Why                                          |
| ------ | --------------------------------------- | -------------------------------------------- |
| 1A1F   | Doesn't matter (`M = N = 1`)            | One GPU each side; topology trivial          |
| 1A2F   | Keep on FFN (Option A)                  | `N > M`, attention can't absorb dense work   |
| 1A4F   | Keep on FFN (Option A)                  | Worst case for the move — 4× compute hit     |
| 2A1F   | **Move to attention**                   | `M > N`, attention has more headroom         |
| 2A4F   | Keep on FFN (Option A)                  | `N > M`                                      |
| 3A1F   | **Move to attention**                   | `M > N`                                      |
| 3A2F   | **Move to attention**                   | `M > N`                                      |
| 4A2F   | **Move to attention**                   | `M > N`                                      |
| 4A4F   | Either; **move** if DBO is enabled      | `M = N`; DBO hides the longer attn path      |
| 6A2F   | **Move to attention**                   | `M ≫ N`, big compute win                     |
| 7A1F   | **Move to attention**                   | Largest compute win, FFN was the bottleneck  |

The redesign isn't one-size-fits-all. The clean "shared experts on
attention" rule is right for `M ≥ N`. For `M < N` keep them on FFN, with
local replication so pre-routing's other gains survive.

### 5.7 Bottom line

For DeepSeek V2-Lite specifically:

- **Shared expert weights = ~900 MB per replica** (450 M params × fp16,
  across 26 MoE layers).
- **Memory cost on attention side: ~0.6% of HBM** per attention GPU.
  Fine for V2-Lite, watch it for V3-class models.
- **The dominant tradeoff is compute, not memory.** Loss of TP scaling
  on shared-expert FLOPs is the biggest cost; it favors high-`M` low-`N`
  configs.
- **The attention critical path grows by ~26 ms per pass.** Negligible if
  DBO is enabled, problematic without it.
- **Use the `M ≥ N` rule** as the default decision boundary; pick a softer
  compromise (Option A on FFN side) for the inverted configs.

---

## 6. The probability math: how much bandwidth does pre-routing save?

This is the part that explains why pre-routing is more than a "nice
refactor" — it's a quantifiable win that *grows with `N`*.

### 6.1 Setup and notation

| Symbol     | Meaning                                                      | DeepSeek V2-Lite value |
| ---------- | ------------------------------------------------------------ | ---------------------- |
| `E`        | Number of routed experts in the layer                        | 64                     |
| `k`        | Top-`k` experts selected per token by the router             | 6                      |
| `N`        | Number of FFN GPUs (split factor)                            | 1, 2, or 4 in the configs |
| `E/N`      | Routed experts per FFN GPU (linear placement, EP)            | 64, 32, or 16          |
| `T`        | Tokens in the layer step (across all attentions, in aggregate) | varies                |
| `H`        | Hidden dim                                                   | 2048                   |
| `T_j`      | Tokens whose top-`k` overlaps FFN `j`'s expert set           | random variable        |

We assume **uniform routing**: every choice of `k` experts out of `E` is
equally likely per token. (Real models have some skew; modern MoE training
uses balancing losses to keep this approximately true. DeepSeek V2-Lite uses
bias-based load balancing.)

### 6.2 Probability that a token hits a specific FFN

Fix an FFN `j` that owns `E/N` experts. A token "hits FFN `j`" iff at least
one of its top-`k` experts is among those `E/N`.

Let `A` = the event "all `k` experts miss FFN `j`'s set". Then `P(hit) = 1 - P(A)`.

The number of ways to pick `k` experts that all miss `j`'s `E/N` is
`C(E - E/N, k)`. The total number of size-`k` subsets is `C(E, k)`. So:

$$
P(\text{hit FFN }j) \;=\; 1 - \frac{\binom{E - E/N}{\,k\,}}{\binom{E}{\,k\,}}
$$

This is the same as drawing `k` cards without replacement and asking for the
probability of seeing at least one of `E/N` "marked" cards in a deck of `E`.

### 6.3 Numerical evaluation for DeepSeek V2-Lite

With `E = 64`, `k = 6`:

```text
N = 1  (E/N = 64)
   P(hit) = 1 - C(0, 6)/C(64, 6) = 1 - 0/74,974,368 = 1.000

N = 2  (E/N = 32)
   P(hit) = 1 - C(32, 6)/C(64, 6)
          = 1 - 906,192 / 74,974,368
          = 1 - 0.01209
          ≈ 0.9879

N = 4  (E/N = 16)            ← e.g. 1A4F, 2A4F, 4A4F
   P(hit) = 1 - C(48, 6)/C(64, 6)
          = 1 - 12,271,512 / 74,974,368
          = 1 - 0.16367
          ≈ 0.8363

N = 8  (hypothetical, E/N = 8)
   P(hit) = 1 - C(56, 6)/C(64, 6)
          = 1 - 32,468,436 / 74,974,368
          = 1 - 0.43306
          ≈ 0.5669

N = 16 (hypothetical, E/N = 4)
   P(hit) = 1 - C(60, 6)/C(64, 6)
          = 1 - 50,063,860 / 74,974,368
          ≈ 0.3322

N = 64 (extreme: 1 expert per FFN)
   P(hit) = k/E = 6/64 = 0.09375
```

### 6.4 Expected per-FFN load and total bytes shipped

By linearity of expectation, the expected number of tokens that hit FFN `j`
is:

$$
\mathbb{E}[T_j] \;=\; T \cdot P(\text{hit FFN }j)
$$

The aggregate volume sent across all `N` FFNs (counting each token once per
FFN it touches) is:

$$
\mathbb{E}\!\left[\sum_j T_j\right] \;=\; N \cdot T \cdot P(\text{hit})
$$

Note: a single token can be sent to multiple FFNs (because its `k=6`
experts may straddle FFN boundaries). The expected number of FFNs each
token visits is exactly `N · P(hit)`. As `N → E`, this approaches `k`
(every token visits exactly its `k` experts' homes).

### 6.5 Bandwidth comparison: pre-routed vs broadcast

Two baselines for comparison:

- **Broadcast-everything**: attention sends full hidden states to all `N`
  FFN partners. Total volume = `N · T · H` per layer.
- **Pre-routed segments**: attention sends only relevant tokens to each FFN.
  Total volume = `N · T · P(hit) · H` per layer.

Bandwidth ratio (pre-routed / broadcast) = `P(hit)`. Savings = `1 - P(hit)`:

| `N` | `E/N` | `P(hit)`  | Per-FFN volume    | Aggregate vs broadcast | Bandwidth saved |
| --- | ----- | --------- | ----------------- | ---------------------- | --------------- |
| 1   | 64    | 1.000     | `1.00 · T · H`    | `1.00 · N · T · H`     | 0%              |
| 2   | 32    | 0.988     | `0.99 · T · H`    | `1.98 · T · H`         | 1.2%            |
| 4   | 16    | 0.836     | `0.84 · T · H`    | `3.35 · T · H`         | 16.4%           |
| 8   | 8     | 0.567     | `0.57 · T · H`    | `4.54 · T · H`         | 43.3%           |
| 16  | 4     | 0.332     | `0.33 · T · H`    | `5.32 · T · H`         | 66.8%           |
| 64  | 1     | 0.094     | `0.09 · T · H`    | `6.00 · T · H`         | 90.6%           |

**Two readings of this table:**

1. For `N = 2` (1A2F, 2A2F variants in this sweep would be — note the sweep
   only has 1A2F as a 2F config; 2F in others) the savings are tiny (~1%) —
   pre-routing is a comm-rounds optimization, not a bandwidth one in this
   regime.
2. For `N = 4` (1A4F, 2A4F, 4A4F) you save ~16% on the attn→ffn volume,
   modest. The big win is the eliminated EP all-to-all (§3), not the byte
   savings.
3. As `N` grows beyond 4, the bandwidth savings start to dominate too. By
   `N = E = 64`, pre-routing ships only `k/E = 9.4%` of what a broadcast
   would.

So the **return-vs-`N`** curve flips:

```text
                Comm overhead vs N

  Today:     overhead grows with N
  ▲                                                   ┌─
  │                                              ┌────┘
  │                                       ┌──────┘
  │                                ┌──────┘    ← EP all-to-all latency
  │                         ┌──────┘             scales with N
  │                  ┌──────┘
  │           ┌──────┘
  │   ┌───────┘
  └──────────────────────────────────────────►  N

  Pre-routed (Optimizations 1+2):
  ▲
  │
  │\
  │ \
  │  \                                    ← per-FFN load drops as 1/N
  │   ───\                                  bandwidth and compute both
  │       ──────\                            scale benignly with N
  │              ───────────\__
  └──────────────────────────────────────────►  N
```

### 6.6 Compute-side scaling for completeness

Per FFN GPU, expected compute per layer:

$$
\text{FLOPs}_j \;\approx\; \mathbb{E}[T_j] \cdot \frac{k}{N} \cdot f_{\text{expert}}
\;=\; T \cdot P(\text{hit}) \cdot \frac{k}{N} \cdot f_{\text{expert}}
$$

where `f_expert` is the FLOP cost of running one expert on one token. Using
the values above:

| `N` | `T_j / T` | `(k/N)` | Per-FFN expert FLOPs as fraction of single-GPU |
| --- | --------- | ------- | ----------------------------------------------- |
| 1   | 1.000     | 6.000   | `6.000 · T · f` (full work, single GPU)         |
| 2   | 0.988     | 3.000   | `2.964 · T · f` ≈ ½× single-GPU                |
| 4   | 0.836     | 1.500   | `1.254 · T · f` ≈ ¼× single-GPU                |
| 8   | 0.567     | 0.750   | `0.425 · T · f` ≈ 1⁄14× single-GPU             |

So with pre-routing, FFN compute scales **roughly as `1/N`** even though
each token may visit multiple FFNs. The `k/N` factor (active fraction of an
FFN's experts per token) and the `P(hit)` factor (whether the token came at
all) compound.

### 6.7 Skew caveat

Everything above assumes uniform routing. Real expert popularity has a
heavy tail. If FFN `j` happens to host the few hottest experts, `T_j` blows
up and `j` becomes the straggler. Mitigations:

- Periodically permute the expert→FFN placement based on observed load
  (covered by `enable_return_routed_experts` machinery in vLLM in some
  configurations).
- Capacity-bounded routing: cap each FFN's per-step token count and overflow
  routes to next-best expert. Reduces straggler impact at small accuracy cost.
- Auxiliary load-balancing loss during training (DeepSeek V2 uses bias-based
  balancing — present in the V2-Lite checkpoint).

---

## 7. Quantitative comparison: 2A4F today vs 2A4F redesigned

For the 20-prompt example (request rate ∞, `T` tokens of activation per
attention engine after batching):

### Today (router on FFN, shared experts on FFN, no DBO)

```text
Per layer, per attention engine (~T tokens):

  attn compute                                               (parallel: 2 attns)
  send hs[T, 2048] to 1 FFN partner                          (NCCL 1 round)
  FFN side:
    router locally on T tokens                               (small)
    EP all-to-all: redistribute T tokens by expert location  (NCCL 1 round, scales w/ N)
    expert compute on local 16/64 experts × ~6T/4 work       (parallel)
    EP all-to-all: combine outputs                           (NCCL 1 round, scales w/ N)
    TP all-reduce on shared-expert path                      (NCCL 1 round, scales w/ N)
  ffn → attn: hs[T, 2048]                                    (NCCL 1 round)
  attn next layer

NCCL rounds on critical path: ≈5
Cross-link bytes per layer: 2 · T · H  (attn↔ffn pair)
Plus all FFN-internal collective bytes
```

### Redesigned (Optimizations 1 + 2)

```text
Per layer, per attention engine (~T tokens):

  attn compute + RMSNorm                                     (parallel: 2 attns)
  router locally + shared_expert locally                     (small dense compute)
  pack 4 segments, one per FFN                               (CPU bookkeeping)
  send segment_j to ffn_j × 4 in parallel                    (NCCL 1 round, 4 streams)
  FFN side:
    expert compute on received tokens × 16/64 experts       (parallel)
  ffn → attn: partial_j × 4 in parallel                      (NCCL 1 round, 4 streams)
  combine partials + add shared_out to residual              (small)
  attn next layer

NCCL rounds on critical path: 2
Cross-link bytes per layer: 2 · 0.836 · T · H  (per pair, 4 pairs)
                          = ≈3.35 · T · H total each direction
                          (vs broadcast 4·T·H, current send-to-one 1·T·H)
No FFN-internal collective rounds at all
```

### Effect on the M×N scaling story

| Config        | Current overhead (≈ rounds × N-factor) | Redesigned overhead          |
| ------------- | -------------------------------------- | ---------------------------- |
| 1A1F          | 2 rounds, no internal collectives      | 2 rounds (no change)         |
| 1A2F          | 5 rounds, light collectives            | 2 rounds (saves 3)           |
| 1A4F          | 5 rounds, heavier all-to-all           | 2 rounds (saves 3)           |
| 2A4F          | 5 rounds + DP coord chatter            | 2 rounds + DP coord chatter  |
| 4A4F          | 5 rounds + N-scaled all-to-all         | 2 rounds (the "N" vanishes)  |
| 7A1F          | 5 rounds                               | 2 rounds                     |

The rounds-per-layer constant becomes 2 across every `MaNbF`, which is what
makes the parallelism actually pay.

---

## 8. When does M×N scaling pay off (post-redesign)?

After Optimizations 1+2 (and ideally DBO too):

- **Increasing `M` (attentions)** scales attention/KV-cache capacity
  approximately linearly (vLLM data parallelism). Limited by request rate
  and KV cache memory, not by AFD comm.
- **Increasing `N` (FFNs)** scales routed-expert compute roughly as `1/N`
  per FFN GPU. Limited by:
  - Per-FFN minimum useful batch size (don't make `T_j` so small that GEMMs
    starve).
  - NVLink bandwidth at the receivers (each FFN GPU receives from `M`
    attention GPUs in parallel; bandwidth converges).
  - Routing skew (hot expert → straggler).
  - Diminishing per-FFN compute, eventually overtaken by per-pair fixed
    costs.

The original sweep suggests `1A1F` is best **today**; with the redesign,
`MaNbF` configurations should beat it for moderate `M` and `N` once the
overhead floor drops to 2 NCCL rounds per layer.

---

## 9. Summary

1. **Why 1A1F wins today**: it has zero internal collectives. Every other
   config adds TP/EP/DP coordination that costs more than the parallelism
   buys.
2. **DBO** would hide attn↔ffn comm behind compute via dual-batch
   pipelining. Currently disabled (`num_afd_stages=1`).
3. **Pre-route on attention** eliminates the EP all-to-all on the FFN side
   and reduces cross-link bytes by `1 - P(hit)`. Bandwidth savings grow
   with `N`.
4. **Move shared experts to attention** so the FFN side has zero internal
   collectives on the routed path either. Cost: replicate ~900 MB of
   shared weights per attention GPU and lose TP scaling on shared-expert
   compute.
5. **Design decisions** (§5) shows the shared-expert relocation is a win
   for `M ≥ N` configs (`7A1F`, `4A4F`, `3A2F`, etc.) and a loss for
   `N > M` configs (`1A2F`, `1A4F`, `2A4F`) — keep them on the FFN side
   in the latter case, replicated rather than TP-sliced.
6. **Probability math** (§6) shows pre-routing's bandwidth win is small
   for `N ≤ 4` but the *rounds saved* (2 instead of 5 per layer) is the
   dominant effect at all `N ≥ 2`.
7. After the redesign, `M × N` scaling should become a real lever — both
   `M` (attn DP) and `N` (FFN expert distribution) earn their keep without
   coordination overhead growing with `N`.
