# AFD: Current State & Path to Multi-GPU Wins

**Model:** DeepSeek V2-Lite · **Sweep:** `run-20260415-220850` · 11 `MaNbF` configurations

> **Companion docs (deep dives):**
> `AFD-communication-structure.md` · `MoE-expert-split.md` · `AFD-optimizations.md`

---

## 1 · The puzzle

- We benchmarked 11 attention/FFN disaggregation (AFD) configs from `1A1F` to `7A1F`.
- **`1A1F` (1 attention GPU + 1 FFN GPU) is the fastest.**
- Every larger configuration loses despite having more parallelism.
- This deck explains *why*, and what changes make `MaNbF > 1A1F` realistic.

---

## 2 · AFD in one picture

```text
                       ┌────────────────────────────────────┐
                       │     attention server pool (×M)     │
                       │    embedding · MLA · KV cache      │
                       └─────────────────┬──────────────────┘
                                         │ hidden_states
                                  M × N pair links
                                         │ hidden_states
                       ┌─────────────────┴──────────────────┐
                       │        FFN server pool (×N)        │
                       │  router · shared experts · routed  │
                       │  experts · MoE all-to-all · TP A/R │
                       └────────────────────────────────────┘
```

- **Attention side** runs DP across `M` engines.
- **FFN side** runs `tensor_parallel_size = N` (TP for dense, EP for routed experts).
- Connected by an **`M × N` bipartite** of pair groups.

---

## 3 · Two communication planes

```text
            afd_pg  (Gloo, world = M+N)        ← control plane
                control / handshakes
                     │
                     ▼ new_group per (a,f)
        ┌────────────────────────────┐
        │  pair group (a, f)         │
        │   ranks = {attn_a, ffn_f}  │
        │   Gloo = small messages    │
        │   NCCL = hidden_states     │ ← data plane, on a CUDA stream
        └────────────────────────────┘
```

- `afd_pg`: every process joins, used for **profiler / shutdown / handshakes**.
- Pair groups: 2-rank, **dual-backend**:
  - Gloo for CPU control (`pair_id`, token counts, acks).
  - NCCL for GPU tensor send/recv on its own CUDA stream → multiple pairs can overlap.
- Plus: **TP-NCCL** inside FFN side (size `N`), **DP-NCCL** inside attn side (size `M`).

---

## 4 · Per-config bipartite

| Config | M (attn) | N (ffn) | GPUs | Pair groups | Streams / attn GPU | Streams / ffn GPU |
| ------ | -------- | ------- | ---- | ----------- | ------------------ | ----------------- |
| 1A1F   | 1        | 1       | 2    | 1           | 1                  | 1                 |
| 1A2F   | 1        | 2       | 3    | 2           | 2                  | 1                 |
| 1A4F   | 1        | 4       | 5    | 4           | 4                  | 1                 |
| 2A1F   | 2        | 1       | 3    | 2           | 1                  | 2                 |
| 2A4F   | 2        | 4       | 6    | 8           | 4                  | 2                 |
| 3A1F   | 3        | 1       | 4    | 3           | 1                  | 3                 |
| 3A2F   | 3        | 2       | 5    | 6           | 2                  | 3                 |
| 4A2F   | 4        | 2       | 6    | 8           | 2                  | 4                 |
| 4A4F   | 4        | 4       | 8    | 16          | 4                  | 4                 |
| 6A2F   | 6        | 2       | 8    | 12          | 2                  | 6                 |
| 7A1F   | 7        | 1       | 8    | 7           | 1                  | 7                 |

---

## 5 · What happens per MoE layer today

```text
ATTENTION                                            FFN (×N, TP+EP)
─────────                                            ───────────────
attn(...)
   │ NCCL send hidden_states[T, 2048]
   ├──────────────────────────────────►   recv
                                          router (local)
                                          EP all-to-all  (dispatch)   ← grows with N
                                          run owned experts
                                          EP all-to-all  (combine)    ← grows with N
                                          TP all-reduce (shared path) ← grows with N
   │                            recv ◄──  send hidden_states[T, 2048]
   ├──── add residual ─────►   next layer
```

**5 NCCL rounds on the per-layer critical path.** `num_afd_stages=1` means strict serialization — attn idle while FFN works, FFN idle while attn works.

---

## 6 · Why `1A1F` wins (root cause)

- **Zero internal collectives**: no EP all-to-all, no TP all-reduce, no DP coordinator.
- Cross-link traffic is just `send` + `recv` on **one** pair stream.
- For DeepSeek V2-Lite (small, comm-bound), fixed coordination cost > parallelism gain.

> **Insight:** The way to make `MaNbF` win isn't more parallelism — it's **less coordination per layer.**

---

## 7 · Two redesign moves

### Move 1 — Pre-route on the attention side

```text
ATTENTION                                            FFN j
─────────                                            ─────
attn(...)
top_k, gate_w = router(hidden)                     ← LOCAL on attn
segment_j = pack(tokens whose top_k touches ffn_j)
   │ send segment_j[T_j, 2048]
   ├──────────────────────────────────►   run owned experts
                                          weight by gate_w
   │                            recv ◄──  send partial_j[T_j, 2048]
combined = Σ_j partial_j   (weighted sum on attn)
```

- Eliminates **EP all-to-all** on FFN side (saves 2 NCCL rounds/layer).
- Bandwidth saved per pair = `1 − P(hit)`, **grows with N**.

### Move 2 — Shared experts on attention side (conditional)

- Run shared experts locally on attn before pre-routing.
- Eliminates **TP all-reduce** on FFN side (saves 1 NCCL round/layer).
- Cost: replicate shared-expert weights on `M` attention GPUs.

---

## 8 · Bandwidth math (the `P(hit)` table)

`P(token hits FFN_j) = 1 − C(E − E/N, k) / C(E, k)` · DeepSeek V2-Lite: `E=64, k=6`

| `N` | `E/N` | `P(hit)` | Bandwidth saved per pair |
| --- | ----- | -------- | ------------------------ |
| 1   | 64    | 1.000    | 0%                       |
| 2   | 32    | 0.988    | 1%                       |
| 4   | 16    | **0.836** | **16%**                |
| 8   | 8     | 0.567    | 43%                      |
| 16  | 4     | 0.332    | 67%                      |
| 64  | 1     | 0.094    | 91%                      |

**Reading:** at the configs in this sweep (`N ≤ 4`) bandwidth savings are modest; the *rounds saved* (5 → 2 per layer) is the bigger win. Bandwidth savings dominate at `N ≥ 8`.

---

## 9 · Where shared experts run — the `M / N` rule

Shared-expert weight per replica: **~900 MB** for V2-Lite (450M params × fp16).

- **`M ≥ N`** → move to attention side. Attention has spare GPUs; replication is cheap.
- **`M < N`** → keep on FFN side, replicated (no TP slice). Attention doesn't have the parallelism to absorb the dense work.

| Config | Recommended | Why                           |
| ------ | ----------- | ----------------------------- |
| 1A1F   | Either      | trivial topology              |
| 1A2F, 1A4F, 2A4F | **Keep on FFN** | `N > M`, attention overloaded |
| 2A1F, 3A1F, 3A2F, 4A2F, 6A2F, 7A1F | **Move to attention** | `M ≥ N`, attention has headroom |
| 4A4F   | Move (with DBO) | `M = N`; DBO hides longer attn path |

---

## 10 · DBO (Dual Batch Overlap)

```text
Without DBO (today, num_afd_stages = 1):
   layer L:  attn ─send─► [FFN waits & computes] ─recv─► attn next
             ◄──────── strict serial barrier ────────►

With DBO (num_afd_stages = 2):
   half-A:   attn─send─►            ◄─recv─attn next ...
   half-B:        attn─send─►            ◄─recv─attn next ...
                  └─FFN A─┘    └─FFN B─┘
   critical path ≈ max(t_attn, t_link + t_ffn)
```

- Today: `num_afd_stages = 1` for every config — every layer is a hard barrier.
- DBO splits the in-flight batch and pipelines so attention compute hides FFN comm + compute.
- **Required for V3-class models**, **valuable for V2-Lite** once Moves 1+2 are in.

---

## 11 · Net effect — rounds per layer

| Step                                 | Today | After Moves 1+2 |
| ------------------------------------ | ----- | --------------- |
| attn↔ffn pair send + recv            | 2     | 2               |
| EP all-to-all (dispatch + combine)   | 2     | **0**           |
| TP all-reduce (shared path)          | 1     | **0**           |
| **Total per-layer NCCL rounds**      | **5** | **2**           |
| Per-layer rounds *grow* with `N`?    | yes   | no              |

The "rounds" count is what flips `1A1F-is-best` into `MaNbF-is-better`.

---

## 12 · Stack of changes (priority order)

1. **Pre-route on attention** — biggest impact, removes the `N`-scaled overhead.
2. **Enable DBO** (`num_afd_stages ≥ 2`) — hides remaining cross-link cost.
3. **Move shared experts** per the `M / N` rule — squeezes the last collective out.
4. **Tune NCCL buffer sizes** for high-`M × N` (16+ pair groups per GPU).

---

## 13 · Open design decisions

- **Combine arithmetic on attention side** — fp16 reduction order may differ from FFN-side combine; needs a numerical-consistency test.
- **Routing skew** — one hot expert can stall a whole layer; capacity-bounded routing or expert-placement reshuffle needed.
- **Shared-expert TP-slicing** across attention DP group (Option B in `AFD-optimizations.md` §5.5) — required when shared weights get larger (V3 ≈ 5 GB / replica).
- **Pair-groups vs flat all-to-all** at large `M × N` — pair-group buffer memory grows as `M × N × buffer_size`; flat all-to-all collapses this.

---

## 14 · Looking ahead — DeepSeek V3

Same framework, different inputs:

- `hidden_size`: 2048 → 7168 (3.5×)
- Routed experts: 64 → **256**, `k`: 6 → 8
- Sparsity `k/E`: 9.4% → **3.1%** (more sparse → pre-routing helps more)
- Shared-expert size: 0.9 GB → **2.5 GB** (fp8) / 5.1 GB (fp16) per replica
- Routed weights: ~16 GB → **654 GB** (forces `N ≥ 16` minimum)

**Implications:**

- `1A1F` doesn't exist for V3 — model can't fit on a single FFN GPU.
- Pre-routing is no longer optional; it's **first-order necessary** (cuts cross-link bytes by 60–90%).
- DBO is required, not optional.
- Shared experts likely **TP-sliced across attention DP group**, not fully replicated.
- Pair-group count grows; consider flat all-to-all for `M × N ≥ 256`.

---

## 15 · One-line summary

> **Today's AFD pays a per-layer coordination tax that scales with `N`.**
> **Pre-routing + DBO + conditional shared-expert relocation removes that tax,**
> **letting `M × N` parallelism finally pay off — and is mandatory for V3-scale models.**

---

## Appendix A · Glossary

- **AFD** — Attention–FFN Disaggregation. Splitting attention and feed-forward across different GPUs.
- **MoE** — Mixture of Experts. Router picks top-`k` of `E` experts per token.
- **TP / EP / DP** — Tensor / Expert / Data parallelism.
- **`afd_pg`** — Global Gloo process group of all `M+N` AFD processes.
- **Pair group** — 2-rank `(attn, ffn)` sub-group; Gloo for control, NCCL for tensors.
- **DBO** — Dual Batch Overlap; pipelines two halves of a batch to hide cross-link latency.
- **`P(hit)`** — Probability that a given token's top-`k` includes any expert on a given FFN.
- **`num_afd_stages`** — AFD pipeline depth; `1` means no DBO.

## Appendix B · Pointers to deeper detail

- Communication mechanics (Gloo / NCCL / pair-group bring-up): `AFD-communication-structure.md`
- MoE layer organization (TP for shared, EP for routed): `MoE-expert-split.md`
- Optimization analysis (probability math, design decisions, scaling): `AFD-optimizations.md`
