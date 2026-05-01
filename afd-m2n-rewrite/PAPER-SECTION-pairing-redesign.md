# AFD Pairing Topology Redesign

## Background

Attention–FFN Disaggregation (AFD) splits the forward pass of a Mixture-of-Experts
(MoE) language model across two pools of GPUs: one running the attention sublayers
and one running the FFN/MoE sublayers. The two pools communicate the per-layer
hidden state across a process boundary using NCCL P2P. We denote a configuration
with $M$ attention ranks and $N$ FFN ranks as $M$A$N$F.

The original implementation in vLLM PR #29772 (commit `004c761a5`) coupled the
two pools with a **1:1 pairing topology** and relied on intra-FFN expert-parallel
collectives to route tokens to their target experts. We replace this with an
**$M \times N$ bipartite topology** combined with **ATTN-side pre-routing**.
This document describes both designs precisely and analyzes the communication
and synchronization cost of each.

The model used throughout this section is DeepSeek-V2-Lite: 26 MoE layers + 1
dense layer, hidden size $H = 2048$, $E = 64$ routed experts, top-$k = 6$ routing,
shared experts present.

---

## Original Design: 1:1 Pairing with EP All-to-All

### Pairing topology

For a configuration $M$A$N$F where $M = N = R$ (the original code only supported
symmetric configurations), the connector created exactly $R$ NCCL pairs by zipping
the rank lists:

$$
\mathcal{P}_{\text{1:1}} = \{(\text{ATTN}_r, \text{FFN}_r) \mid r \in [0, R)\}
$$

Asymmetric configurations ($M \ne N$) were blocked by an assertion at
`p2p_connector.py:174` of the original code. This restriction excluded common
deployment patterns such as $M$A1F (multiple attention DP ranks served by a
single FFN process) and 1A$N$F (a single attention rank fanning out to $N$ FFN
TP/EP workers).

### Per-layer data flow

Each pair operated independently along the AFD boundary: ATTN rank $r$ sent its
slice of the hidden state $h_r \in \mathbb{R}^{n_r \times H}$ to FFN rank $r$,
which then participated in an **intra-FFN expert-parallel collective** to gather
the global token set, ran the routed-expert compute on its local expert shard,
and reduce-scattered partials back. The full per-MoE-layer data flow for a 2A2F
configuration was:

```
ATTN_DP0  ──[h0]──►  FFN_TP0/EP0  ◄──┐
                                      │  EP all-gather: each FFN gets [h0; h1]
ATTN_DP1  ──[h1]──►  FFN_TP1/EP1  ──►┘
                          │
                  router(global_h) → topk_ids ∈ [0, 64)
                  fused_experts: each FFN runs only its local 32 experts
                          │
ATTN_DP0  ◄──[h0]──  FFN_TP0/EP0  ◄──┐
                                      │  EP reduce-scatter: partials combined
ATTN_DP1  ◄──[h1]──  FFN_TP1/EP1  ──►┘
```

### Communication cost per MoE layer

With $N_{\text{tot}}$ tokens distributed across $R$ ATTN ranks ($n_r = N_{\text{tot}}/R$):

| Stage | Volume |
|-------|-------:|
| ATTN → FFN P2P (M sends) | $N_{\text{tot}} \cdot H$ |
| EP all-gather (between FFN workers) | $N_{\text{tot}} \cdot H \cdot (R-1)$ |
| EP reduce-scatter (between FFN workers) | $N_{\text{tot}} \cdot H$ |
| FFN → ATTN P2P (M sends) | $N_{\text{tot}} \cdot H$ |
| **Total bytes (asymptotic)** | $\sim (R+2) \cdot N_{\text{tot}} \cdot H$ |

For 2A2F, this is $\sim 4 N_{\text{tot}} H$ in collectives plus $2 N_{\text{tot}} H$
in P2P, dominated by the EP collectives.

### Synchronization properties

The MoE router (`gate` linear projection followed by `grouped_topk`), the shared
experts, and the routed experts all executed **on the FFN side**. The ATTN side
only forwarded the post-attention hidden state. This had three structural
consequences:

1. **EP collectives between FFN workers were on the critical path of every MoE
   layer.** They could not be overlapped with the ATTN-side compute because ATTN
   is idle waiting for the FFN response.
2. **The router was redundantly available on every FFN worker** (each holds the
   full gate weights for the all-gather path), wasting weight memory.
3. **Expert balance across FFN workers was determined by routing decisions made
   inside the EP collective**, making it impossible to send only the needed
   tokens to each FFN worker — every token went to every worker.

The original 2A2F achieved ~2076 ms TPOT on L40S in this design (with diagnostic
logging removed; the unfixed `.item()` log path produced 88,342 ms TPOT due to a
hidden CUDA sync).

---

## New Design: $M \times N$ Bipartite Pairing with ATTN-Side Pre-Routing

### Pairing topology

We replace the 1:1 zip with a **full bipartite product**:

$$
\mathcal{P}_{M\times N} = \{(\text{ATTN}_i, \text{FFN}_j) \mid i \in [0, M), \, j \in [0, N)\}
$$

The connector creates $M \cdot N$ NCCL pair-groups, each with a deterministic
`pair_id = i * N + j` that maps to a unique TCP port for the Gloo handshake.
Only the two members of each pair participate in its initialization; non-member
ranks skip the iteration. This works for arbitrary $M, N \ge 1$, including
asymmetric configurations such as 1A2F, 3A1F, 2A4F.

**Pair count by configuration:**

| Config | Pairs |
|--------|------:|
| 1A1F | 1 |
| 1A2F | 2 |
| 2A2F | 4 |
| 3A1F | 3 |
| 1A4F | 4 |
| 2A4F | 8 |

Pair-group construction uses `init_afd_process_group` directly (bypassing
`torch.distributed.new_group`'s global counter, which would otherwise desync
between ATTN and FFN processes that have made different numbers of internal
`new_group` calls during model initialization).

### Migration of router and shared experts to ATTN

The MoE router and the shared experts are moved out of the FFN-side
`DeepseekV2MoE` and into a new ATTN-side module `DeepseekV2MoEAttentionStub`,
which contains:

- `gate`: the `ReplicatedLinear(H, E)` router projection
- `shared_experts`: the `DeepseekV2MLP` shared-expert block
- `compute_route_and_shared(h)` → `(topk_ids, topk_weights, shared_out)`

Submodule names are preserved so HuggingFace checkpoint weights load unchanged.
The ATTN-side weight loading filter is updated to accept `gate.*` and
`shared_experts.*` parameters; routed-expert parameters remain skipped on ATTN
and loaded only on FFN.

### Per-layer data flow

For each MoE layer, the ATTN side now executes:

```
1.  h ← attention(x)
2.  h ← post_attention_layernorm(h)
3.  router_logits ← gate(h)                              # ATTN-local
4.  topk_weights, topk_ids ← grouped_topk(router_logits) # ATTN-local
5.  shared_out ← shared_experts(h)                       # ATTN-local
6.  for j in [0, N):
        nccl_send(h, topk_ids, topk_weights → FFN_j)     # M×N pair (i,j)
7.  for j in [0, N):
        partial_j ← nccl_recv(FFN_j)                     # M×N pair (i,j)
8.  h ← Σ_j partial_j + shared_out
```

And each FFN rank $j$:

```
1.  for i in [0, M):
        h_i, topk_ids_i, topk_weights_i ← nccl_recv(ATTN_i)
2.  h ← cat(h_i)        # concat across source ATTN ranks
    topk_ids ← cat(topk_ids_i)
    topk_weights ← cat(topk_weights_i)
3.  partial ← fused_experts(h, topk_ids, topk_weights, expert_map=local_E_j)
4.  for i in [0, M):
        nccl_send(partial[source_slice_i] → ATTN_i)
```

`fused_experts` is the raw Triton MoE kernel; it uses the `expert_map` parameter
to mark non-local expert IDs as $-1$, causing those routing entries to
contribute zero. This means each FFN worker computes only the experts it owns,
and **no EP all-gather or reduce-scatter is performed**.

### Bypass of `FusedMoEModularKernel`

A subtle but critical detail: vLLM's `FusedMoE.forward_impl` dispatches to
`FusedMoEModularKernel.forward`, which internally calls
`prepare_finalize.prepare()` (the EP all-gather) and `prepare_finalize.finalize()`
(the EP reduce-scatter) around the per-expert compute. Calling
`quant_method.apply()` does not bypass this path; it dispatches to the same
modular kernel.

Our new `FusedMoE.forward_pre_routed` method bypasses the modular kernel
entirely and calls `fused_experts(...)` directly:

```python
def forward_pre_routed(self, hidden_states, topk_ids, topk_weights):
    return fused_experts(
        hidden_states=hidden_states,
        w1=self.w13_weight,
        w2=self.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=self.activation,
        global_num_experts=self.global_num_experts,
        expert_map=self.expert_map,    # marks non-local experts as -1
        quant_config=quant_config,
    )
```

Without this bypass, pre-routing on the ATTN side eliminates the *need* for EP
collectives, but the collectives still execute inside the kernel — the
performance benefit is lost. The bypass is what makes the architecture
end-to-end coherent.

### Static-shape broadcast (Option B)

The ideal pre-routing protocol sends only the subset of tokens that have at
least one local expert on each FFN partner: ATTN computes per-partner masks
$m_j = \bigvee_k \mathbb{1}[\text{topk\_ids}_{\cdot k} \in \mathcal{E}_j]$ and
sends $h[m_j]$ to partner $j$. However, `h[m_j]` returns a tensor whose first
dimension is data-dependent. PyTorch must allocate the output on the host side,
which forces a CPU/GPU synchronization that drains the entire pending NCCL
queue (sends from the previous layer, FFN compute, FFN sends back). We
measured this drain cost at ~400 ms per MoE layer on H200, completely
swamping the bandwidth savings from the smaller transfer.

We therefore adopt **Option B: fixed-size broadcast**. Each ATTN sends the full
$[N_{\text{tot}}/M, H]$ hidden state (and topk metadata) to every FFN partner.
Shapes are statically known from the once-per-forward `dp_metadata` broadcast,
so no per-layer CPU/GPU sync is needed. The FFN side runs the routed-expert
compute over all received tokens, with `expert_map` filtering ensuring only
local experts contribute.

For top-6 of 64 experts split across $N$ FFN workers, the probability that a
token has *no* local expert on a given worker is
$\binom{E - E/N}{k} / \binom{E}{k}$. For $N = 2$ this is only 1.9%, so the
bandwidth penalty for broadcasting vs. true masking is negligible. For $N \ge 4$
the penalty grows (20% at $N=4$, 43% at $N=8$), at which point a sync-free
masking implementation (e.g., via CUDA graph capture or pre-allocated max-size
buffers with a single upfront sync) becomes worthwhile.

### Communication cost per MoE layer

Under Option B with the $M \times N$ topology and $N_{\text{tot}}$ total tokens
across $M$ ATTN ranks:

| Stage | Volume |
|-------|-------:|
| ATTN → FFN P2P ($M N$ sends, full per-rank slice) | $N \cdot N_{\text{tot}} \cdot H$ |
| FFN → ATTN P2P ($M N$ partials, full per-rank slice) | $N \cdot N_{\text{tot}} \cdot H$ |
| Topk metadata (negligible: $K=6$ int32 per token) | $\ll N_{\text{tot}} H$ |
| **EP collectives** | $0$ |
| **Total bytes (asymptotic)** | $\sim 2 N \cdot N_{\text{tot}} \cdot H$ |

For 2A2F, this is $\sim 4 N_{\text{tot}} H$ — **slightly less than the original
design's $\sim (R+2) N_{\text{tot}} H = 4 N_{\text{tot}} H$** for the same $R = 2$.
Bandwidth-wise the two designs are comparable for symmetric small configurations.
For 1A2F (the asymmetric case the original could not handle), the new design
sends $2 N_{\text{tot}} H$ vs. an undefined baseline.

The structural wins are not in bandwidth but in:

1. **No collective coordination between FFN workers** — each FFN runs a pure
   expert-parallel-aware kernel, no inter-FFN synchronization.
2. **Asymmetric configurations are supported natively.**
3. **Sync-free hot path** — Option B's static shapes plus `expert_map` filtering
   eliminate every per-layer CPU/GPU sync that the original design needed.

---

## Side-by-Side Summary

| Property | Original (1:1 + EP) | New ($M\times N$ + Pre-Routing) |
|----------|---------------------|---------------------------------|
| Pairing | 1:1, $R$ pairs | $M \times N$ bipartite, $MN$ pairs |
| Asymmetric configs | Blocked by assertion | Native support |
| Router location | FFN side | ATTN side |
| Shared experts location | FFN side | ATTN side |
| EP collective per MoE layer | All-gather + reduce-scatter | None |
| Per-layer CPU/GPU syncs | 1+ (`.item()`-style) | 0 (static shapes) |
| Sends per MoE layer (sym 2A2F) | 4 P2P + 2 collectives | 8 P2P |
| Total bytes per layer (2A2F) | $\sim 4 N_{\text{tot}} H$ | $\sim 4 N_{\text{tot}} H$ |
| FFN MoE kernel entry point | `FusedMoEModularKernel.forward` | `fused_experts` (direct) |
| DP coordination backend | Gloo CPU all-reduce | NCCL GPU all-reduce |

## Empirical Validation

Measured on a single H200 node (NVSwitch fabric, NV18 between all GPU pairs)
running DeepSeek-V2-Lite, batch 20 concurrent requests, 128-token prompt,
32-token output. Numbers are warm-state TPOT (median over second of two
back-to-back runs to exclude Triton JIT compilation time):

| Config | Original TPOT | New TPOT | Speedup |
|--------|-------------:|--------:|--------:|
| 1A1F | 54 ms | 54 ms | 1.0× (no change expected; symmetric & EP=1 elides the collective in both designs) |
| 1A2F | 1,687 ms (asymmetric, partially broken in original) | 138 ms | 12.2× |
| 2A2F | 2,076 ms (L40S, original) | 311 ms | 6.7× |
| 3A1F | (unsupported) | 62 ms | n/a |
| 2A1F | (unsupported) | 55 ms | n/a |

The ~3× residual gap between 2A2F and 1A1F is structural: 4 NCCL pairs per
layer instead of 1, plus DP synchronization between the two ATTN ranks. We
believe this gap can be closed further by enabling CUDA graph capture (currently
disabled via `--enforce-eager` due to a connector-side initialization race that
is orthogonal to the pairing redesign) and by implementing dual-batch overlap
(DBO), neither of which is in scope for this section.

---

## File-Level Changes

The redesign is implemented across four files:

| File | Change |
|------|--------|
| `vllm/distributed/afd_transfer/afd_connector/p2p_connector.py` | Full rewrite of `init_afd_connector` (1:1 → $M \times N$); new `send_attn_output` and `recv_attn_output` carrying topk metadata; element-wise partial combine in `recv_ffn_output` |
| `vllm/model_executor/models/deepseek_v2.py` | New `DeepseekV2MoEAttentionStub`; `forward_with_afd` now calls `compute_route_and_shared` on ATTN before send; `compute_ffn_output` calls `forward_pre_routed`; weight loading filter accepts gate + shared_experts on ATTN |
| `vllm/model_executor/layers/fused_moe/layer.py` | New `FusedMoE.forward_pre_routed` method bypassing `FusedMoEModularKernel` |
| `vllm/config/vllm.py` | Exception in the auto-detection of `disable_nccl_for_dp_synchronization` so AFD configurations keep NCCL GPU all-reduce instead of falling back to Gloo CPU |

The aggregate diff is approximately 1000 lines, of which ~700 are deletions
from the original asymmetric handling code that became unnecessary under the
unified bipartite model.
