# AFD (Attention–FFN Disaggregation) Communication Structure

Derived from the `no-profiler/{attn,ffn}.log` files under `configs/MaNbF/` for the
`run-20260415-220850` sweep, plus `config-metadata.json`. Model:
`deepseek-ai/DeepSeek-V2-Lite` running on vLLM with the `p2pconnector` AFD
connector.

---

## 1. The two roles

The `MaNbF` configurations split the MoE decoder across two role types:

- **Attention server(s)** — `afd_role=attention`. Run embedding, MLA attention,
  KV-cache, and sampling. Started by the OpenAI API server with
  `api_server_count=M` and `data_parallel_size=M`. One engine per attention GPU,
  process name pattern `EngineCore_DPi`.
- **FFN server(s)** — `afd_role=ffn`. Run the dense MLP, MoE router, expert
  FFNs, and the output projection / norm. Started by `afd_ffn_server.py` with
  `tensor_parallel_size=N`, process name pattern `Worker_TPi_EPi`.

DeepSeek V2-Lite has **64 routed experts**. The FFN side splits them linearly
across its `N` GPUs (visible in 4A4F as
`[EP Rank k/4] 0->16k ... 15->16k+15`).

> **Note on "TP" vs "EP" on the FFN side.** vLLM launches the FFN server with
> `tensor_parallel_size=N` and that single `N`-rank world does **two different
> jobs** depending on the sublayer:
>
> - **Dense sublayers** — the first 1–2 fully-dense MLP layers, the **shared
>   expert** inside every MoE block, and the down-projection — are sliced
>   **classically TP**: `Linear(in, out)` is column-parallel + row-parallel
>   across the `N` GPUs, finished by `tensor_model_parallel_all_reduce`.
> - **Routed-expert MoE sublayers** reinterpret the same `N`-rank world as
>   **EP**: each rank owns `64/N` experts and an all-to-all dispatches tokens
>   to whichever rank holds their top-`k` experts, then combines.
>
> Both signatures appear in the logs: `[EP Rank k/N] Expert placement
> strategy: linear` (EP semantics) **and**
> `output = tensor_model_parallel_all_reduce(output_parallel)` in the 4A4F
> crash stack (real TP on the dense / shared path). Processes are named
> `Worker_TPi_EPi` because rank `k` is **simultaneously** TP rank `k` and
> EP rank `k` in vLLM's parallel-state design.
>
> So when the table below says "FFN parallelism: TP=N", read it as "the
> `N`-way parallel world that does TP for dense / shared sublayers and EP
> for routed experts."

### Per-config layout

| Config | M (attn) | N (ffn) | GPUs | Experts/FFN GPU | FFN parallelism | Attn parallelism |
| ------ | -------- | ------- | ---- | --------------- | --------------- | ---------------- |
| 1A1F   | 1        | 1       | 2    | 64              | TP=1            | DP=1             |
| 1A2F   | 1        | 2       | 3    | 32              | TP=2            | DP=1             |
| 1A4F   | 1        | 4       | 5    | 16              | TP=4            | DP=1             |
| 2A1F   | 2        | 1       | 3    | 64              | TP=1            | DP=2             |
| 2A4F   | 2        | 4       | 6    | 16              | TP=4            | DP=2             |
| 3A1F   | 3        | 1       | 4    | 64              | TP=1            | DP=3             |
| 3A2F   | 3        | 2       | 5    | 32              | TP=2            | DP=3             |
| 4A2F   | 4        | 2       | 6    | 32              | TP=2            | DP=4             |
| 4A4F   | 4        | 4       | 8    | 16              | TP=4            | DP=4             |
| 6A2F   | 6        | 2       | 8    | 32              | TP=2            | DP=6             |
| 7A1F   | 7        | 1       | 8    | 64              | TP=1            | DP=7             |

All configs use `num_afd_stages=1` — a single attention→FFN→attention pipeline
stage with no microbatch pipelining.

---

## 2. The two communication planes

`p2p_connector.py` builds two layered process groups, both visible in the logs.

### 2.1 Global `afd_pg` — Gloo, world-size `M+N` (control plane)

Every process — every FFN worker and every attention engine — joins one big
Gloo PG.

```text
1A1F : "Gloo Rank 0/1 connected to 1 peer ranks. Expected: 1"
2A1F : "Gloo Rank 0..2 ... Expected: 2"
2A4F : "Gloo Rank 0..5 ... Expected: 5"
4A4F : "Gloo Rank 0..7 ... Expected: 7"
7A1F : "Gloo Rank 0..7 ... Expected: 7"
```

Each engine then logs `afd_pg initialized world_rank=<R>`. World-rank assignment
is **FFN ranks first, attention ranks after**:

| Config | FFN world ranks      | Attention world ranks |
| ------ | -------------------- | --------------------- |
| 1A1F   | 0                    | 1                     |
| 2A1F   | 0                    | 1, 2                  |
| 2A4F   | 0..3                 | 4, 5                  |
| 4A4F   | 0..3 (Worker_TP0..3) | 4..7 (DP0..DP3)       |
| 7A1F   | 0                    | 1..7                  |

**Why Gloo for this layer.** It's CPU-side, supports `new_group` cheaply,
doesn't need a CUDA context for every rank, and is used for low-frequency
control messages (handshakes, shape/metadata broadcasts, profiler control,
shutdown). NCCL would be wrong here — NCCL communicators are sticky and
expensive to create, and would force GPU synchronization for tiny control
messages.

### 2.2 Per-pair sub-groups — Gloo coordination + NCCL data, world-size 2

For each `(attn_i, ffn_j)` cell of the `M × N` bipartite, a 2-rank sub-group is
created with `torch.distributed.new_group` over `afd_pg`. The log line is:

```text
creating pair attn=A ffn=F pair_id=P rank_in_pair=R
```

with:

- **`pair_id = A * N + F`**
- **`rank_in_pair = 0` for FFN, `1` for attention**

Example from 4A4F:

```text
FFN side, world_rank=0:
  attn=0 ffn=0 pair_id=0   rank_in_pair=0
  attn=1 ffn=0 pair_id=4   rank_in_pair=0
  attn=2 ffn=0 pair_id=8   rank_in_pair=0
  attn=3 ffn=0 pair_id=12  rank_in_pair=0
  → "world_rank=0 role=ffn created 4 pair groups (M×N bipartite 4A4F)"

Attn side, world_rank=4:
  attn=0 ffn=0 pair_id=0   rank_in_pair=1
  attn=0 ffn=1 pair_id=1   rank_in_pair=1
  attn=0 ffn=2 pair_id=2   rank_in_pair=1
  attn=0 ffn=3 pair_id=3   rank_in_pair=1
  → "world_rank=4 role=attention created 4 pair groups (M×N bipartite 4A4F)"
```

Counts the logs print:

- Each **attention** ends with `created N pair groups` (one per FFN partner).
- Each **FFN** ends with `created M pair groups` (one per attention partner).
- Total pair groups across the cluster = `M × N`.

The `[Gloo] Rank 1 is connected to 1 peer ranks. Expected: 1` line appearing
twice per `pair_id` is the Gloo store handshake for that 2-rank sub-group;
both ends print on connection. Then on top of that 2-rank sub-group, an
**NCCL communicator** is initialized (`vLLM is using nccl==2.27.5`) for the
actual GPU↔GPU tensor transfer.

### 2.3 Why both Gloo and NCCL on the same pair group

- **Gloo PG** carries small CPU-resident messages: per-step request count,
  token-count headers, sequence ids, "ready" / "done" flags, profiler control.
  Tiny, latency-sensitive, no need to touch the GPU.
- **NCCL on a CUDA stream** carries the bulk GPU tensors: post-attention
  `hidden_states` (attn → ffn) and post-MLP `hidden_states` (ffn → attn).
  Uses NVLink/PCIe direct GPU memcpy.

---

## 3. What flows on each link, per layer

For every transformer layer where the MLP is offloaded:

```text
ATTENTION GPU (rank_in_pair=1)            FFN GPU (rank_in_pair=0)
─────────────────────────────             ─────────────────────────
RMSNorm + MLA(qkv, kv-cache)
   │
   │  [Gloo]  header: (n_tokens, dtype, layer_id)        ───►
   │  [NCCL]  hidden_states[T, hidden]  send (cuda stream)──►
   │                                                          │
   │                                              router → top-k experts
   │                                              dispatch (TP+EP all-to-all
   │                                              within the FFN side over its
   │                                              own TP NCCL group, NOT the
   │                                              pair group)
   │                                              expert FFN GEMMs
   │                                              combine + tensor_model_parallel_all_reduce
   │                                                          │
   │  ◄─── NCCL  hidden_states[T, hidden]  recv (cuda stream) │
   │  ◄─── Gloo  ack / next-iter token (small)                │
add residual, next layer
```

The error stack from `4A4F/no-profiler/ffn.log` confirms the FFN-internal
NCCL group:

```text
output = tensor_model_parallel_all_reduce(output_parallel)
  → get_tp_group().all_reduce(input_)           ← FFN's internal TP NCCL group
hidden_states = self.mlp(hidden_states)         ← MoE dispatch/combine
```

---

## 4. The three NCCL "domains"

When both `M > 1` and `N > 1` you have three distinct NCCL communicators in
play:

1. **TP NCCL group** — across the `N` FFN GPUs of one FFN side. Built by vLLM's
   `parallel_state` (`world_size=N rank=k backend=nccl
   distributed_init_method=tcp://...`). Used for
   `tensor_model_parallel_all_reduce` and the MoE all-to-all (handled inside
   the FFN side via `AgRsAll2AllManager` etc.). Each FFN process is
   `Worker_TPi_EPi`.
2. **Attention DP NCCL group** — across the `M` attention GPUs. Built by vLLM
   (`world_size=M`, also nccl). Coordinated by the **DP Coordinator** process
   (you can see `Started DP Coordinator process (PID: …)` in the 4A4F attn
   log; later `coordinator.py:326 Received stats for out-of-order step (1,32)
   from engine 1`). Carries DP-rank scheduling stats. Does **not** touch FFN.
3. **Pair NCCL stream** — the `M × N` bipartite, one stream per
   `(attn, ffn)` cell. Carries only the cross-role hidden-state exchange.
   Each pair runs send/recv on its own CUDA stream so multiple pairs can
   overlap on the same GPU.

### Streams per role, per config

| Config | Streams on each attn GPU | Streams on each ffn GPU | Internal NCCL groups inside FFN side |
| ------ | ------------------------ | ----------------------- | ------------------------------------ |
| 1A1F   | 1                        | 1                       | none (TP=1)                          |
| 1A2F   | 2                        | 1                       | TP-NCCL(2)                           |
| 1A4F   | 4                        | 1                       | TP-NCCL(4)                           |
| 2A1F   | 1                        | 2                       | DP-NCCL(2) on attn side              |
| 2A4F   | 4                        | 2                       | TP-NCCL(4) + DP-NCCL(2)              |
| 3A1F   | 1                        | 3                       | DP-NCCL(3)                           |
| 3A2F   | 2                        | 3                       | TP-NCCL(2) + DP-NCCL(3)              |
| 4A2F   | 2                        | 4                       | TP-NCCL(2) + DP-NCCL(4)              |
| 4A4F   | 4                        | 4                       | TP-NCCL(4) + DP-NCCL(4)              |
| 6A2F   | 2                        | 6                       | TP-NCCL(2) + DP-NCCL(6)              |
| 7A1F   | 1                        | 7                       | DP-NCCL(7)                           |

The `*A1F` configs collapse the FFN-internal TP/EP all-reduce because there's
only one FFN GPU; you can confirm this in 1A1F/2A1F/3A1F/7A1F where
`tensor_parallel_size=1` in the FFN startup log and the FFN process is just
plain `INFO ...` (no `Worker_TPi`).

---

## 5. Why the staggered group creation

In 4A4F the timestamps show pair_groups being added **one at a time, with a
Gloo handshake per pair**:

```text
00:38:55  attn=0 ffn=0 pair_id=0
00:38:56  attn=0 ffn=1 pair_id=1
00:38:57  attn=0 ffn=2 pair_id=2 ; attn=1 ffn=1 pair_id=5
00:38:57  attn=0 ffn=3 pair_id=3
...
00:39:04  attn=3 ffn=3 pair_id=15  → done
```

`torch.distributed.new_group` is a **collective on `afd_pg`** — every rank
must call it in the same order even if they don't participate. That's why
every `[Gloo] Rank N is connected to 1 peer` line has its mirror on the
other side: each `pair_id` round, the two ranks that join actually shake
hands; everyone else just enters and exits the collective. With `M × N`
up to 16, you see ~16 of these rounds back-to-back during init.

---

## 6. Summary diagram

```text
                ┌─────────────── afd_pg (Gloo, world=M+N) ───────────────┐
                │  control: handshakes, shapes, shutdown, profiler       │
                └─────────────────────────────┬──────────────────────────┘
                                              │ new_group per (a,f)
                                              ▼
            M×N pair groups: each is 2-rank, dual backend
              ├─ Gloo sub-PG : per-step headers / acks (CPU)
              └─ NCCL comm   : hidden_states send/recv (GPU, CUDA stream)
                                rank_in_pair: ffn=0, attn=1
                                pair_id     = attn_idx * N + ffn_idx

   Attention side (M GPUs)                    FFN side (N GPUs)
   ─────────────────────                      ─────────────────
   DP-NCCL across M ranks                     TP-NCCL across N ranks
   (vLLM data-parallel,                       (all_reduce in MLP,
    DP Coordinator)                            MoE EP all-to-all)
```

Per layer: attention computes hidden states → Gloo header + NCCL send on its
pair stream(s) → FFN receives, runs router/expert FFNs (with internal TP
all-reduce on its own NCCL group) → NCCL sends back → attention adds residual
and proceeds. Gloo handles small/CPU bookkeeping; NCCL handles the bulk GPU
tensors; the per-pair group keeps every (attention, FFN) link independent so
an `M × N` grid can stream concurrently.

---

## 7. Design discussion: should the router move to the attention side?

**Short answer.** Maybe — but the win only shows up if you also change *what*
gets sent over the pair link, not just *where* the router runs.

### 7.1 Why the bare swap is mostly a wash

The router is a single `Linear(hidden, n_experts) + topk + softmax` — tiny
compute. Moving it to attn just relocates ~µs of work and adds the gate
weights to every attention GPU. The bytes on the attn↔ffn pair NCCL stream
don't shrink: you still send `hidden_states[T, hidden]` plus now also
`topk_ids[T, k] + topk_w[T, k]`. Slightly **more** traffic, not less.

### 7.2 Where it actually pays off — expert-aware dispatch

If router-on-attn lets attn segment tokens by which FFN owns the chosen
experts, two things change in the `N > 1` configs (1A2F, 1A4F, 2A4F, 3A2F,
4A2F, 4A4F, 6A2F):

1. **Smaller attn → ffn payload.** Attn no longer needs to broadcast full
   `hidden_states` to every FFN partner — it sends only the tokens whose
   top-k experts live on that FFN. With `k = 6` and 16 experts/FFN in the
   4F configs, a token typically hits 2–4 FFNs out of 4, so you save
   ~25–50% on the attn → ffn link.
2. **No EP all-to-all on FFN side.** The `tensor_model_parallel_all_reduce`
   visible in the 4A4F crash trace is precisely that step; with pre-routed
   dispatch each FFN already has only its own tokens and produces partial
   outputs that attn combines additively.

### 7.3 Where it doesn't help

All `*A1F` configs (1A1F, 2A1F, 3A1F, 7A1F) — `N = 1`, nothing to dispatch
**to**. Pure overhead: router weights duplicated `M` times, no comms savings.
Also, the FFN-side router currently amortizes its weights across all `M`
attention partners; moving it to attn turns 1 copy into `M` copies.

### 7.4 Other costs to weigh

- Variable per-pair payload size (need a Gloo header per step with the
  per-FFN token count).
- More book-keeping for combine on attn (gather `k` partial outputs,
  weighted sum).
- You lose the symmetric, predictable shapes that make the current pair
  NCCL streams easy to overlap.

### 7.5 Recommendation

For the `N ≥ 2` configs with cross-NUMA / PCIe attn↔ffn links, yes —
router-on-attn with **expert-aware dispatch** is the right move and is what
most production MoE-disaggregation stacks (DeepSpeed-MoE, Tutel-style) end
up doing. For the `N = 1` configs, leave it on the FFN.

---

## 8. Glossary

- **AFD** — Attention-FFN Disaggregation. Splitting a transformer's attention
  and feed-forward blocks onto different GPUs / processes that communicate
  per layer.
- **MoE** — Mixture of Experts. The FFN is replaced by a router that selects
  `top-k` of `E` expert FFNs per token.
- **TP (Tensor Parallel)** — slice each weight matrix across `N` GPUs;
  combine via all-reduce.
- **EP (Expert Parallel)** — place different MoE experts on different GPUs;
  combine via all-to-all.
- **DP (Data Parallel)** — run independent batch shards on `M` replicas.
- **`afd_pg`** — the global Gloo process group containing all `M + N`
  AFD processes.
- **Pair group** — a 2-rank `(attn_i, ffn_j)` sub-group with both Gloo (for
  CPU control) and NCCL (for GPU tensor transfer).
- **`pair_id`** — flat index of a pair group, `attn_idx * N + ffn_idx`.
- **`rank_in_pair`** — `0` for the FFN end, `1` for the attention end.

---

## 9. Source pointers (for follow-up)

- Connector init log lines: `p2p_connector.py:332` (`init_afd_connector
  begin`), `:366` (`afd_pg initialized`), `:385` (`creating pair`), `:430`
  (final `created N/M pair groups`).
- FFN entry point: `afd_ffn_server.py` (`Start AFD FFN Server`,
  `FFN worker loop started`, `FFN server loop running`).
- Per-layer crash trace showing the FFN call chain (4A4F):
  `self.mlp(hidden_states)` → `tensor_model_parallel_all_reduce` →
  `get_tp_group().all_reduce`.
- DP coordinator (4A4F attn): `Started DP Coordinator process (PID: …)`,
  later `coordinator.py:200 All engine subscriptions received`.
