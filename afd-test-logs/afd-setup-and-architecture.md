# AFD (Attention-FFN Disaggregation) — Complete Setup & Architecture Guide

A detailed walkthrough of how AFD works in vLLM: what it is, why it exists for MoE models, how it initializes, how communication groups are created, and how inference flows end-to-end.

---

## Table of Contents

1. [What is AFD and Why It Exists](#1-what-is-afd-and-why-it-exists)
2. [MoE Models and the Motivation for Disaggregation](#2-moe-models-and-the-motivation-for-disaggregation)
3. [AFD in vLLM — High-Level Architecture](#3-afd-in-vllm--high-level-architecture)
4. [DP and TP in the Context of AFD](#4-dp-and-tp-in-the-context-of-afd)
5. [How AFD Gets Launched (CLI and Config)](#5-how-afd-gets-launched-cli-and-config)
6. [Model Loading — What Each Side Loads](#6-model-loading--what-each-side-loads)
7. [Process Group Initialization — The Full Story](#7-process-group-initialization--the-full-story)
8. [Per-Pair NCCL Communicator Creation](#8-per-pair-nccl-communicator-creation)
9. [Metadata Exchange (Gloo Control Plane)](#9-metadata-exchange-gloo-control-plane)
10. [Inference Flow — End to End](#10-inference-flow--end-to-end)
11. [Config-Specific Communication Patterns](#11-config-specific-communication-patterns)
12. [Key Code Files Reference](#12-key-code-files-reference)

---

## 1. What is AFD and Why It Exists

**AFD (Attention-FFN Disaggregation)** splits a transformer model's decoder layers so that:
- **Attention layers** (self-attention + KV cache) run on one set of GPUs
- **FFN/MoE layers** (feed-forward / mixture-of-experts) run on a different set of GPUs

The two sides communicate the intermediate hidden states over NCCL point-to-point links every layer.

**Why bother?** In MoE models like DeepSeek-V2 and DeepSeek-V3:
- The **attention** portion is relatively lightweight but needs fast access to the KV cache (memory-bound)
- The **MoE FFN** portion is compute-heavy (64+ experts, router, weighted sum) but doesn't need KV cache
- Their resource profiles are fundamentally different — attention wants memory bandwidth, MoE wants compute FLOPs

By putting them on separate GPUs, each side can be independently scaled and optimized:
- Add more attention GPUs (DP) to handle more concurrent sequences
- Add more FFN GPUs (TP/EP) to parallelize expert computation
- Each GPU's memory holds only the weights it needs (attention OR MoE, not both)

---

## 2. MoE Models and the Motivation for Disaggregation

### How MoE Works in DeepSeek-V2

Each decoder layer has:
```
Input → RMSNorm → MLA Attention → RMSNorm → MoE FFN → Output
                                              │
                                     ┌────────┴────────┐
                                     │   MoE Router    │
                                     │  (gating network) │
                                     └────────┬────────┘
                                              │
                              Selects top-K of 64 experts
                                              │
                           ┌──────┬──────┬──────┬──────┐
                           │ E₀   │ E₁   │ ...  │ E₆₃  │  (each expert = small FFN)
                           └──────┴──────┴──────┴──────┘
                                              │
                                   Weighted sum of outputs
```

- **64 routed experts** + shared experts per MoE layer
- Router selects **top-K** experts per token (typically K=6)
- Each expert is a small FFN (gate_proj + up_proj + down_proj)
- The weighted sum of selected expert outputs becomes the layer output

### Why MoE Makes Disaggregation Attractive

| Property | Attention | MoE FFN |
|----------|-----------|---------|
| Memory need | KV cache (grows with sequence length) | Expert weights (fixed, large) |
| Compute pattern | Memory-bandwidth bound (Q×K^T attention) | Compute bound (expert matrix multiplies) |
| Parallelism | Data Parallel (each GPU handles different requests) | Expert Parallel (each GPU holds different experts) + Tensor Parallel (each GPU holds weight slices) |
| Weight size (DeepSeek-V2-Lite) | ~300M params (attention only) | ~12B params (64 experts × ~190M each) |

Without AFD, a single GPU must hold both attention weights AND all expert weights, limiting batch size. With AFD, attention GPUs only load ~300M params (plus KV cache), and FFN GPUs only load the expert weights.

---

## 3. AFD in vLLM — High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   USER REQUEST                          │
│              curl /v1/completions                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              ATTENTION SERVER (vllm serve)               │
│                                                         │
│  Scheduler → Tokenize → KV Cache Alloc → Model Forward  │
│                                                         │
│  Model Forward per decoder layer:                       │
│    1. RMSNorm → MLA Attention → post_attention_layernorm│
│    2. ──NCCL send──► hidden_states to FFN               │
│    3. ◄──NCCL recv── MoE output from FFN                │
│    4. residual += MoE output → next layer               │
│                                                         │
│  After all layers: RMSNorm → LM Head → Sample → Detok  │
└─────────────────────────────────────────────────────────┘
                     ▲         │
                     │  NCCL   │  NCCL
                     │  P2P    │  P2P
                     │         ▼
┌─────────────────────────────────────────────────────────┐
│              FFN SERVER (vllm fserver)                   │
│                                                         │
│  Worker Loop (blocks on recv, runs forever):            │
│    1. ◄──Gloo recv── dp_metadata from ATTN              │
│    2. For each layer:                                   │
│       a. ◄──NCCL recv── hidden_states from ATTN         │
│       b. RMSNorm → MoE Router → Expert Compute → Sum   │
│       c. ──NCCL send──► MoE output to ATTN              │
└─────────────────────────────────────────────────────────┘
```

Key points:
- The ATTN server is a full vLLM API server (handles HTTP, scheduling, sampling)
- The FFN server is a headless worker (no API, just a recv→compute→send loop)
- Communication happens **every single decoder layer** (27 times for DeepSeek-V2-Lite)

---

## 4. DP and TP in the Context of AFD

### Data Parallel (DP) — Attention Side

When `--data-parallel-size 3` is set on the ATTN server:
- vLLM spawns 3 independent engine cores (DP0, DP1, DP2), each on its own GPU
- Each DP rank runs the **same** attention model but processes **different** requests
- The scheduler distributes requests across DP ranks (load balancing)
- DP ranks do NOT communicate with each other during inference

```
GPU 0: ATTN DP0 — handles requests A, D, G, ...
GPU 1: ATTN DP1 — handles requests B, E, H, ...
GPU 2: ATTN DP2 — handles requests C, F, I, ...
```

### Tensor Parallel (TP) + Expert Parallel (EP) — FFN Side

When `--tensor-parallel-size 2 --enable-expert-parallel` is set on the FFN server:
- vLLM spawns 2 FFN workers (TP0, TP1), each on its own GPU
- **Tensor Parallel**: Each worker holds a **slice** of every expert's weights (e.g., half the hidden dimension). All workers must process the **same tokens** and allreduce their partial results.
- **Expert Parallel**: The 64 experts are distributed across workers (e.g., TP0 holds experts 0-31, TP1 holds experts 32-63). An all-to-all shuffle routes tokens to the correct expert owner.

```
GPU 2: FFN TP0_EP0 — expert weights 0-31 (sliced), processes ALL tokens
GPU 3: FFN TP1_EP1 — expert weights 32-63 (sliced), processes ALL tokens
                     ↕ EP all-to-all (route tokens to expert owners)
                     ↕ TP allreduce (combine partial weight results)
```

**Critical insight:** TP requires all workers to see the **same input tokens** (they split weights, not data). This is why 1A2F broadcasts the full tensor to both FFN workers rather than chunking.

### How DP and TP Interact with AFD Pairing

| Config | ATTN parallelism | FFN parallelism | Pairing |
|--------|-----------------|----------------|---------|
| **1A1F** | None | None | 1 pair: ATTN↔FFN |
| **4A4F** | DP=4 | TP=4 + EP=4 | 4 pairs: ATTN_DPi ↔ FFN_TPi (1:1) |
| **3A1F** | DP=3 | None | 3 pairs: each ATTN_DPi ↔ same FFN (many:1) |
| **1A2F** | None | TP=2 + EP=2 | 2 pairs: same ATTN ↔ each FFN_TPi (1:many) |

---

## 5. How AFD Gets Launched (CLI and Config)

### Two Separate Processes

AFD requires launching **two** separate vLLM processes (always start FFN first):

**FFN Server** — `vllm fserver`:
```bash
CUDA_VISIBLE_DEVICES=3 vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn",
                  "afd_host":"127.0.0.1","afd_port":"29500",
                  "num_afd_stages":"1",
                  "afd_extra_config":{"afd_size":"3A1F"}}'
```

**ATTN Server** — `vllm serve`:
```bash
CUDA_VISIBLE_DEVICES=0,1,2 vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --data-parallel-size 3 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention",
                  "afd_host":"127.0.0.1","afd_port":"29500",
                  "num_afd_stages":"1",
                  "afd_extra_config":{"afd_size":"3A1F"}}'
```

### AFDConfig (`vllm/config/afd.py`)

The `--afd-config` JSON is parsed into:

```python
@dataclass
class AFDConfig:
    afd_connector: str = "dummy"         # "p2pconnector" for real communication
    afd_role: Literal["attention", "ffn"] # which side this process is
    afd_port: int = 1239                  # TCP port for Gloo rendezvous
    afd_host: str = "127.0.0.1"          # host for rendezvous
    num_afd_stages: int = 3              # pipeline micro-batching stages
    afd_extra_config: dict = {}          # {"afd_size": "3A1F"} etc.
```

### Entrypoint Flow

**`vllm fserver`** → `vllm/entrypoints/cli/fserver.py` → `afd_ffn_server.py`:
1. Creates `VllmConfig` from args (with `afd_role="ffn"`)
2. Creates `Executor` → spawns `GPUWorker` instances (one per FFN GPU)
3. Each worker creates `GPUFFNModelRunner` (instead of `GPUModelRunner`)
4. Calls `collective_rpc("start_ffn_server_loop")` → each worker starts a `ffn_worker_loop` thread
5. Main thread blocks forever until Ctrl+C

**`vllm serve`** → `vllm/entrypoints/openai/api_server.py` (standard path):
1. Creates `VllmConfig` from args (with `afd_role="attention"`)
2. Standard vLLM startup: Executor → Workers → `GPUModelRunner`
3. Model runner detects `afd_config` and initializes AFD connector
4. Starts HTTP API server, handles requests normally

---

## 6. Model Loading — What Each Side Loads

### Attention Server — Loads ONLY Attention Weights

In `DeepseekV2DecoderLayer.__init__()` (`deepseek_v2.py`):

```python
# Only create attention module if role is attention (or no AFD)
if self.afd_role is None or self.afd_role == "attention":
    self.self_attn = DeepseekV2MLAAttention(...)

# Only create MoE/FFN module if role is ffn (or no AFD)
if self.afd_role is None or self.afd_role == "ffn":
    self.mlp = DeepseekV2MoE(...)  # or DeepseekV2MLP for dense layers
```

In `load_weights()`:
```python
# Skip MoE weights entirely on attention side
if self.afd_role == "attention" and self.is_moe_weight(name):
    continue
```

The attention side also has `input_layernorm` and `post_attention_layernorm` (RMSNorms), the embedding table, and the final LM head.

### FFN Server — Loads ONLY MoE/FFN Weights

The FFN model runner (`GPUFFNModelRunner`) loads the same `DeepseekV2ForCausalLM` model but with `afd_role="ffn"`. Due to the conditional `__init__`:
- No `self_attn` module is created → no attention weights loaded
- `self.mlp` is created → MoE expert weights loaded
- `post_attention_layernorm` is loaded (FFN needs it for the pre-FFN RMSNorm)

### Memory Savings

For DeepSeek-V2-Lite on a single GPU:
| Component | Params | Memory (FP16) |
|-----------|--------|---------------|
| Full model | ~15.7B | ~31 GB |
| Attention only | ~300M + KV cache | ~0.6 GB + KV cache |
| MoE/FFN only | ~12B (64 experts) | ~24 GB |

---

## 7. Process Group Initialization — The Full Story

This is the most complex part of AFD setup. Here's exactly what happens in `P2PAFDConnector.init_afd_connector()` (`p2p_connector.py`):

### Step 1: Parse the AFD Size

```python
afd_size = "3A1F"  # from afd_extra_config
attn_size, ffn_size = 3, 1  # parsed via regex
```

Key derived values:
```python
min_size = min(attn_size, ffn_size)  # 1
max_size = max(attn_size, ffn_size)  # 3
is_tp_ffn = (attn_size == 1 and ffn_size > 1)  # False for 3A1F
```

### Step 2: Compute World Ranks

Each process gets a `world_rank` in the global AFD space:
```python
# FFN ranks: 0 .. ffn_size-1
# ATTN ranks: ffn_size .. ffn_size+attn_size-1
self.world_rank = self.rank if role == "ffn" else self.rank + ffn_size
```

For 3A1F (ffn_size=1, attn_size=3):
```
FFN (rank 0)   → world_rank = 0
ATTN DP0 (rank 0) → world_rank = 0 + 1 = 1
ATTN DP1 (rank 1) → world_rank = 1 + 1 = 2
ATTN DP2 (rank 2) → world_rank = 2 + 1 = 3
```

### Step 3: Global Gloo Rendezvous (`afd_pg`)

All processes join a single global Gloo process group:

```python
afd_pg = init_afd_process_group(
    backend="gloo",                          # CPU-based, no GPU mapping issues
    init_method="tcp://127.0.0.1:29500",     # TCP store for rendezvous
    world_size=ffn_size + attn_size,         # 4 for 3A1F
    rank=self.world_rank,                    # 0, 1, 2, or 3
    group_name="afd",
)
```

**Why Gloo, not NCCL?**
NCCL barriers hang with heterogeneous `CUDA_VISIBLE_DEVICES` because NCCL guesses device ID from global rank. If FFN has `CUDA_VISIBLE_DEVICES=3` (device 0 locally) but world_rank=0, and ATTN DP2 has `CUDA_VISIBLE_DEVICES=0,1,2` (device 2 locally) with world_rank=3, NCCL gets confused. Gloo barriers are CPU-based and have no such issue.

### Step 4: Per-Pair Standalone Gloo Groups

This is where the asymmetric fix is critical. For each pair (ATTN↔FFN), we create **independent** Gloo process groups with their own TCP stores:

```python
for i in range(max_size):  # 3 iterations for 3A1F
    ffn_rank_i = ffn_ranks[i % ffn_size]    # 0, 0, 0 (same FFN for all)
    attn_rank_i = attn_ranks[i % attn_size]  # 1, 2, 3 (different ATTNs)
    pair_ranks = [ffn_rank_i, attn_rank_i]

    if self.world_rank not in pair_ranks:
        continue  # Skip if not in this pair

    rank_in_pair = pair_ranks.index(self.world_rank)  # 0=FFN, 1=ATTN

    a2e_port = 29500 + 100 + i * 2      # 29600, 29602, 29604
    e2a_port = 29500 + 100 + i * 2 + 1  # 29601, 29603, 29605

    a2e_pg = init_afd_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{a2e_port}",
        world_size=2,
        rank=rank_in_pair,
        group_name=f"a2e_{i}",
    )
```

For 3A1F, this creates:
```
Pair 0: FFN (rank_in_pair=0) ↔ ATTN DP0 (rank_in_pair=1)  ports 29600/29601
Pair 1: FFN (rank_in_pair=0) ↔ ATTN DP1 (rank_in_pair=1)  ports 29602/29603
Pair 2: FFN (rank_in_pair=0) ↔ ATTN DP2 (rank_in_pair=1)  ports 29604/29605
```

**Why standalone groups instead of `torch.distributed.new_group()`?**

`new_group()` uses a global counter (`_group_count`) that ALL ranks must increment in lockstep. But FFN (DP=1) and ATTN (DP=3) create different numbers of internal groups during `initialize_model_parallel()`:
- FFN (DP=1): ~12 `new_group` calls → `_group_count` reaches ~12
- ATTN (DP=3): ~36 `new_group` calls → `_group_count` reaches ~36

When AFD init runs and calls `new_group()`, FFN generates group ID 13 while ATTN generates group ID 37. They never match → **permanent deadlock**. No amount of barriers can fix this because the counters diverged before AFD init.

`init_afd_process_group()` bypasses this entirely by calling `_new_process_group_helper()` directly with its own TCP store. Only the two pair members participate:

```python
# From vllm/distributed/parallel_state.py
def init_afd_process_group(...):
    rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
    store, rank, world_size = next(rendezvous_iterator)
    store = PrefixStore(group_name, store)

    pg, _ = _new_process_group_helper(
        world_size, rank, [], backend, store,
        group_name=group_name, ...
    )
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg
```

### Result After Step 4

**FFN process** (for 3A1F):
- 3 `a2e_gloo_pgs` (one per ATTN partner)
- 3 `a2e_groups` / `a2e_comm_ids` (PairGroup + NCCL comm per pair)
- 3 `e2a_groups` / `e2a_comm_ids`

**Each ATTN process** (for 3A1F):
- 1 `a2e_gloo_pgs` (to the single FFN)
- 1 `a2e_groups` / `a2e_comm_ids`
- 1 `e2a_groups` / `e2a_comm_ids`

---

## 8. Per-Pair NCCL Communicator Creation

Each Gloo pair group also gets a corresponding NCCL communicator for high-speed GPU tensor transfer. This is done via `_create_pynccl_comm_for_pair()`:

### The Problem with Standard PyNcclCommunicator

`PyNcclCommunicator.__init__()` uses `dist.broadcast(src=global_rank)` to share the NCCL unique ID. This fails for standalone groups because:
1. `dist.broadcast(src=ranks[0])` goes through `c10d_logger`
2. Logger calls `dist.get_rank(group)` → `get_group_rank(group, default_pg.rank())`
3. For ATTN DP2 (default PG rank = 2), the standalone pair group's `pg_group_ranks = {0:0, 1:1}` — rank 2 isn't in there
4. → `ValueError: Global rank 2 is not part of group`

### The Solution: Direct Gloo Send/Recv

```python
def _create_pynccl_comm_for_pair(gloo_pg, rank_in_pair, device):
    nccl = NCCLLibrary()

    if rank_in_pair == 0:  # FFN
        unique_id = nccl.ncclGetUniqueId()
        tensor = torch.ByteTensor(list(unique_id.internal))
        gloo_pg.send([tensor], 1, 0).wait()   # Direct Gloo send to ATTN (group rank 1)
    else:  # ATTN
        unique_id = ncclUniqueId()
        tensor = torch.ByteTensor(list(unique_id.internal))
        gloo_pg.recv([tensor], 0, 0).wait()   # Direct Gloo recv from FFN (group rank 0)
        for idx, byte in enumerate(tensor.tolist()):
            unique_id.internal[idx] = byte

    # Create NCCL communicator with exchanged unique ID
    comm = nccl.ncclCommInitRank(2, unique_id, rank_in_pair)

    # Build a PyNcclCommunicator shell
    pynccl = object.__new__(PyNcclCommunicator)
    pynccl.rank = rank_in_pair
    pynccl.world_size = 2
    pynccl.comm = comm
    # ... set other attributes ...
    return pynccl
```

**Why `gloo_pg.send()` instead of `torch.distributed.send()`?**

`torch.distributed.send(tensor, dst=0, group=gloo_pg)` also goes through `c10d_logger` → same `dist.get_rank(group)` problem. Calling `gloo_pg.send([tensor], dst_group_rank, tag).wait()` directly bypasses the logger entirely. The `dst` is a group-local rank (0 or 1), not a global rank.

### NCCL Communicators Are Used for Tensor Transfer

After setup, each pair has:
- **Gloo PG**: for metadata (dp_metadata — tiny, CPU tensors, ~100 bytes)
- **NCCL comm**: for hidden states (GPU tensors, `[N, 2048]`, ~16MB per transfer at full batch)

The NCCL comms are registered with integer IDs in `_AFD_COMMUNICATORS` dict and used via custom ops:
```python
torch.ops.vllm.afd_p2p_send(hidden_states, dst=0, comm_id=3)
torch.ops.vllm.afd_p2p_recv(buffer, src=1, comm_id=3)
```

---

## 9. Metadata Exchange (Gloo Control Plane)

Before each inference batch, ATTN sends shape information to FFN so FFN knows what tensor sizes to expect.

### What Gets Sent

```python
DPMetadata:
    max_tokens_across_dp_cpu: tensor([128])      # max across all DP ranks
    num_tokens_across_dp_cpu: tensor([128, 128, 128])  # per-DP-rank token counts
```

Plus `is_graph_capturing: bool` (whether ATTN is doing CUDA graph capture).

### Who Sends

Controlled by `is_attn_top_min_size_rank()`:

```python
def is_attn_top_min_size_rank(self, rank):
    if self.config.afd_config.afd_role != "attention":
        return False
    dp_rank = self.config.parallel_config.data_parallel_rank
    return dp_rank < self.min_size
```

| Config | min_size | Who sends metadata |
|--------|---------|-------------------|
| **4A4F** | 4 | All 4 ATTN ranks (dp_rank 0,1,2,3 all < 4) |
| **3A1F** | 1 | Only ATTN DP0 (dp_rank 0 < 1; DP1, DP2 skip) |
| **1A2F** | 1 | Only ATTN DP0 (the single ATTN) |

### How It's Sent (Direct Gloo P2P)

```python
def send_dp_metadata_list(self, data, is_graph_capturing):
    send_data = (data, is_graph_capturing)
    object_bytes = pickle.dumps(send_data)
    object_tensor = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)
    size_tensor = torch.tensor([object_tensor.numel()], dtype=torch.long)

    for i, gloo_pg in enumerate(self.a2e_gloo_pgs):
        gloo_pg.send([size_tensor], 0, 0).wait()    # send size to FFN (group rank 0)
        gloo_pg.send([object_tensor], 0, 0).wait()   # send payload to FFN
```

FFN receives:
```python
def recv_dp_metadata_list(self):
    gloo_pg = self.a2e_gloo_pgs[0]  # Always reads from pair 0

    size_tensor = torch.empty(1, dtype=torch.long)
    gloo_pg.recv([size_tensor], 1, 0).wait()    # recv from ATTN (group rank 1)

    object_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8)
    gloo_pg.recv([object_tensor], 1, 0).wait()

    data, is_graph_capturing = pickle.loads(object_tensor.numpy().tobytes())
    return data, is_graph_capturing
```

### Non-Sending ATTN Ranks

For 3A1F, ATTN DP1 and DP2 don't send metadata but still need to know tensor shapes for their own `recv_ffn_output`. They call:
```python
# In gpu_model_runner.py _prepare_inputs:
elif self.afd_config and self.afd_connector.is_initialized():
    self.afd_connector.update_state_from_dp_metadata(dp_metadata_list)
```

---

## 10. Inference Flow — End to End

Here's the complete flow when a user sends a request to the ATTN server:

### Step-by-Step (3A1F Example with 128 Input Tokens)

```
User → curl /v1/completions {"prompt": "Hello, how are you", "max_tokens": 32}

═══════════════════════════════════════════════════════════════
ATTN SERVER
═══════════════════════════════════════════════════════════════

1. HTTP handler receives request
2. Scheduler assigns request to DP rank 0 (round-robin or load-based)
3. Tokenizer converts prompt → 5 input_ids

─── _prepare_inputs (gpu_model_runner.py) ───

4. Build dp_metadata: num_tokens_across_dp = [5, 0, 0]
   (DP0 has 5 tokens, DP1 and DP2 have 0 this batch)

5. ATTN DP0 (is_attn_top_min_size_rank=True):
   → send_dp_metadata_list() via Gloo to FFN
   DP1, DP2: update_state_from_dp_metadata() locally

─── model.forward_with_afd (deepseek_v2.py) ───

6. For each decoder layer (0..26):

   Layer 0:
     a. hidden_states = embed_tokens(input_ids)  → [5, 2048]
     b. hidden_states, residual = layer.forward(positions, hidden_states, residual)
        └─ RMSNorm → MLA Attention (Q, K, V, output proj) → post_attn RMSNorm
        └─ Returns BEFORE MoE (afd_role="attention" → early return)
     c. send_attn_output(hidden_states)
        └─ ATTN DP0: NCCL send [5, 2048] to FFN via pair 0
        └─ ATTN DP1: NCCL send [0, 2048] to FFN via pair 1 (empty tensor)
        └─ ATTN DP2: NCCL send [0, 2048] to FFN via pair 2 (empty tensor)

   Layer 1..26:
     d. hidden_states = recv_ffn_output()
        └─ ATTN DP0: NCCL recv [5, 2048] from FFN
     e. Same as (b) and (c) above

   After layer 26:
     f. hidden_states = recv_ffn_output()  ← last layer's MoE output

7. RMSNorm → LM Head → Sample next token
8. Repeat for next decode step (autoregressive, 1 token at a time)
9. After max_tokens: detokenize → return JSON response

═══════════════════════════════════════════════════════════════
FFN SERVER (running concurrently on GPU 3)
═══════════════════════════════════════════════════════════════

─── ffn_worker_loop (gpu_worker.py) ───

1. Block on recv_dp_metadata_list() via Gloo
   → receives dp_metadata: num_tokens = [5, 0, 0]
   → update_state_from_dp_metadata() → tensor sizes for recv buffers

─── _ffn_forward (gpu_ffn_model_runner.py) ───

2. For each decoder layer (0..26):

   a. hidden_states, metadata = recv_attn_output()
      └─ n=3 pairs → recv from each pair:
         pair 0: NCCL recv [5, 2048] from ATTN DP0
         pair 1: NCCL recv [0, 2048] from ATTN DP1
         pair 2: NCCL recv [0, 2048] from ATTN DP2
      └─ torch.cat([...], dim=0) → [5, 2048] (empty tensors vanish)

   b. rank_ffn_output = _execute_eager_mode(hidden_states, layer_idx)
      └─ TP=1 for 3A1F, so direct call:
         model.compute_ffn_output(hidden_states, layer_idx)
           └─ layer.mlp(hidden_states)
              └─ MoE Router → top-K expert selection
              └─ Dispatch tokens to selected experts
              └─ Expert compute (gate_proj, up_proj, down_proj)
              └─ Weighted sum → [5, 2048]

   c. send_ffn_output(rank_ffn_output)
      └─ n=3 pairs → torch.chunk(output, 3, dim=0):
         chunk 0: [5, 2048] → NCCL send to ATTN DP0 (pair 0)
         chunk 1: [0, 2048] → NCCL send to ATTN DP1 (pair 1)
         chunk 2: [0, 2048] → NCCL send to ATTN DP2 (pair 2)

3. After all 27 layers, torch.cuda.synchronize()
4. Loop back to step 1, wait for next batch's metadata
```

---

## 11. Config-Specific Communication Patterns

### 4A4F (Symmetric, 1:1 Pairing)

```
ATTN DP0 ──[N₀, 2048]──► FFN TP0 ──┐
ATTN DP1 ──[N₁, 2048]──► FFN TP1 ──┤ EP all-to-all + TP allreduce
ATTN DP2 ──[N₂, 2048]──► FFN TP2 ──┤ inside compute_ffn_output
ATTN DP3 ──[N₃, 2048]──► FFN TP3 ──┘
                                      │
FFN TP0 ──[N₀, 2048]──► ATTN DP0  ←─┘ each sends own result
FFN TP1 ──[N₁, 2048]──► ATTN DP1
FFN TP2 ──[N₂, 2048]──► ATTN DP2
FFN TP3 ──[N₃, 2048]──► ATTN DP3
```

- 8 NCCL transfers per layer (4 send + 4 recv)
- Each FFN worker gets independent data from its paired ATTN
- `_execute_eager_mode()`: `tensor_model_parallel_all_gather` before MoE (so all workers see all tokens for EP routing), then slice result back

### 3A1F (Many ATTN → 1 FFN)

```
ATTN DP0 ──[N₀, 2048]──► FFN (pair 0) ──┐
ATTN DP1 ──[N₁, 2048]──► FFN (pair 1) ──┼─ concat → [N₀+N₁+N₂, 2048]
ATTN DP2 ──[N₂, 2048]──► FFN (pair 2) ──┘
                                           │
                                    MoE on combined batch
                                           │
                            chunk → 3 parts
                                           │
FFN ──[N₀, 2048]──► ATTN DP0 (pair 0)  ←─┘
FFN ──[N₁, 2048]──► ATTN DP1 (pair 1)
FFN ──[N₂, 2048]──► ATTN DP2 (pair 2)
```

- 6 NCCL transfers per layer (3 send + 3 recv)
- Single FFN processes combined batch (3× the tokens → bottleneck)
- No TP/EP needed (single GPU)

### 1A2F (1 ATTN → Many FFN TP)

```
ATTN DP0 ──[N, 2048]──► FFN TP0 (experts 0-31)   ← broadcast same tensor
ATTN DP0 ──[N, 2048]──► FFN TP1 (experts 32-63)   ← to both
                          │
                     EP all-to-all + TP allreduce
                          │
FFN TP0 ──[N, 2048]──► ATTN DP0           ← only TP rank 0 sends back
FFN TP1: skip send (is_tp_ffn && rank != 0)
```

- 3 NCCL transfers per layer (2 send + 1 recv)
- ATTN broadcasts FULL tensor (TP needs same input on all workers)
- Only TP rank 0 sends back (after allreduce, all have same result)

### Summary Table

| Config | ATTN→FFN per pair | FFN compute | FFN→ATTN per pair | NCCL ops/layer | Metadata sender |
|--------|-------------------|-------------|-------------------|---------------|----------------|
| **4A4F** | `[Nᵢ, H]` own shard | EP all-to-all + TP allreduce | `[Nᵢ, H]` own result | 8 | All 4 ATTN |
| **3A1F** | `[Nᵢ, H]` own shard | concat → MoE → chunk | `[Nᵢ, H]` own chunk | 6 | Only DP0 |
| **1A2F** | `[N, H]` broadcast | EP all-to-all + TP allreduce | `[N, H]` only TP0 | 3 | Only DP0 |

---

## 12. Key Code Files Reference

| File | Purpose |
|------|---------|
| `vllm/config/afd.py` | `AFDConfig` dataclass — parsed from `--afd-config` JSON |
| `vllm/entrypoints/afd_ffn_server.py` | `AFDFFNServer` — FFN server main class, starts worker loop |
| `vllm/entrypoints/cli/fserver.py` | `vllm fserver` CLI command |
| `vllm/distributed/parallel_state.py` | `init_afd_process_group()` — standalone process group creation bypassing `new_group` |
| `vllm/distributed/afd_transfer/afd_connector/p2p_connector.py` | `P2PAFDConnector` — core AFD logic: group creation, NCCL comms, send/recv, metadata transfer |
| `vllm/distributed/afd_transfer/afd_connector/base.py` | `AFDConnectorBase` — abstract interface for AFD connectors |
| `vllm/distributed/afd_transfer/afd_connector/factory.py` | `AFDConnectorFactory` — creates connector instances |
| `vllm/v1/worker/gpu_model_runner.py` | ATTN-side model runner — `_prepare_inputs` sends metadata, `_model_forward` triggers layer loop |
| `vllm/v1/worker/gpu_ffn_model_runner.py` | FFN-side model runner — `_ffn_forward` loop: recv → compute → send per layer |
| `vllm/v1/worker/gpu_worker.py` | `start_ffn_server_loop()` — spawns `ffn_worker_loop` thread on each FFN worker |
| `vllm/model_executor/models/deepseek_v2.py` | `DeepseekV2DecoderLayer` — conditional module creation, `forward_with_afd`, `compute_ffn_output` |
| `vllm/forward_context.py` | `DPMetadata` — data parallel metadata (token counts per rank) |

---

## Communication Stack Summary

```
┌──────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  send_attn_output / recv_ffn_output / send_dp_metadata_list  │
├──────────────────────────────────────────────────────────────┤
│                     Connector Layer                          │
│  P2PAFDConnector (pair selection, concat/chunk/broadcast)    │
├──────────────────────────────────────────────────────────────┤
│                     Transport Layer                          │
│  NCCL (hidden states)  │  Gloo (metadata, ncclUniqueId)     │
│  torch.ops.vllm.       │  gloo_pg.send/recv directly        │
│  afd_p2p_send/recv     │  (bypasses torch.distributed.*)    │
├──────────────────────────────────────────────────────────────┤
│                     Group Layer                              │
│  Per-pair standalone groups via init_afd_process_group       │
│  Own TCP store per pair (no global _group_count dependency)  │
├──────────────────────────────────────────────────────────────┤
│                     Hardware Layer                           │
│  NVLink / PCIe (NCCL)  │  TCP/IP loopback (Gloo)           │
└──────────────────────────────────────────────────────────────┘
```
