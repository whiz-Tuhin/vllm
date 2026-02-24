# AFD (Attention-FFN Disaggregation) Architecture — Code Flow

This document traces the full execution flow for the 1A1F (1 Attention GPU + 1 FFN GPU) setup running DeepSeek-V2-Lite, as configured in PR #29772.

---

## Overview

AFD splits each transformer layer's computation across two separate GPU processes:
- **Attention Server (GPU 0):** Runs embeddings, attention (MLA), layernorms, and the final LM head
- **FFN Server (GPU 1):** Runs only the MLP/MoE layers

They communicate intermediate activations via NCCL P2P send/recv after every layer.

```
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│     ATTENTION SERVER (GPU 0)    │     │      FFN SERVER (GPU 1)         │
│                                 │     │                                 │
│  vllm serve ... --afd-config    │     │  vllm fserver ... --afd-config  │
│    afd_role = "attention"       │     │    afd_role = "ffn"             │
│                                 │     │                                 │
│  Loads: embeddings, attention   │     │  Loads: MLP/MoE weights,        │
│    weights, layernorms, lm_head │     │    shared experts, layernorms   │
│                                 │     │                                 │
│  For each layer:                │     │  For each layer:                │
│    1. input_layernorm           │     │    1. recv hidden_states        │
│    2. self_attn (MLA)           │     │    2. MLP/MoE forward           │
│    3. post_attention_layernorm  │     │    3. send FFN output back      │
│    4. SEND hidden_states ──────────>  │                                 │
│    5. RECV FFN output    <──────────  │                                 │
│                                 │     │                                 │
│  After all layers:              │     │                                 │
│    final_norm → lm_head → logits│     │                                 │
└─────────────────────────────────┘     └─────────────────────────────────┘
```

---

## 1. Entry Points

### Attention Server: `vllm serve`

```
Command:
  CUDA_VISIBLE_DEVICES=0 vllm serve <model> --enforce_eager --enable-dbo --afd-config '{...}'

Code path:
  vllm/entrypoints/cli/serve.py          # CLI subcommand "serve"
    → vllm/entrypoints/openai/api_server.py  # FastAPI app, creates AsyncLLMEngine
      → AsyncLLMEngine → EngineCore (subprocess)
        → Executor → GPUWorker → GPUModelRunner
```

This is the **standard vLLM serve path** — it creates a full API server with OpenAI-compatible endpoints (`/v1/completions`, `/v1/chat/completions`). The AFD config makes it behave as an attention-only server during forward passes.

### FFN Server: `vllm fserver`

```
Command:
  CUDA_VISIBLE_DEVICES=1 vllm fserver <model> --enforce_eager --afd-config '{...}'

Code path:
  vllm/entrypoints/cli/fserver.py        # CLI subcommand "fserver"
    → FServerCommand.cmd(args)
      → vllm/entrypoints/afd_ffn_server.py   # AFDFFNServer class
        → AFDFFNServer.__init__()             # Creates VllmConfig from args
        → AFDFFNServer.start()
          → Executor → GPUWorker → GPUFFNModelRunner   # FFN-specific model runner
          → AFDFFNServer._run_server_loop()
            → executor.collective_rpc("start_ffn_server_loop")
```

This is a **headless server** — no HTTP API, no scheduler. It just loads the model (FFN weights only) and enters an infinite loop waiting for activations from the attention server.

---

## 2. Configuration

### AFDConfig (`vllm/config/afd.py`)

Parsed from the `--afd-config` JSON string:

```python
@dataclass
class AFDConfig:
    afd_connector: str = "dummy"       # "p2pconnector" for NCCL P2P
    afd_role: str = "attention"        # "attention" or "ffn"
    afd_host: str = "127.0.0.1"       # NCCL rendezvous address
    afd_port: int = 1239              # NCCL rendezvous port
    num_afd_stages: int = 3           # Pipeline stages (2 with DBO)
    afd_extra_config: dict = {}       # e.g. {"afd_size": "1A1F"}
```

The `afd_size` string `"1A1F"` means 1 Attention GPU + 1 FFN GPU.

---

## 3. Model Loading & Weight Splitting

### File: `vllm/model_executor/models/deepseek_v2.py`

#### DeepseekV2DecoderLayer.__init__() (line ~920)

Each decoder layer checks `afd_role` to decide which submodules to create:

```python
self.afd_role = afd_config.afd_role if afd_config else None

# Only create attention on attention server (or no AFD)
if self.afd_role is None or self.afd_role == "attention":
    self.self_attn = DeepseekV2MLA(...)    # MLA attention

# Only create MLP/MoE on FFN server (or no AFD)
if self.afd_role is None or self.afd_role == "ffn":
    if layer_idx >= config.first_k_dense_replace:
        self.mlp = DeepseekV2MoE(...)      # MoE (64 experts)
    else:
        self.mlp = DeepseekV2MLP(...)      # Dense FFN (layer 0 only)
```

#### DeepseekV2ForCausalLM.load_weights() (line ~1525)

Weight loading filters by role:

```python
# Attention server: skip MoE/MLP weights
if self.afd_role == "attention" and self.is_moe_weight(name):
    continue

# FFN server: skip non-MoE, non-common weights (i.e., skip attention weights)
if self.afd_role == "ffn" and not self.is_moe_weight(name) and not self.is_common_weight(name):
    continue
```

Where:
- `is_moe_weight()`: matches "experts", "shared_experts", "gate", "up", "down"
- `is_common_weight()`: matches "lm_head", "model.norm", "embed_tokens", "input_layernorm", "post_attention_layernorm"

**Result for DeepSeek-V2-Lite (hidden_size=2048, 27 layers, first_k_dense_replace=1):**
- Attention server loads: embeddings, all 27 attention layers (MLA), all layernorms, lm_head
- FFN server loads: 1 dense MLP (layer 0) + 26 MoE layers (64 experts each), layernorms, embeddings

---

## 4. NCCL P2P Connector Initialization

### File: `vllm/distributed/afd_transfer/afd_connector/p2p_connector.py`

#### P2PAFDConnector.init_afd_connector() (line ~148)

Called during startup on both sides. For 1A1F:

```
afd_size = "1A1F" → attn_size=1, ffn_size=1

Rank assignment:
  FFN server:  world_rank = 0  (rank + 0, since role="ffn")
  Attn server: world_rank = 1  (rank + ffn_size=1, since role="attention")

Process group creation (NCCL backend):
  afd_pg: world_size=2, all ranks [0, 1]

Sub-groups (both contain ranks [0, 1]):
  a2e_group: attention → FFN communication (send_attn_output / recv_attn_output)
  e2a_group: FFN → attention communication (send_ffn_output / recv_ffn_output)

PyNCCL communicators:
  a2e_pynccl: PyNcclCommunicator on a2e_group
  e2a_pynccl: PyNcclCommunicator on e2a_group

Metadata channel (Gloo backend):
  p2p_pg: for sending DPMetadata (token counts) before each forward pass
```

---

## 5. FFN Server Loop

### File: `vllm/v1/worker/gpu_worker.py` — `start_ffn_server_loop()` (line 722)

After initialization, the FFN server enters an infinite loop:

```python
def ffn_worker_loop():
    while not shutdown_event.is_set():
        # 1. Wait for DPMetadata from attention server (blocking recv via Gloo)
        dp_metadata_list, is_graph_capturing = connector.recv_dp_metadata_list()

        # 2. Execute FFN computation
        model_runner.execute_model(dp_metadata_list=dp_metadata_list)

        torch.cuda.synchronize()
```

### File: `vllm/v1/worker/gpu_ffn_model_runner.py` — `_ffn_forward()` (line 144)

The core FFN execution loop:

```python
def _ffn_forward(self, dp_metadata_list):
    # Update tensor metadata (shapes for P2P buffers)
    self.connector.update_state_from_dp_metadata(dp_metadata_list)

    for layer_idx in range(0, self.num_layers):      # 27 layers
        for ubatch_idx in range(num_ubatches):        # 1 ubatch (no DBO split)
            # Recv attention output from attention server
            hidden_states, metadata = self.connector.recv_attn_output(ubatch_idx)

            # Run FFN/MoE for this layer
            ffn_output = self.model.compute_ffn_output(hidden_states, layer_idx)

            # Send FFN output back to attention server
            self.connector.send_ffn_output(ffn_output, metadata)
```

---

## 6. Attention Server Forward Pass (with AFD)

### File: `vllm/v1/worker/gpu_model_runner.py` — `execute_model()` (line ~4900)

When a request arrives, the attention server's `execute_model` runs:

```python
# 1. Build AFD metadata
afd_metadata = self._build_afd_metadata(ubatch_slices, num_tokens)

# 2. Set forward context (carries afd_metadata through the model)
with set_forward_context(afd_metadata=afd_metadata, ...):

    # 3. Build and send DPMetadata to FFN server (token counts)
    dp_metadata_list = {0: DPMetadata(...)}
    self.afd_connector.send_dp_metadata_list(dp_metadata_list)

    # 4. Run model forward (enters forward_with_afd)
    outputs = self.model(input_ids, positions, ...)
```

### File: `vllm/model_executor/models/deepseek_v2.py` — `DeepseekV2Model.forward()` (line ~1280)

```python
def forward(self, input_ids, positions, ...):
    hidden_states = self.embed_tokens(input_ids)    # Embedding lookup
    residual = None

    afd_metadata = get_forward_context().afd_metadata
    if afd_metadata is not None:
        hidden_states, residual = self.forward_with_afd(
            hidden_states, residual, positions, afd_metadata
        )
    else:
        # Normal forward (no AFD)
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

    hidden_states, _ = self.norm(hidden_states, residual)   # Final RMSNorm
    return hidden_states  # → lm_head → logits
```

### `forward_with_afd()` (line 1143) — The Core AFD Orchestration

```python
def forward_with_afd(self, hidden_states, residual, positions, afd_metadata):
    for layer_idx, layer in enumerate(self.layers):
        afd_connector = afd_metadata.afd_connector

        if layer_idx > 0:
            # Receive FFN output from previous layer
            hidden_states = afd_connector.recv_ffn_output(ref_tensor=hidden_states)

        # Run attention-only forward for this layer
        # (layer.forward() returns early when afd_role=="attention")
        hidden_states, residual = layer(positions, hidden_states, residual)

        # Send post-attention hidden_states to FFN server
        afd_connector.send_attn_output(hidden_states, metadata)

        # DBO yield point (for micro-batch pipelining)
        hidden_states = apply_dbo_yield(hidden_states)

    # Receive final FFN output (from last layer)
    hidden_states = afd_connector.recv_ffn_output(ref_tensor=hidden_states)

    return hidden_states, residual
```

### `DeepseekV2DecoderLayer.forward()` with `afd_role="attention"` (line ~1000)

```python
def forward(self, positions, hidden_states, residual, ...):
    # 1. Input LayerNorm (fused add + RMSNorm)
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # residual = hidden_states + residual (accumulated)
        # hidden_states = RMSNorm(residual)

    # 2. Self-Attention (MLA)
    hidden_states = self.self_attn(positions, hidden_states)

    # 3. Post-Attention LayerNorm (fused add + RMSNorm)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # residual = attn_output + residual
        # hidden_states = RMSNorm(residual)

    # 4. EARLY RETURN — skip MLP (FFN server handles it)
    if self.afd_role == "attention":
        return hidden_states, residual    # ← sent to FFN via P2P

    # (Normal path: would run self.mlp(hidden_states) here)
```

---

## 7. P2P Communication Details

### File: `vllm/distributed/afd_transfer/afd_connector/p2p_connector.py`

#### Sending (Attention → FFN): `send_attn_output()` (line 383)

```python
def send_attn_output(self, hidden_states, metadata):
    dst = (self.a2e_group.rank_in_group - 1) % world_size  # → FFN rank
    self._send_hidden_states(hidden_states, dst, self.a2e_group)

def _send_hidden_states(self, hidden_states, dst, group):
    torch.ops.vllm.afd_p2p_send(hidden_states, dst, self.a2e_comm_id)
    # → PyNcclCommunicator.send(tensor, dst) → ncclSend on current CUDA stream
```

#### Receiving (FFN side): `recv_attn_output()` (line 430)

```python
def recv_attn_output(self, ubatch_idx=0):
    src = (self.a2e_group.rank_in_group - 1) % world_size  # → Attn rank
    hidden_states = self._recv_hidden_states(src, self.a2e_group, tensor_metadata)
    return hidden_states, metadata

def _recv_hidden_states(self, src, group, tensor_metadata, ref_tensor=None):
    # Allocate receive buffer: shape=[num_tokens, hidden_size], dtype=model_dtype
    size = tensor_metadata.size  # e.g., [6, 2048]
    hidden_states = torch.empty(size, dtype=tensor_metadata.dtype, device=...)

    torch.ops.vllm.afd_p2p_recv(hidden_states, src, self.a2e_comm_id)
    # → PyNcclCommunicator.recv(tensor, src) → ncclRecv on current CUDA stream
    return hidden_states
```

#### Metadata exchange: `send_dp_metadata_list()` / `recv_dp_metadata_list()`

Before each forward pass, the attention server sends `DPMetadata` (containing token counts) to the FFN server via Gloo (CPU-based, blocking):

```python
# Attention side (sends):
send_dp_metadata_list(dp_metadata_list)
  → pickle.dumps(data) → torch.distributed.send(tensor, dst, group=p2p_pg)

# FFN side (receives, in ffn_worker_loop):
dp_metadata_list = recv_dp_metadata_list()
  → torch.distributed.recv(tensor, src, group=p2p_pg) → pickle.loads(data)
```

This tells the FFN server how many tokens to expect, so it can allocate correctly-sized P2P receive buffers via `update_state_from_dp_metadata()`.

---

## 8. Per-Request Data Flow (Complete Trace)

For a request: `"The capital of France is"` (6 tokens, 64 max_tokens)

### Prefill Phase (6 input tokens processed at once)

```
ATTENTION SERVER                          FFN SERVER
────────────────                          ──────────
1. Receive HTTP request
2. Scheduler creates batch (6 tokens)
3. execute_model() called
4. Build DPMetadata(num_tokens=6)
5. send_dp_metadata_list ──────────────→  recv_dp_metadata_list
                                          update_state_from_dp_metadata
                                            → tensor shape = [6, 2048]

6. embed_tokens(input_ids) → [6, 2048]
7. forward_with_afd() begins

   Layer 0:
     input_layernorm → self_attn(MLA)
     → post_attention_layernorm
     → hidden_states [6, 2048]
     send_attn_output ──────────────────→  recv_attn_output
                                           compute_ffn_output (dense MLP)
     recv_ffn_output ←──────────────────   send_ffn_output

   Layer 1:
     input_layernorm → self_attn(MLA)
     → post_attention_layernorm
     send_attn_output ──────────────────→  recv_attn_output
                                           compute_ffn_output (MoE, 64 experts)
     recv_ffn_output ←──────────────────   send_ffn_output

   ... (layers 2-26 same as layer 1) ...

   Layer 26 (final):
     send_attn_output ──────────────────→  recv_attn_output
                                           compute_ffn_output (MoE)
     recv_ffn_output ←──────────────────   send_ffn_output

8. final_norm(hidden_states, residual)
9. lm_head → logits → sample token
10. Return first token to client
```

### Decode Phase (1 token at a time, repeated 64 times)

Same flow but with `num_tokens=1` per step. Each step:
- Attention server sends DPMetadata(num_tokens=1)
- Tensor shape becomes [1, 2048]
- 27 layer round-trips (send/recv) per token
- Total: 27 × 2 = 54 NCCL P2P operations per generated token

---

## 9. Key Files Summary

| File | Role |
|------|------|
| `vllm/config/afd.py` | AFDConfig dataclass (role, connector, host/port) |
| `vllm/entrypoints/cli/fserver.py` | `vllm fserver` CLI subcommand |
| `vllm/entrypoints/afd_ffn_server.py` | FFN server main class (AFDFFNServer) |
| `vllm/v1/worker/gpu_worker.py:722` | `start_ffn_server_loop()` — FFN worker infinite loop |
| `vllm/v1/worker/gpu_ffn_model_runner.py` | GPUFFNModelRunner — FFN-side model runner |
| `vllm/v1/worker/gpu_model_runner.py` | GPUModelRunner — Attention-side (extended for AFD) |
| `vllm/model_executor/models/deepseek_v2.py` | DeepSeek V2 model with AFD-aware forward paths |
| `vllm/distributed/afd_transfer/afd_connector/p2p_connector.py` | NCCL P2P connector (send/recv) |
| `vllm/distributed/afd_transfer/afd_connector/base.py` | BaseAFDConnector interface |
| `vllm/distributed/afd_transfer/afd_connector/factory.py` | Connector factory (creates by name) |
| `vllm/distributed/afd_transfer/afd_connector/metadata.py` | AFDConnectorMetadata structures |
| `vllm/forward_context.py` | AFDMetadata, DPMetadata, forward context threading |
| `vllm/v1/worker/ubatching.py` | DBO (Dual-Batch Orchestration) yield/scheduling |

---

## 10. Residual Stream Handling

A critical architectural detail — the **residual tensor never crosses the P2P boundary**:

```
Layer N (Attention Server):
  hidden_states, residual = input_layernorm(ffn_output_N-1, residual_N-1)
    → residual = ffn_output_N-1 + residual_N-1  (accumulated on attention side)
    → hidden_states = RMSNorm(residual)

  attn_output = self_attn(hidden_states)

  hidden_states, residual = post_attention_layernorm(attn_output, residual)
    → residual = attn_output + residual  (accumulated on attention side)
    → hidden_states = RMSNorm(residual)

  SEND hidden_states to FFN  ← only the normalized activation crosses the wire
  RECV ffn_output from FFN   ← FFN output comes back

  (residual stays local, used in next layer's input_layernorm)
```

This means:
- Only `[num_tokens, 2048]` tensors are transferred per layer (not residual)
- The residual stream is accumulated entirely on the attention server
- The FFN server is stateless — it just transforms hidden_states through MLP/MoE
