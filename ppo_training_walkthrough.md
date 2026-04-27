# PPO Training Walkthrough in verl: Tensors, Communication, and Parallelism

This document traces a concrete PPO training step in verl end-to-end.
Every tensor has explicit shape notation, every communication is tied to a specific NCCL primitive,
and every parallelism strategy is explained.

---

## 1. Concrete Configuration

### 1.1 Model — LLaMA-7B

| Symbol | Name | Value |
|--------|------|-------|
| H | hidden size | 4096 |
| A | number of attention heads | 32 |
| d | head dimension (H/A) | 128 |
| F | FFN intermediate size | 11008 |
| L | number of transformer layers | 32 |
| V | vocabulary size | 32000 |

### 1.2 Hardware and Parallelism

```
Total GPUs: 8  (2 nodes × 4 GPUs per node, NVLink intra-node, InfiniBand cross-node)

Training strategy :  TP=2, DP=4, PP=1  (FSDP-based data parallelism)
Rollout strategy  :  TP=2, DP=4, PP=1  (hybrid engine — same physical GPUs)
```

**GPU layout** (DP rank × TP rank):

```
Node 0:
  GPU 0  →  DP=0, TP=0
  GPU 1  →  DP=0, TP=1
  GPU 2  →  DP=1, TP=0
  GPU 3  →  DP=1, TP=1
Node 1:
  GPU 4  →  DP=2, TP=0
  GPU 5  →  DP=2, TP=1
  GPU 6  →  DP=3, TP=0
  GPU 7  →  DP=3, TP=1
```

Each DP rank (a TP-parallel pair) holds one complete copy of the model, sharded along the head / FFN dimension.
During training with FSDP the parameters are further sharded across DP ranks (see §8).

### 1.3 Batch Sizes and Sequence Lengths

| Symbol | Name | Value |
|--------|------|-------|
| B | prompts per step (global) | 16 |
| n | rollouts per prompt | 4 |
| B_seq | total sequences = B × n | 64 |
| P | prompt length (tokens) | 512 |
| R | response length (tokens) | 512 |
| S | full sequence length = P + R | 1024 |
| M | PPO mini-batch size (global) | 16 |
| E | PPO epochs | 1 |

**Why mini-batches?** Rollout is ~100× more expensive than a training step (autoregressive
generation is sequential; training is parallel). Mini-batches amortize that cost: collect 64
sequences once, then extract multiple gradient updates cheaply. With M=16 and E=1 you get 4
updates per rollout instead of 1.

Multiple gradient updates on the same batch are beneficial because a single gradient step
under-exploits the collected experience — the gradient estimated from one pass is noisy and
the policy barely moves. Doing several steps lets the policy climb further up the advantage
landscape of the current batch, extracting more signal per expensive rollout. This is the
same intuition as running multiple SGD steps on a fixed dataset rather than discarding it
after one pass.

**What PPO epoch (E) means**: one full pass over the collected rollout batch. With E=1, each
of the 64 sequences is seen exactly once before the batch is discarded. Increasing E squeezes
more updates from the same rollout but risks destabilizing training: each gradient step moves
π_new further from the π_old that collected the data, making the importance ratios
`π_new/π_old` in the clipped surrogate loss increasingly inaccurate. E=1 is standard for
LLM RL (e.g. GRPO); E=3–4 is used in classic RL environments where per-update drift is
smaller.

Per-DP-rank sizes derived from the above:

| Quantity | Formula | Value |
|----------|---------|-------|
| prompts per DP rank (rollout) | B / DP | 4 |
| sequences per DP rank (after repeat) | B_seq / DP | 16 |
| mini-batch per DP rank | M / DP | 4 |

**Dtype**: bfloat16 (2 bytes) for activations and parameters; float32 for optimizer states.

---

## 2. Parallelism Strategies Explained

### 2.1 Tensor Parallelism (TP=2)

The model weights inside each transformer layer are split across the two GPUs of a DP rank.
verl uses the Megatron-LM column-parallel / row-parallel convention, designed so each
sub-layer needs only **one AllReduce** regardless of how many matrix multiplies it contains.

* **Column-parallel** — split `W` along the output dimension. Each TP rank holds `(H, K//TP)`
  and produces a partial output `(…, K//TP)`. No communication needed.
* **Row-parallel** — split `W` along the input dimension. Each TP rank holds `(K//TP, H)`
  and produces a partial sum `(…, H)`. **AllReduce (sum)** across the TP group combines them.

The key insight is that column-parallel and row-parallel are always paired:
column-parallel produces a partial output that feeds directly into row-parallel,
so only one AllReduce is needed at the end of the pair, not after each multiply.

**Attention** applies this pattern to QKV and O. With TP=2, each rank owns 16 of 32 heads:

```
x : (16, 512, H=4096)   — same on both TP ranks

# Column-parallel: split output (heads) across TP ranks, no comm
W_Q_local : (4096, 2048)    Q_local = x @ W_Q_local  → (16, 512, 2048)
W_K_local : (4096, 2048)    K_local = x @ W_K_local  → (16, 512, 2048)
W_V_local : (4096, 2048)    V_local = x @ W_V_local  → (16, 512, 2048)

# Local attention over 16 heads, no comm
Q_local : (16, 16 heads, 512, 128)
attn_out : (16, 16 heads, 512, 128)  →  concat  →  (16, 512, 2048)

# Row-parallel: split input (local head outputs) across TP ranks
W_O_local : (2048, 4096)    partial = attn_out @ W_O_local  → (16, 512, 4096)
                             AllReduce (sum) across TP group
out : (16, 512, 4096)       — full hidden, same on both ranks
```

W_O is `(2048, 4096)` because its input dimension is the concatenated local heads
`(16 heads × 128 = 2048)`, not the full hidden size.

**FFN** uses SwiGLU (LLaMA's FFN variant). SwiGLU = Swish + Gated Linear Unit.
Standard FFN is `ReLU(x @ W1) @ W2`. SwiGLU splits W1 into two parallel projections —
gate and up — so gating and content are learned independently:

```
SwiGLU(x) = (x @ W_up) * Swish(x @ W_gate)
           = (x @ W_up) * (x @ W_gate * sigmoid(x @ W_gate))
```

This adds a third matrix (W_gate) but each matrix is made narrower (F = 11008 ≈ 2/3 × 4 × H
instead of 4 × H) to keep total parameter count similar to a standard FFN.

With TP=2, the same column→row pattern applies:

```
# Column-parallel gate + up, no comm
W_gate_local : (4096, 5504)   gate = x @ W_gate_local  → (16, 512, 5504)
W_up_local   : (4096, 5504)   up   = x @ W_up_local    → (16, 512, 5504)
hidden = SiLU(gate) * up                               → (16, 512, 5504)

# Row-parallel down
W_down_local : (5504, 4096)   partial = hidden @ W_down_local  → (16, 512, 4096)
                               AllReduce (sum) across TP group
out : (16, 512, 4096)
```

Full weight table with TP=2:

| Weight | Shape (full) | Shape per TP rank | Parallel type |
|--------|-------------|-------------------|---------------|
| W_Q | (4096, 4096) | (4096, 2048) | column |
| W_K | (4096, 4096) | (4096, 2048) | column |
| W_V | (4096, 4096) | (4096, 2048) | column |
| W_O | (4096, 4096) | (2048, 4096) | row → **AllReduce** |
| W_gate | (4096, 11008) | (4096, 5504) | column |
| W_up | (4096, 11008) | (4096, 5504) | column |
| W_down | (11008, 4096) | (5504, 4096) | row → **AllReduce** |
| Embed | (32000, 4096) | (16000, 4096) | vocab-column |

Each transformer layer requires exactly **2 AllReduce calls** per forward pass
(one after W_O, one after W_down), and 2 more during the backward pass —
**4 AllReduce calls per layer per forward-backward pass**.

### 2.2 Data Parallelism (DP=4, FSDP)

The global batch is split evenly across the 4 DP ranks.
With FSDP (ZeRO-3 equivalent), each DP rank holds only `1/DP` of each weight shard,
saving memory by a factor of 4.

* During the **forward pass**, each layer's weights are reconstructed via **AllGather**
  before computation, then immediately discarded.
* During the **backward pass**, gradients are immediately **ReduceScattered**:
  each GPU accumulates only `1/DP` of the gradient tensor.
* The optimizer step runs on the local gradient shard.
* Before the **next forward pass** the updated shard is AllGathered again.

This means the effective memory footprint for parameters per GPU is:
```
7B params × 2 bytes / (TP=2 × DP=4) ≈ 437 MB per GPU
```
plus optimizer state at 4 bytes × 2 moments / (TP × DP) ≈ 875 MB per GPU.

### 2.3 Pipeline Parallelism (PP=1 — disabled in this example)

PP=1 means the entire model lives on one pipeline stage. With PP>1, the model's layers would be
split into stages and connected by point-to-point sends/receives of activations
(`torch.distributed.send` / `recv`), forming a pipeline bubble.

### 2.4 Sequence Parallelism (SP=1 — disabled in this example)

SP (Ulysses-style) distributes the sequence dimension across GPUs.
With SP=s, each GPU processes `S/s` tokens.  AllToAll collectives shuffle the (batch, seq, heads)
dimensions between TP and SP groups.  Disabled here; included for reference.

### 2.5 Hybrid Engine

verl's hybrid engine reuses the same physical GPUs for both rollout (inference) and
training (gradient update).  After each training step the updated parameters are
AllGathered from FSDP shards to reconstruct the full TP-sharded model for rollout.

---

## 3. Model Parameter Memory Layout (per GPU)

Each GPU holds one row of the DP × TP grid. At training time (FSDP active) it owns
`1/(TP × DP) = 1/8` of every weight tensor.  During rollout (FSDP AllGathered) it owns
`1/TP = 1/2` of every weight tensor.

Example: W_down in one layer
* Full weight: (11008, 4096) = 45M params × 2 B = 90 MB
* Rollout shard (TP=2): (5504, 4096) per GPU = 45 MB
* Training shard (TP=2, DP=4): 5504×4096 / 4 rows ≈ 11 MB per GPU

---

## 4. Phase-by-Phase Walkthrough

Each phase lists:
* **Input tensors** (shape, dtype, placement)
* **Computation steps** with intermediate tensor shapes
* **NCCL communications** (primitive, group, message volume)
* **Output tensors** (shape, dtype, placement)

---

### Phase 1 — Data Loading and Scatter

**Where**: Ray controller (CPU process)

The controller samples a mini-batch from the dataset and broadcasts prompt tokens to each DP rank.

| Tensor | Shape | Dtype | Size | Placement |
|--------|-------|-------|------|-----------|
| `prompts` (global) | (B=16, P=512) | int64 | 64 KB | CPU, controller |
| `attention_mask` (global) | (16, 512) | bool | 8 KB | CPU, controller |

The controller sends `prompts[dp_rank * 4 : (dp_rank+1) * 4]` to each DP rank worker group via Ray object store.

After scatter (per DP rank, on TP_rank=0 — TP_rank=1 receives the same):
```
prompts_local :  (B/DP=4, P=512)   int64  ~16 KB
```

Before rollout, the prompts are repeated n=4 times:
```
prompts_repeated :  (4*n=16, P=512)   int64  ~64 KB per DP rank
```

---

### Phase 2 — Rollout (Autoregressive Generation)

**Where**: Each DP rank (pair of TP-parallel GPUs)

The model generates R=512 new tokens token-by-token.
A KV cache is maintained to avoid recomputing attention over the already-generated prefix.

#### 2a. Prefill (process the prompt)

Input:
```
input_ids :  (batch=16, P=512)  int64   on both TP ranks
```

For each of the L=32 transformer layers, one forward pass over the full prompt:

**Step i — Token Embedding** (vocab-parallel):
```
x = embed(input_ids)  →  (16, 512, H=4096)  bfloat16
```
Each TP rank looks up only its 16000 vocab rows, then AllReduces to get the full embedding.

**Step ii — Attention (QKV projection, column-parallel)**:
```
Q_local = x @ W_Q_local  →  (16, 512, H/TP=2048)   no comm, each rank independent
K_local = x @ W_K_local  →  (16, 512, 2048)
V_local = x @ W_V_local  →  (16, 512, 2048)
```

Reshape for multi-head:
```
Q_local :  (16, A/TP=16, P=512, d=128)
K_local :  (16, 16, 512, 128)
V_local :  (16, 16, 512, 128)
```

Populate KV cache (per TP rank, per layer):
```
K_cache :  (16, 16, t=512, 128)   bfloat16   ~64 MB
V_cache :  (16, 16, 512, 128)     bfloat16   ~64 MB
```

Flash attention (local, no comm):
```
attn_out :  (16, 16, 512, 128)  →  (16, 512, 2048)
```

**Step iii — Output projection (row-parallel) + AllReduce**:
```
partial = attn_out @ W_O_local  →  (16, 512, H=4096)   per TP rank
```
**NCCL — AllReduce (SUM)** across TP group (GPU pair within a node):
```
  group:   {GPU 0, GPU 1}  (or whichever DP-rank pair)
  tensor:  (16, 512, 4096)  bfloat16
  volume:  16 × 512 × 4096 × 2 bytes = 64 MB   (each direction in ring-allreduce)
```
After AllReduce, both TP ranks have the same `x` of shape `(16, 512, 4096)`.

**Step iv — FFN (gate+up column-parallel, down row-parallel) + AllReduce**:
```
gate = x @ W_gate  →  (16, 512, F/TP=5504)   per TP rank, no comm
up   = x @ W_up    →  (16, 512, 5504)
hidden = SiLU(gate) * up  →  (16, 512, 5504)   local
partial = hidden @ W_down  →  (16, 512, H=4096)   per TP rank
```
**NCCL — AllReduce (SUM)** across TP group:
```
  group:   same TP pair
  tensor:  (16, 512, 4096)  bfloat16
  volume:  64 MB
```

**Per-layer total**: 2 AllReduce calls × 64 MB = 128 MB communicated.
**Prefill total**: 32 layers × 2 AllReduce = **64 AllReduce calls**, ~4 GB total.

#### 2b. Decode (generate R=512 tokens one at a time)

At decode step t (t = 0 … R-1), only the new token is processed:

```
x :  (16, 1, 4096)   — embedding of the last-generated token, bfloat16
```

Each layer step:
```
Q = x @ W_Q_local  →  (16, 1, 2048)       per TP rank
K = x @ W_K_local  →  (16, 1, 2048)
V = x @ W_V_local  →  (16, 1, 2048)

# Append to KV cache
K_cache :  (16, 16, P+t, 128)  [grows each step]
V_cache :  (16, 16, P+t, 128)

# Attention: query against full prefix
scores = Q_reshaped @ K_cache.T  →  (16, 16, 1, P+t)
attn   = softmax(scores) @ V_cache  →  (16, 16, 1, 128)  →  (16, 1, 2048)

# Row-parallel output projection
partial = attn @ W_O_local  →  (16, 1, 4096)   per TP rank
```
**NCCL — AllReduce (SUM)** across TP group:
```
  tensor:  (16, 1, 4096)  bfloat16
  volume:  16 × 1 × 4096 × 2 bytes = 128 KB   (tiny at decode time)
```

FFN similarly produces another AllReduce of 128 KB.

**Per-decode-step total**: 32 layers × 2 AllReduce × 128 KB ≈ **8 MB per token step**.
**Decode total for R=512**: 512 steps × 8 MB = **4 GB** (same order as prefill).

#### 2c. Log-Prob Collection During Generation

At each decode step, after the final layer the logit for the sampled token is recorded:
```
logit_local :  (16, 1, V/TP=16000)   per TP rank
```
The TP ranks AllGather (or AllReduce) to get:
```
logits :  (16, 1, V=32000)   bfloat16
```
then:
```
log_prob_t :  (16,)   float32   — log probability of the sampled token at step t
```

**NCCL — AllGather** across TP group for logits (or equivalently AllReduce on partial sums):
```
  tensor per rank:  (16, 1, 16000)
  volume:           16 × 32000 × 2 bytes ≈ 1 MB per step
```

#### 2d. Output of Rollout Phase

After generation, each DP rank produces:

| Tensor | Shape | Dtype | Size | Placement |
|--------|-------|-------|------|-----------|
| `sequences` | (B_seq/DP=16, S=1024) | int64 | 128 KB | GPU (TP_rank=0) |
| `attention_mask` | (16, 1024) | bool | 16 KB | GPU |
| `rollout_log_probs` | (16, R=512) | float32 | 32 KB | GPU |
| KV cache (per layer) | (16, A/TP=16, S=1024, d=128) | bfloat16 | ~64 MB × 32 | GPU |

KV cache is freed after generation; it is not needed for training.

The controller collects results from all 4 DP ranks. **No additional NCCL** is needed here —
Ray moves the data over the network via its object store.

Global tensors on the controller (CPU):
```
sequences        :  (B_seq=64, S=1024)   int64
attention_mask   :  (64, 1024)           bool
rollout_log_probs:  (64, 512)            float32
```

---

### Phase 3 — Reward Computation

**Where**: Reward model workers (may be colocated with actor workers or on separate GPUs)

For a rule-based reward (e.g., exact-match or verifiable math), all computation is local CPU/GPU:

```
rewards :  (B_seq=64,)   float32   — one scalar per sequence
```

For a learned reward model, the forward pass is identical in structure to Phase 2 prefill,
producing a scalar value at the `[EOS]` position.

Output (on controller):
```
token_level_rewards :  (64, R=512)   float32
  — zero everywhere except the final response token, which holds the scalar reward
```

---

### Phase 4 — Old Log-Prob Recomputation

**Where**: Actor rollout workers (same physical GPUs as rollout)

Although log probs were recorded during generation, verl recomputes them with a full
non-autoregressive forward pass for numerical stability.

Input per DP rank:
```
input_ids      :  (16, S=1024)   int64
attention_mask :  (16, 1024)     bool
```

Forward pass (micro-batch size = 4 to fit memory):
Process 4 micro-batches of 4 sequences each.

For each layer, the AllReduce tensors are now full-sequence:
```
attn AllReduce :  (micro_batch=4, S=1024, H=4096)   bfloat16   64 MB each
FFN  AllReduce :  (4, 1024, 4096)                    bfloat16   64 MB each
```

After the final layer, collect log probs for response tokens only:
```
log_probs_local :  (4, R=512, V/TP=16000)   per TP rank
```

To obtain per-token log probs without materializing the full vocab dimension,
each TP rank computes its partial log-sum and a local AllReduce is done:
```
old_log_probs :  (4, 512)   float32   per micro-batch
```
Concatenated across micro-batches:
```
old_log_probs :  (16, 512)   float32   per DP rank
```

**Per-layer communication**:
```
NCCL — AllReduce (SUM) across TP group (per layer, ×2):
  tensor:  (4, 1024, 4096)   bfloat16
  volume:  4 × 1024 × 4096 × 2 bytes = 32 MB each
  total for 32 layers: 64 calls × 32 MB = 2 GB
```

Global result gathered on controller:
```
old_log_probs :  (B_seq=64, R=512)   float32   ~128 KB
```

---

### Phase 5 — Reference Policy Log-Probs

**Where**: Reference policy workers (same architecture, frozen weights — or actor with LoRA disabled)

Identical computation to Phase 4, same communication pattern.

Output (on controller):
```
ref_log_probs :  (64, 512)   float32
```

---

### Phase 6 — Value Estimation (Critic)

**Where**: Critic workers (LLaMA-7B with a linear value head)

Input per DP rank:
```
input_ids      :  (16, 1024)   int64
attention_mask :  (16, 1024)   bool
```

Same forward pass as Phase 4, plus a value head:
```
hidden_last_layer :  (16, 1024, H=4096)   bfloat16
values = hidden_last_layer @ W_value  →  (16, 1024, 1)  →  (16, 1024)
```
`W_value : (H=4096, 1)` is a small head, not TP-sharded.

Same AllReduce communication as Phase 4.

Output per DP rank:
```
values :  (16, S=1024)   float32   — value for each token position
```
Global result on controller:
```
values :  (64, 1024)   float32
```

---

### Phase 7 — Advantage Estimation (GAE)

**Where**: Controller (CPU, no GPU, no NCCL)

All input tensors from Phases 3–6 are now on the controller.

```
token_level_rewards :  (64, 512)   float32
old_log_probs       :  (64, 512)   float32
ref_log_probs       :  (64, 512)   float32
values (response)   :  (64, 512)   float32   — slice [:, -512:]
response_mask       :  (64, 512)   bool
```

#### 7a. KL Penalty

```
kl = old_log_probs - ref_log_probs      # (64, 512)  float32
penalized_rewards = token_level_rewards - beta * kl   # (64, 512)
```

`beta` is a scalar adaptive KL coefficient (e.g., 0.01).

#### 7b. GAE Backward Pass (from `core_algos.py:compute_gae_advantage_return`)

```python
# Pseudocode matching verl's implementation
nextvalues = 0
lastgaelam = 0
advantages = []

for t in reversed(range(R=512)):
    delta = penalized_rewards[:, t] + gamma * nextvalues - values[:, t]  # (64,)
    lastgaelam = delta + gamma * lam * lastgaelam * response_mask[:, t]  # (64,)
    nextvalues  = values[:, t] * response_mask[:, t]                     # (64,)
    advantages.append(lastgaelam)

advantages = stack(reversed(advantages))  # (64, 512)  float32
returns     = advantages + values          # (64, 512)  float32
advantages  = whiten(advantages, response_mask)  # zero mean, unit variance
```

Typical values:  `gamma=1.0`, `lam=0.95` (standard PPO).

Output:
```
advantages :  (64, 512)   float32   ~128 KB
returns    :  (64, 512)   float32   ~128 KB
```

**No NCCL in this phase** — purely local scalar operations on the controller.

---

### Phase 8 — Actor Update (PPO)

**Where**: Actor rollout workers (all 8 GPUs, same as rollout)

The controller scatters the full enriched batch to DP ranks.

Input per DP rank (16 sequences):
```
input_ids      :  (16, S=1024)    int64
attention_mask :  (16, 1024)      bool
old_log_probs  :  (16, R=512)     float32
advantages     :  (16, 512)       float32
returns        :  (16, 512)       float32   [not used for actor, only for critic]
response_mask  :  (16, 512)       bool
```

PPO epochs = 1, global mini-batch = 16, so per DP rank mini-batch = 4 sequences.

#### 8a. FSDP AllGather (parameter reconstruction per layer)

Before computing each layer's forward pass, FSDP reconstructs the full TP-sharded parameters
from their DP shards.

For W_Q (one weight, column-parallel):
```
NCCL — AllGather across DP group (4 GPUs, cross-node):
  each rank holds:  (4096, 2048/DP=512)   bfloat16   ~2 MB
  gathered result:  (4096, 2048)           bfloat16   ~8 MB per TP rank
  volume sent per GPU: 2 MB (one shard)
```

All 7 weight matrices per layer are AllGathered before that layer runs, then discarded.
Total AllGather volume per layer ≈ 7 matrices × ~16 MB (full size before TP split) / 2 ≈ 56 MB.

```
NCCL — AllGather (FSDP, per layer, all 7 weight matrices):
  group:   DP group (4 GPUs, cross-node via InfiniBand)
  volume per layer: ~56 MB   (reconstructed then discarded)
  total 32 layers: ~1.8 GB AllGather per forward pass
```

#### 8b. Forward Pass (same as Phase 4, TP AllReduce per layer)

Mini-batch size = 4:
```
AllReduce per layer (TP group):
  attn:  (4, 1024, 4096)  bfloat16  = 32 MB
  FFN:   (4, 1024, 4096)  bfloat16  = 32 MB
32 layers × 2 × 32 MB = 2 GB total for forward pass
```

#### 8c. PPO Loss Computation (local)

```
log_ratio = new_log_probs - old_log_probs   # (4, 512)  float32
ratio     = exp(log_ratio)                  # (4, 512)

surr1     = ratio * advantages              # (4, 512)
surr2     = clamp(ratio, 1-eps, 1+eps) * advantages   # (4, 512)   eps=0.2

# Actor loss per token (masked)
actor_loss = -mean(min(surr1, surr2) * response_mask)  # scalar

# Entropy bonus (optional)
entropy    = -sum(p * log_p, dim=-1)       # (4, 512)  approximated from top-k logits
entropy_loss = -entropy_coeff * mean(entropy * response_mask)

total_loss = actor_loss + entropy_loss     # scalar
```

#### 8d. Backward Pass (TP AllReduce + FSDP ReduceScatter)

The backward pass is the mirror of the forward pass.

**TP AllReduce** (same as forward, for row-parallel weight gradients):
```
NCCL — AllReduce (SUM) per layer, ×2 (TP group, intra-node):
  tensor:  (4, 1024, 4096)   bfloat16
  volume:  32 MB each
  total 32 layers: 2 GB AllReduce
```

**FSDP ReduceScatter** (synchronize gradients across DP ranks, per layer):
After the backward pass through a layer, FSDP immediately ReduceScatters the gradient
so each DP rank holds only its `1/DP` shard of the gradient.

```
NCCL — ReduceScatter (SUM) per layer (DP group, cross-node via InfiniBand):
  input per GPU:   full gradient for that layer's weights, e.g. W_Q grad: (4096, 2048) = 16 MB per TP
  output per GPU:  shard of gradient: (4096, 512) = 4 MB
  volume per GPU sent: 16 MB (for W_Q alone, summed over 7 matrices ≈ 56 MB per layer)
  total 32 layers: 32 × 56 MB ≈ 1.8 GB ReduceScatter
```

#### 8e. Loss AllReduce (aggregate scalar loss across DP ranks)

After all micro-batches complete:
```python
# from engine_workers.py:198
torch.distributed.all_reduce(loss, op=ReduceOp.AVG, group=dp_group)
```
```
NCCL — AllReduce (AVG) across DP group:
  tensor:  scalar float32  (negligible volume)
  group:   DP group (4 GPUs, cross-node)
```

#### 8f. AllGather for Metrics (token counts, per verl engine_workers.py:290)

```python
torch.distributed.all_gather_object(global_token_num_output, global_token_num, dp_group)
```
```
NCCL — AllGather across DP group:
  object:  list of token counts (Python integers, negligible volume)
  purpose: compute MFU (model FLOPs utilization)
```

#### 8g. Optimizer Step (local, no NCCL)

AdamW on each GPU's local gradient shard:
```
params_shard    :  ~437 MB per GPU (bfloat16)
grad_shard      :  ~437 MB per GPU (bfloat16)
m_shard (mean)  :  ~875 MB per GPU (float32)
v_shard (var)   :  ~875 MB per GPU (float32)
```
Total optimizer state per GPU: ~2.2 GB.

---

### Phase 9 — Critic Update

Structure is identical to Phase 8, but using the value regression loss:

```
value_pred   :  (4, S=1024)   float32   — predicted values
value_clipped = clip(value_pred, old_values - eps_v, old_values + eps_v)
value_loss   = 0.5 * mean((value_clipped - returns)^2 * response_mask)
```

All NCCL operations (FSDP AllGather, TP AllReduce, FSDP ReduceScatter, loss AllReduce)
are identical in type and volume to Phase 8.

---

### Phase 10 — Weight Synchronization (Hybrid Engine)

After training, the rollout model (used for next iteration's generation) must be
updated with the new weights.

1. **sleep_replicas()**: rollout workers are paused.
2. **FSDP AllGather** — reconstruct full TP-sharded model from FSDP shards on each DP rank:
   ```
   NCCL — AllGather across DP group:
     input per GPU:  ~875 MB (FSDP shard, bfloat16)
     output per GPU: ~3.5 GB (full TP=2 copy, bfloat16)
     volume: 875 MB × 4 GPUs in DP group = 3.5 GB
   ```
3. Rollout parameters are now up-to-date on each TP-parallel GPU pair.
4. **wake_up()**: rollout workers resume.

---

## 5. Communication Summary Table

| Phase | NCCL Primitive | Group | Tensor | Volume | Direction |
|-------|---------------|-------|--------|--------|-----------|
| Rollout prefill (per layer) | AllReduce ×2 | TP (2 GPUs, intra-node) | (16,512,4096) bf16 | 64 MB × 2 | intra-node |
| Rollout decode (per layer, per step) | AllReduce ×2 | TP | (16,1,4096) bf16 | 128 KB × 2 | intra-node |
| Rollout logits (per step) | AllGather | TP | (16,1,16000) bf16 | 1 MB | intra-node |
| Fwd pass log-prob (per layer) | AllReduce ×2 | TP | (4,1024,4096) bf16 | 32 MB × 2 | intra-node |
| FSDP fwd pass (per layer) | AllGather | DP (4 GPUs, cross-node) | ~56 MB worth of weights | 56 MB | cross-node |
| FSDP bwd pass (per layer) | ReduceScatter | DP | ~56 MB worth of grads | 56 MB | cross-node |
| TP backward (per layer) | AllReduce ×2 | TP | (4,1024,4096) bf16 | 32 MB × 2 | intra-node |
| Loss aggregation | AllReduce (AVG) | DP | scalar | < 1 KB | cross-node |
| Metrics collection | AllGather (object) | DP | token count list | < 1 KB | cross-node |
| Weight sync (post-training) | AllGather | DP | 875 MB shard | 875 MB | cross-node |

**Per-iteration communication budget (rough)**:

| Communication type | Approx. total volume |
|-------------------|---------------------|
| TP AllReduce (rollout + log-prob + training) | ~15 GB |
| FSDP AllGather (fwd) × 2 models (actor + critic) | ~3.5 GB |
| FSDP ReduceScatter (bwd) × 2 models | ~3.5 GB |
| Weight sync (AllGather) | ~3.5 GB |
| **Total** | **~26 GB** |

TP communications are NVLink (within node, ~600 GB/s bidirectional).
DP communications (FSDP, weight sync) are InfiniBand cross-node (~200 GB/s).

---

## 6. Full Tensor Inventory

| Tensor | Shape | Dtype | Bytes | Where |
|--------|-------|-------|-------|-------|
| `prompts` (global) | (16, 512) | int64 | 64 KB | controller CPU |
| `sequences` (global) | (64, 1024) | int64 | 512 KB | controller CPU |
| `attention_mask` (global) | (64, 1024) | bool | 64 KB | controller CPU |
| `rollout_log_probs` (global) | (64, 512) | float32 | 128 KB | controller CPU |
| `ref_log_probs` (global) | (64, 512) | float32 | 128 KB | controller CPU |
| `old_log_probs` (global) | (64, 512) | float32 | 128 KB | controller CPU |
| `rewards` (global) | (64, 512) | float32 | 128 KB | controller CPU |
| `values` (global) | (64, 1024) | float32 | 256 KB | controller CPU |
| `kl` (global) | (64, 512) | float32 | 128 KB | controller CPU |
| `advantages` (global) | (64, 512) | float32 | 128 KB | controller CPU |
| `returns` (global) | (64, 512) | float32 | 128 KB | controller CPU |
| `response_mask` (global) | (64, 512) | bool | 32 KB | controller CPU |
| Model weights (per GPU, rollout) | — | bfloat16 | ~3.5 GB | GPU (TP shard) |
| Model weights (per GPU, training) | — | bfloat16 | ~437 MB | GPU (TP+DP shard) |
| Optimizer state (per GPU) | — | float32 | ~2.2 GB | GPU |
| KV cache (per GPU, during rollout) | (16, 16, ≤1024, 128)×32×2 | bfloat16 | ~4 GB | GPU |
| Activation (per layer fwd) | (4, 1024, 4096) | bfloat16 | ~32 MB | GPU |

---

## 7. Dimension Index

| Dimension symbol | Meaning | Value in example |
|-----------------|---------|-----------------|
| B | global prompt batch size | 16 |
| n | rollouts per prompt | 4 |
| B_seq | total sequences (B×n) | 64 |
| P | prompt length | 512 |
| R | response length | 512 |
| S | full sequence length (P+R) | 1024 |
| H | hidden size | 4096 |
| A | number of attention heads | 32 |
| d | head dimension (H/A) | 128 |
| F | FFN intermediate size | 11008 |
| L | number of layers | 32 |
| V | vocabulary size | 32000 |
| TP | tensor parallel size | 2 |
| DP | data parallel size | 4 |
| PP | pipeline parallel size | 1 |
| M | PPO mini-batch size (global) | 16 |
| E | PPO epochs | 1 |

---

## 8. Key Code Pointers

| Concept | File | Lines |
|---------|------|-------|
| PPO training loop | `verl/trainer/ppo/ray_trainer.py` | 1260–1654 |
| GAE advantage computation | `verl/trainer/ppo/core_algos.py` | 216–263 |
| GRPO advantage computation | `verl/trainer/ppo/core_algos.py` | 268–334 |
| Worker train_mini_batch | `verl/workers/engine_workers.py` | 235–323 |
| DP loss AllReduce | `verl/workers/engine_workers.py` | 195–198 |
| DP token count AllGather | `verl/workers/engine_workers.py` | 286–292 |
| DP metrics AllGather | `verl/workers/engine_workers.py` | 204–207 |
| Rollout device mesh | `verl/workers/engine_workers.py` | 592–601 |
| Padding utilities | `verl/workers/utils/padding.py` | — |
| DataProto protocol | `verl/protocol.py` | — |
