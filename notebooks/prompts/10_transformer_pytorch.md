Create a Jupyter Notebook named:

```text
10_transformer_pytorch.ipynb
```

This notebook is the tenth part of a **PyTorch Interview Preparation** course for someone preparing for **AI Engineer / Machine Learning Engineer / Generative AI Engineer interviews**.

Assume the learner has already completed:

```text
01_tensor_basics.ipynb
02_tensor_operations.ipynb
03_autograd.ipynb
04_neural_networks.ipynb
05_losses_optimizers.ipynb
06_training_loop.ipynb
07_dataset_dataloader.ipynb
08_gpu_and_devices.ipynb
09_debugging_and_performance.ipynb
```

and already understands:

- tensor shapes and broadcasting
- batched matrix multiplication
- `nn.Module`
- `nn.Linear`
- `nn.Embedding`
- `nn.LayerNorm`
- softmax
- Dataset / DataLoader
- training loops
- device handling
- debugging

Do not turn this notebook into a full theoretical Transformer course.

The purpose is to teach the **PyTorch syntax, tensor shapes, and implementation patterns behind Transformers and LLMs** that an AI Engineer should be able to understand and discuss during interviews.

The learner should implement important Transformer building blocks from tensors first, then learn the corresponding high-level PyTorch modules.

---

# Main Learning Goals

By the end of this notebook, the learner should confidently understand and use:

```python
nn.Embedding
nn.Linear
nn.LayerNorm
nn.Dropout
nn.MultiheadAttention

torch.matmul
torch.softmax
torch.triu
torch.ones
torch.arange
```

and understand concepts such as:

- token IDs
- vocabulary size
- embedding dimension
- sequence length
- batch dimension
- token embeddings
- positional information
- query, key, value
- scaled dot-product attention
- attention scores
- attention weights
- causal masking
- padding masks
- self-attention
- cross-attention
- multi-head attention
- residual connections
- LayerNorm
- feed-forward network
- Transformer block
- logits over vocabulary
- language-model output shapes
- next-token prediction
- attention tensor shapes
- transformer parameter counts
- inference vs training behavior
- KV cache at a conceptual level

Mark these as:

```text
🔥 Interview Essential
```

- embedding shapes
- Q / K / V
- scaled dot-product attention
- why divide by `sqrt(d_k)`
- causal mask
- self-attention
- multi-head attention
- number of heads
- head dimension
- residual connections
- LayerNorm
- feed-forward block
- `(batch, sequence, hidden_size)`
- language-model logits `(batch, sequence, vocab_size)`

---

# Notebook Structure

Use approximately:

```text
# PyTorch for Transformers and LLMs

## 1. Setup
## 2. Transformer Tensor Shapes
## 3. Token IDs
## 4. Token Embeddings
## 5. Positional Information
## 6. Query, Key, and Value
## 7. Scaled Dot-Product Attention
## 8. Attention Score Shapes
## 9. Softmax in Attention
## 10. Causal Masks
## 11. Padding Masks
## 12. Self-Attention
## 13. Cross-Attention
## 14. Multi-Head Attention
## 15. Splitting Into Heads
## 16. Combining Heads
## 17. nn.MultiheadAttention
## 18. Residual Connections
## 19. LayerNorm
## 20. Feed-Forward Networks
## 21. Building a Transformer Block
## 22. Stacking Transformer Blocks
## 23. Output Projection
## 24. Language Model Logits
## 25. Next-Token Prediction
## 26. Parameter Counting
## 27. Attention Complexity
## 28. KV Cache Concept
## 29. Common Mistakes
## 30. Debugging Exercises
## 31. Interview Questions
## 32. Knowledge Check Quiz
## 33. Write From Memory
## 34. Final Mini Transformer Challenge
## 35. Cheat Sheet
```

---

# 1. Setup

Start with:

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
```

The notebook must:

- work on CPU
- require no internet access
- require only PyTorch
- use small synthetic tensors
- execute quickly

Do not use Hugging Face libraries in this notebook.

The goal is to understand PyTorch Transformer mechanics directly.

---

# 2. Core Transformer Shape Convention

Make this one of the first major sections.

Mark:

```text
🔥 Interview Essential
```

Use:

```text
B = batch size
T = sequence length
D = hidden / embedding dimension
V = vocabulary size
H = number of attention heads
Dh = head dimension
```

Teach the main shape:

```text
(batch, sequence, hidden_size)
```

For example:

```text
(32, 128, 768)
```

means:

```text
32 sequences
128 tokens per sequence
768 features per token
```

Create a diagram:

```text
batch
  ↓

[
  sequence 1: token token token ...
  sequence 2: token token token ...
  ...
]

each token
→ vector of hidden_size values
```

---

# 3. Shape Flashcards

Ask rapid questions.

Example:

```text
x.shape = (16, 128, 512)
```

Ask:

- batch size?
- sequence length?
- hidden size?

Expected:

```text
16
128
512
```

Create at least 10 quick shape questions.

---

# 4. Token IDs

Explain that text is converted to integer token IDs before entering a Transformer.

Example:

```python
tokens = torch.tensor([
    [12, 45, 983, 5],
    [92, 17, 5, 0]
])
```

Shape:

```text
(batch, sequence)
=
(2, 4)
```

Explain:

> These numbers are indices into a vocabulary. They are not semantic vectors yet.

Input dtype should typically be:

```python
torch.long
```

Show:

```python
print(tokens.dtype)
```

---

# 5. Vocabulary

Explain simply:

```text
Vocabulary size = number of possible token IDs
```

Example:

```text
vocab_size = 50,000
```

Valid IDs roughly:

```text
0 ... 49,999
```

Do not discuss tokenizer algorithms deeply.

---

# 6. Token Embeddings

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
embedding = nn.Embedding(
    num_embeddings=50000,
    embedding_dim=768
)
```

Input:

```python
tokens = torch.randint(
    0,
    50000,
    (32, 128)
)
```

Output:

```python
x = embedding(tokens)

print(x.shape)
```

Expected:

```text
(32, 128, 768)
```

Explain transformation:

```text
token IDs
(batch, sequence)

        ↓ Embedding

token vectors
(batch, sequence, hidden)
```

---

# 7. Embedding Parameter Count

Ask:

```python
nn.Embedding(50000, 768)
```

How many parameters?

Expected:

```text
50,000 × 768
```

Show:

```python
embedding.weight.shape
```

Expected:

```text
(50000, 768)
```

Create several parameter-count exercises.

---

# 8. Positional Information

Explain:

> Self-attention by itself does not inherently know token order, so Transformer inputs need positional information.

Introduce two broad approaches:

```text
learned positional embeddings
fixed positional encodings
```

Do not go deeply into RoPE yet.

Show a simple learned positional embedding.

```python
max_seq_len = 512
hidden_size = 128

position_embedding = nn.Embedding(
    max_seq_len,
    hidden_size
)
```

Positions:

```python
positions = torch.arange(10)

pos = position_embedding(positions)
```

Expected shape:

```text
(10, 128)
```

---

# 9. Add Position Embeddings to Token Embeddings

Example:

```python
B = 4
T = 10
D = 128

tokens = torch.randint(
    0,
    10000,
    (B, T)
)

token_embedding = nn.Embedding(
    10000,
    D
)

position_embedding = nn.Embedding(
    512,
    D
)

x = token_embedding(tokens)

positions = torch.arange(T)

p = position_embedding(positions)

x = x + p
```

Explain broadcasting:

```text
x:
(B, T, D)

p:
(T, D)

result:
(B, T, D)
```

This should connect back to the broadcasting notebook.

---

# 10. Positional Embedding Exercise

Ask learner to create:

```text
batch = 8
sequence = 20
hidden = 64
```

and add learned positional embeddings.

Validate final shape:

```text
(8, 20, 64)
```

---

# 11. Query, Key, Value

Make this a major section.

Mark:

```text
🔥 Interview Essential
```

Explain that each token representation is projected into:

```text
Query  Q
Key    K
Value  V
```

Using linear layers:

```python
q_proj = nn.Linear(D, D)
k_proj = nn.Linear(D, D)
v_proj = nn.Linear(D, D)
```

Then:

```python
Q = q_proj(x)
K = k_proj(x)
V = v_proj(x)
```

Shapes remain:

```text
(B, T, D)
```

---

# 12. Intuition for Q, K, V

Keep this simple.

Explain:

```text
Query
→ What information is this token looking for?

Key
→ What information does this token advertise?

Value
→ What information should be passed along?
```

Do not oversimplify into misleading claims.

State this is an intuition rather than a literal semantic interpretation.

---

# 13. Attention Without Heads First

Before multi-head attention, implement single-head scaled dot-product attention.

Use:

```python
B = 2
T = 4
D = 8

x = torch.randn(B, T, D)
```

Generate:

```python
Q = q_proj(x)
K = k_proj(x)
V = v_proj(x)
```

---

# 14. Attention Scores

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
scores = Q @ K.transpose(-2, -1)
```

Shape reasoning:

```text
Q:
(B, T, D)

K.transpose(-2, -1):
(B, D, T)

scores:
(B, T, T)
```

Explain:

> Every token gets a score against every token.

Visualize:

```text
        keys
       T tokens
     ┌─────────┐
Q T  │ T × T   │
     └─────────┘
```

Create at least 10 score-shape exercises.

---

# 15. Why `transpose(-2, -1)`?

Explain:

```python
K.transpose(-2, -1)
```

changes:

```text
(B, T, D)
```

to:

```text
(B, D, T)
```

so matrix multiplication works:

```text
(T, D) @ (D, T)
→
(T, T)
```

Create a write-from-memory exercise.

---

# 16. Scaled Dot-Product Attention

Teach:

```python
scores = (
    Q @ K.transpose(-2, -1)
) / math.sqrt(D)
```

Mark:

```text
🔥 Very Common Interview Question
```

Explain:

> The dot products tend to grow as the key/query dimension grows. Dividing by `sqrt(d_k)` keeps the score magnitude more controlled before softmax.

Do not overdo statistical derivation.

---

# 17. Softmax Over Attention Scores

Teach:

```python
weights = torch.softmax(
    scores,
    dim=-1
)
```

Shape remains:

```text
(B, T, T)
```

Explain `dim=-1`:

> For each query token, normalize scores across all key positions.

Verify:

```python
weights.sum(dim=-1)
```

should be approximately all ones.

---

# 18. Attention Output

Teach:

```python
output = weights @ V
```

Shape:

```text
weights:
(B, T, T)

V:
(B, T, D)

output:
(B, T, D)
```

Show full attention pipeline:

```text
Q × Kᵀ
   ↓
scores
   ↓ scale
softmax
   ↓
attention weights
   ↓ × V
output
```

---

# 19. Implement `scaled_dot_product_attention()`

Ask learner to implement:

```python
def scaled_dot_product_attention(
    Q,
    K,
    V,
    mask=None
):
    ...
```

Requirements:

1. Compute QKᵀ.
2. Scale by square root of key dimension.
3. Apply optional mask.
4. Softmax.
5. Multiply by V.
6. Return output and weights.

Use `assert` validation.

---

# 20. Causal Attention

Mark:

```text
🔥 Interview Essential for LLMs
```

Explain:

> In autoregressive language modeling, token position `t` must not attend to future token positions.

Example:

```text
Token 1 can see: 1
Token 2 can see: 1,2
Token 3 can see: 1,2,3
Token 4 can see: 1,2,3,4
```

Create matrix:

```text
✓ × × ×
✓ ✓ × ×
✓ ✓ ✓ ×
✓ ✓ ✓ ✓
```

---

# 21. Create a Causal Mask

Teach:

```python
mask = torch.triu(
    torch.ones(T, T),
    diagonal=1
).bool()
```

For `T=4`, show result conceptually:

```text
False True  True  True
False False True  True
False False False True
False False False False
```

Explain `True` positions represent future locations to block, depending on convention.

---

# 22. Apply the Causal Mask

Teach:

```python
scores = scores.masked_fill(
    mask,
    float("-inf")
)
```

Then:

```python
weights = torch.softmax(
    scores,
    dim=-1
)
```

Explain:

```text
softmax(-inf)
→ 0
```

so future tokens receive zero attention probability.

---

# 23. Causal Mask Exercise

Ask learner to build a mask for sequence length:

```text
6
```

Validate:

- shape `(6, 6)`
- upper triangle above diagonal is masked
- diagonal remains available

---

# 24. Padding Masks

Explain:

> Padding masks prevent attention from using padded positions.

Example tokens:

```python
tokens = torch.tensor([
    [5, 7, 9, 0, 0],
    [8, 3, 4, 2, 0]
])
```

Assume:

```text
0 = padding token
```

Create:

```python
padding_mask = tokens == 0
```

Shape:

```text
(B, T)
```

Explain causal mask and padding mask solve different problems.

---

# 25. Causal vs Padding Mask

Create comparison:

| Mask | Purpose |
|---|---|
| causal mask | hide future tokens |
| padding mask | hide padding tokens |

Mark:

```text
🔥 Interview Essential
```

---

# 26. Self-Attention

Explain:

> Self-attention means Q, K, and V are all derived from the same sequence representation.

```python
Q = q_proj(x)
K = k_proj(x)
V = v_proj(x)
```

This is common in Transformer encoder and decoder blocks.

---

# 27. Cross-Attention

Explain conceptually:

> In cross-attention, queries come from one sequence while keys and values come from another.

Example:

```text
decoder hidden states
→ Q

encoder hidden states
→ K and V
```

Shape example:

```text
Q:
(B, T_query, D)

K/V:
(B, T_context, D)

scores:
(B, T_query, T_context)
```

Create 3–5 shape questions.

---

# 28. Multi-Head Attention

Make this one of the largest sections.

Mark:

```text
🔥 Interview Essential
```

Explain:

> Multi-head attention splits the hidden dimension into multiple heads so different attention subspaces can be computed in parallel.

Example:

```text
hidden_size = 768
num_heads = 12

head_dim = 64
```

because:

```text
768 / 12 = 64
```

---

# 29. Important Divisibility Rule

Teach:

```text
hidden_size % num_heads == 0
```

Create exercises:

```text
D = 512
H = 8

head_dim = ?
```

Expected:

```text
64
```

Another:

```text
D = 768
H = 12
```

Expected:

```text
64
```

Another:

```text
D = 1000
H = 12
```

Ask whether it divides evenly.

---

# 30. Splitting Into Heads

Start with:

```text
Q shape:
(B, T, D)
```

For:

```text
H heads
Dh = D / H
```

reshape:

```python
Q = Q.view(
    B,
    T,
    H,
    Dh
)
```

Then transpose:

```python
Q = Q.transpose(1, 2)
```

Result:

```text
(B, H, T, Dh)
```

Explain this shape carefully.

---

# 31. Why Head Dimension Shape Changes

Show:

```text
Before:
(B, T, D)

D = H × Dh

After split:
(B, T, H, Dh)

After transpose:
(B, H, T, Dh)
```

Create at least 10 reshape exercises.

---

# 32. Multi-Head Attention Scores

Use:

```python
scores = (
    Q @ K.transpose(-2, -1)
) / math.sqrt(Dh)
```

Shapes:

```text
Q:
(B, H, T, Dh)

Kᵀ:
(B, H, Dh, T)

scores:
(B, H, T, T)
```

Mark:

```text
🔥 Interview Essential Shape
```

---

# 33. Multi-Head Attention Weights

Softmax:

```python
weights = torch.softmax(
    scores,
    dim=-1
)
```

Shape:

```text
(B, H, T, T)
```

Explain each head gets its own attention matrix.

---

# 34. Multiply by V

```python
head_output = weights @ V
```

Shapes:

```text
weights:
(B, H, T, T)

V:
(B, H, T, Dh)

output:
(B, H, T, Dh)
```

---

# 35. Combine Heads

Teach:

```python
output = output.transpose(1, 2)
```

Shape:

```text
(B, T, H, Dh)
```

Then:

```python
output = output.reshape(
    B,
    T,
    D
)
```

Shape returns to:

```text
(B, T, D)
```

Then output projection:

```python
out_proj = nn.Linear(D, D)

output = out_proj(output)
```

---

# 36. Build Multi-Head Attention Manually

Ask learner to implement a simplified class:

```python
class SimpleMultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads
    ):
        ...
```

Requirements:

- Q projection
- K projection
- V projection
- output projection
- split heads
- scaled dot product
- optional causal mask
- combine heads

Use small sizes such as:

```text
hidden_size = 32
num_heads = 4
```

Validate output shape:

```text
(B, T, 32)
```

---

# 37. `nn.MultiheadAttention`

After manual implementation, teach the PyTorch module.

```python
attention = nn.MultiheadAttention(
    embed_dim=128,
    num_heads=8,
    batch_first=True
)
```

Mark:

```text
🔥 Important PyTorch Syntax
```

Explain `batch_first=True` means input shape:

```text
(B, T, D)
```

rather than sequence-first format.

---

# 38. Use `nn.MultiheadAttention`

Example:

```python
x = torch.randn(
    4,
    10,
    128
)

output, weights = attention(
    x,
    x,
    x
)
```

Shapes:

```text
output:
(4, 10, 128)
```

Discuss attention weight shape carefully according to default averaging behavior of the installed/current PyTorch API.

If necessary, use:

```python
average_attn_weights=False
```

to inspect per-head weights.

Ensure all syntax matches current PyTorch APIs.

---

# 39. Q, K, V Arguments to MultiheadAttention

Explain:

```python
attention(
    query,
    key,
    value
)
```

For self-attention:

```python
attention(x, x, x)
```

For cross-attention:

```python
attention(
    decoder_states,
    encoder_states,
    encoder_states
)
```

Create exercises identifying which is self-attention vs cross-attention.

---

# 40. Attention Masks in `nn.MultiheadAttention`

Introduce at an interview-relevant level:

```python
attn_mask
key_padding_mask
```

Explain conceptually:

```text
attn_mask
→ e.g. causal / structural attention constraints

key_padding_mask
→ padding positions per batch
```

Do not make API edge cases the main focus.

Use documented/current PyTorch behavior.

---

# 41. Residual Connections

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
x = x + attention_output
```

Explain:

> Residual connections add the sublayer output back to its input.

Visualize:

```text
x ─────────────────┐
│                  │
↓                  │
Attention          │
│                  │
└────── + ◄────────┘
        ↓
      output
```

Explain benefits conceptually:

- easier gradient flow
- preserves information
- supports deep networks

---

# 42. LayerNorm

Review in Transformer context:

```python
norm = nn.LayerNorm(D)
```

Input:

```text
(B, T, D)
```

Output remains:

```text
(B, T, D)
```

Explain LayerNorm normalizes across the hidden/features dimension for each token.

Mark:

```text
🔥 Interview Essential
```

---

# 43. Pre-Norm vs Post-Norm

Introduce briefly.

### Pre-Norm

```text
x
↓ LayerNorm
Sublayer
↓
+ residual
```

Conceptual code:

```python
x = x + attention(
    norm1(x)
)
```

### Post-Norm

```text
x
↓ Sublayer
↓ + residual
↓ LayerNorm
```

Conceptual:

```python
x = norm1(
    x + attention(x)
)
```

Explain modern Transformer implementations often use variants of pre-norm, but architectures differ.

Do not make this section overly deep.

---

# 44. Feed-Forward Network

Mark:

```text
🔥 Interview Essential
```

Teach common structure:

```python
ffn = nn.Sequential(
    nn.Linear(D, 4 * D),
    nn.GELU(),
    nn.Linear(4 * D, D)
)
```

Explain:

```text
D
↓
larger hidden dimension
↓
activation
↓
D
```

For example:

```text
768
→ 3072
→ 768
```

---

# 45. Why Feed-Forward Is Applied Per Token

Explain input:

```text
(B, T, D)
```

`nn.Linear` operates on the final dimension, so:

```python
nn.Linear(D, 4 * D)
```

produces:

```text
(B, T, 4D)
```

without manually flattening batch and sequence dimensions.

This is an important PyTorch shape insight.

---

# 46. FFN Shape Exercise

Input:

```text
(32, 128, 768)
```

Layer:

```python
nn.Linear(768, 3072)
```

Expected:

```text
(32, 128, 3072)
```

Then:

```python
nn.Linear(3072, 768)
```

returns:

```text
(32, 128, 768)
```

Create several exercises.

---

# 47. Transformer Block

Build a simplified pre-norm block:

```python
class TransformerBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            hidden_size
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(
            hidden_size
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                hidden_size,
                4 * hidden_size
            ),
            nn.GELU(),
            nn.Linear(
                4 * hidden_size,
                hidden_size
            ),
            nn.Dropout(dropout)
        )
```

Then forward:

```python
def forward(
    self,
    x,
    attn_mask=None,
    key_padding_mask=None
):
    ...
```

Use residual connections.

---

# 48. Transformer Block Shape Invariance

Emphasize:

```text
Input:
(B, T, D)

Transformer block

Output:
(B, T, D)
```

This makes stacking blocks easy.

Create a strong interview note:

> Most Transformer blocks preserve the main hidden-state shape.

---

# 49. Stack Transformer Blocks

Teach:

```python
layers = nn.ModuleList([
    TransformerBlock(
        hidden_size=128,
        num_heads=8
    )
    for _ in range(4)
])
```

Then:

```python
for layer in layers:
    x = layer(x)
```

Explain why `nn.ModuleList` should be used instead of a plain Python list when the modules need to be registered.

Mark:

```text
🔥 Interview Useful
```

---

# 50. `ModuleList` vs Python List

Explain:

```python
nn.ModuleList([...])
```

registers child modules correctly.

A normal Python list of layers may not register them as trainable submodules in the expected way.

Create a short debugging exercise.

---

# 51. Output Projection for Language Models

Mark:

```text
🔥 Interview Essential for LLMs
```

After Transformer blocks:

```text
hidden states:
(B, T, D)
```

Create:

```python
lm_head = nn.Linear(
    D,
    vocab_size
)
```

Then:

```python
logits = lm_head(x)
```

Shape:

```text
(B, T, V)
```

Example:

```text
(8, 128, 50000)
```

Explain:

> For every token position, the model produces one score for every vocabulary token.

---

# 52. Language Model Logits

Use:

```text
batch = 8
sequence = 128
vocab = 50,000
```

Output:

```text
(8, 128, 50,000)
```

Ask:

> What does `logits[0, 10]` represent?

Expected:

> Scores over the vocabulary for position 10 of sequence 0.

---

# 53. Next-Token Prediction

Explain conceptual training:

```text
Input:
"I like deep"

Target:
"like deep learning"
```

At each position, model predicts the next token.

Do not require a real tokenizer.

Use synthetic token IDs.

---

# 54. Shifted Inputs and Targets

Example:

```python
tokens = torch.tensor([
    [10, 20, 30, 40, 50]
])
```

Input:

```python
input_ids = tokens[:, :-1]
```

Expected:

```text
[10, 20, 30, 40]
```

Target:

```python
targets = tokens[:, 1:]
```

Expected:

```text
[20, 30, 40, 50]
```

Mark:

```text
🔥 Interview Essential
```

---

# 55. Language Model Loss Shape

Suppose:

```text
logits:
(B, T, V)

targets:
(B, T)
```

For `CrossEntropyLoss`, reshape:

```python
loss = loss_fn(
    logits.reshape(-1, V),
    targets.reshape(-1)
)
```

Explain:

```text
(B × T, V)

vs

(B × T)
```

This should be a major shape exercise.

---

# 56. Why Flatten Batch and Sequence for Loss?

Explain:

> Each token position acts like one multiclass classification example over the vocabulary.

So:

```text
B sequences × T positions
=
B*T classification examples
```

Create 5–8 exercises.

---

# 57. Full Tiny Language Model Skeleton

Build:

```text
Token IDs
↓
Token Embedding
+
Position Embedding
↓
Transformer Blocks
↓
LayerNorm
↓
Linear vocab projection
↓
Logits
```

Create a class:

```python
class TinyLanguageModel(nn.Module):
    ...
```

Use very small settings:

```text
vocab_size = 100
hidden_size = 32
num_heads = 4
num_layers = 2
max_seq_len = 32
```

It must run quickly on CPU.

---

# 58. Tiny Language Model Forward Shape

Input:

```python
tokens = torch.randint(
    0,
    100,
    (4, 16)
)
```

Expected output:

```text
(4, 16, 100)
```

Validate with `assert`.

---

# 59. Parameter Counting

Teach rough parameter sources:

```text
Token embeddings
Position embeddings
Q/K/V projections
Attention output projection
FFN layers
LayerNorm
LM head
```

Have learner count selected pieces manually.

---

# 60. QKV Projection Parameter Count

For:

```text
D = 768
```

Q projection:

```text
Linear(768, 768)
```

has:

```text
768 × 768 + 768
```

Same for K and V.

Ask for combined QKV parameters.

Then output projection adds another:

```text
768 × 768 + 768
```

Do not require mental multiplication of huge numbers without allowing Python calculations.

---

# 61. Feed-Forward Parameter Count

For:

```text
D = 768
FFN hidden = 3072
```

Layers:

```text
768 → 3072
3072 → 768
```

Ask learner to calculate total weights and biases.

Connect to the fact that FFNs hold a large portion of Transformer parameters.

---

# 62. Weight Tying

Introduce conceptually:

> Some language models reuse the token embedding matrix for the output projection.

Do not require implementation.

Explain this can reduce parameter count.

Mark optional/interview awareness.

---

# 63. Attention Complexity

Mark:

```text
🔥 Interview Essential for LLM roles
```

Explain attention scores shape:

```text
(T, T)
```

for every sequence/head.

Therefore standard self-attention has quadratic dependence on sequence length for the attention matrix:

```text
O(T²)
```

Conceptual comparison:

```text
T = 1,000
→ ~1 million pairwise positions

T = 2,000
→ ~4 million
```

Do not claim this is the entire Transformer runtime complexity; make clear it describes the sequence-length dependence of the attention score matrix.

---

# 64. Sequence Length and Memory

Explain:

```text
Longer sequence
→ larger attention matrices
→ more memory
```

This connects to GPU OOM from Notebook 08/09.

Create interview scenarios.

---

# 65. KV Cache

Introduce conceptually.

Mark:

```text
🔥 Interview Useful for LLM Inference
```

Explain:

> During autoregressive generation, previously computed keys and values can be cached so the model does not recompute them from scratch for every generated token.

Without cache:

```text
token 1
recompute previous states

token 2
recompute previous states

...
```

With KV cache:

```text
previous K/V
→ reuse
```

Explain trade-off:

```text
faster decoding
but
additional memory use
```

Do not implement a full production KV cache.

---

# 66. Prefill vs Decode

Briefly introduce:

```text
Prefill
→ process prompt tokens

Decode
→ generate tokens one at a time
```

Explain KV caching is particularly important during decode.

This is useful for modern AI Engineer interviews.

---

# 67. Training vs Generation

Explain:

### Training

Typically process many tokens in parallel:

```text
(B, T)
```

and predict next tokens for all positions.

### Autoregressive inference

Generate one new token at a time.

Keep concise.

---

# 68. Greedy Next-Token Selection

Show:

```python
next_token_logits = logits[:, -1, :]

next_token = next_token_logits.argmax(
    dim=-1
)
```

Shapes:

```text
logits:
(B, T, V)

last position:
(B, V)

argmax:
(B,)
```

Create shape exercises.

---

# 69. Sampling Preview

Briefly mention:

```python
probs = torch.softmax(
    next_token_logits,
    dim=-1
)

next_token = torch.multinomial(
    probs,
    num_samples=1
)
```

Explain greedy vs sampling conceptually.

Do not turn this into a decoding-strategy notebook.

---

# 70. Temperature

Introduce lightly:

```python
scaled_logits = (
    logits / temperature
)
```

Concept:

```text
lower temperature
→ sharper distribution

higher temperature
→ flatter distribution
```

Do not make it central.

---

# 71. Common Mistakes

Include at least these.

## Mistake 1 — Wrong hidden-state shape assumption

Confusing:

```text
(B, T, D)
```

with:

```text
(T, B, D)
```

especially when `batch_first` settings differ.

---

## Mistake 2 — Hidden size not divisible by heads

```text
D = 100
H = 8
```

cannot split evenly.

---

## Mistake 3 — Scaling attention by `sqrt(D)` instead of `sqrt(head_dim)` after head splitting

Explain correct dimension is per-head key dimension.

---

## Mistake 4 — Softmax on wrong dimension

Wrong:

```python
softmax(scores, dim=1)
```

when intending to normalize each query over key positions.

Usually:

```python
dim=-1
```

---

## Mistake 5 — Forgetting `K.transpose(-2, -1)`

---

## Mistake 6 — Wrong causal-mask direction

Future tokens remain visible.

---

## Mistake 7 — Padding mask semantics reversed

---

## Mistake 8 — Forgetting residual connection

---

## Mistake 9 — Wrong LayerNorm dimension

Should generally match hidden size.

---

## Mistake 10 — Flattening sequence dimension before Transformer layers

Loses token-position structure.

---

## Mistake 11 — Applying softmax before `CrossEntropyLoss`

Again emphasize raw language-model logits.

---

## Mistake 12 — Wrong LM target shift

Input and target accidentally identical rather than shifted.

---

## Mistake 13 — Wrong reshape for token-level loss

---

## Mistake 14 — Using plain Python list instead of `ModuleList` for trainable blocks

---

## Mistake 15 — Forgetting causal masking in autoregressive decoder-style self-attention

---

# 72. Debugging Exercises

Create at least 20 realistic exercises.

Example:

```python
Q = torch.randn(4, 8, 16)
K = torch.randn(4, 8, 16)

scores = Q @ K
```

Ask why this fails.

Expected:

```text
K needs its last two dimensions transposed.
```

---

Another:

```python
scores = Q @ K.transpose(-2, -1)

weights = torch.softmax(
    scores,
    dim=1
)
```

Ask which dimension is likely wrong for single-head attention shaped `(B,T,T)`.

Expected:

```text
dim=-1
```

---

Another:

```text
hidden_size = 768
num_heads = 10
```

Ask what is wrong.

Expected:

> 768 is not divisible by 10.

---

Another:

```python
logits.shape = (8, 128, 50000)
targets.shape = (8, 128)
```

Then:

```python
loss_fn(logits, targets)
```

Ask why reshaping may be needed for the usual CrossEntropyLoss usage.

---

# 73. Attention Shape Exercises

Create approximately 25 shape questions.

Example:

```text
Q:
(32, 12, 128, 64)

K:
(32, 12, 128, 64)

Q @ K.transpose(-2,-1)
```

Expected:

```text
(32, 12, 128, 128)
```

Another:

```text
weights:
(8, 8, 100, 100)

V:
(8, 8, 100, 64)
```

Output:

```text
(8, 8, 100, 64)
```

Another:

After combining 8 heads each of 64 dimensions:

```text
hidden size = 512
```

---

# 74. Interview Questions

Create approximately 40 questions.

Include:

1. What is the standard hidden-state shape in a batch-first Transformer?
2. What does `nn.Embedding` do?
3. Why do Transformers need positional information?
4. What are Q, K, and V?
5. How are attention scores computed?
6. What shape does QKᵀ have?
7. Why divide by `sqrt(d_k)`?
8. Why is softmax applied over the key dimension?
9. What does the attention matrix represent?
10. What is self-attention?
11. What is cross-attention?
12. What is causal attention?
13. Why do decoder language models need a causal mask?
14. What is a padding mask?
15. Causal mask vs padding mask?
16. What is multi-head attention?
17. Why use multiple attention heads?
18. What is head dimension?
19. What requirement exists between hidden size and number of heads?
20. What is the Q shape after splitting into heads?
21. How do you combine heads?
22. What does `nn.MultiheadAttention` do?
23. Why might `batch_first=True` be convenient?
24. What is a residual connection?
25. Why are residual connections important?
26. Why is LayerNorm common in Transformers?
27. BatchNorm vs LayerNorm for Transformers?
28. What is the Transformer feed-forward network?
29. Why does an FFN preserve sequence length?
30. What shape does a Transformer block usually preserve?
31. What does a language model output?
32. What shape are language-model logits?
33. Why is the vocabulary dimension last?
34. How is next-token prediction trained?
35. Why are inputs and targets shifted?
36. How do you compute CrossEntropyLoss for `(B,T,V)` logits?
37. Why does self-attention scale quadratically with sequence length?
38. What is a KV cache?
39. Why does KV caching speed autoregressive inference?
40. What is the memory trade-off of KV caching?
41. What is prefill vs decode?
42. What is greedy decoding?
43. What does temperature do?

For each provide:

### Short Interview Answer

and:

### Detailed Explanation

Use collapsible sections.

---

# 75. Knowledge Check Quiz

Create approximately 30 multiple-choice questions.

Example:

```text
Input hidden states:

(16, 128, 768)

Number of heads = 12

What is head_dim?

A. 12
B. 64
C. 128
D. 768
```

Correct:

```text
B
```

Another:

```text
Q shape:
(B, H, T, Dh)

K shape:
(B, H, T, Dh)

What is the attention score shape?

A. (B,H,Dh,Dh)
B. (B,T,T)
C. (B,H,T,T)
D. (B,H,T,Dh)
```

Correct:

```text
C
```

Another:

```text
Why use a causal mask?

A. Hide padding only
B. Prevent attending to future tokens
C. Normalize embeddings
D. Reduce vocabulary size
```

Correct:

```text
B
```

Put answers separately.

---

# 76. Write From Memory

Create approximately 20 prompts.

Examples:

> Create a token embedding layer for vocab size 30,000 and hidden size 512.

> Generate position indices for sequence length `T`.

> Create Q, K, V projections.

> Compute attention scores from Q and K.

> Scale scores correctly.

> Apply softmax to attention scores.

> Multiply attention weights by V.

> Create a causal upper-triangular mask.

> Mask future scores with `-inf`.

> Reshape `(B,T,D)` into `(B,H,T,Dh)`.

> Combine heads back to `(B,T,D)`.

> Create `nn.MultiheadAttention` with batch-first inputs.

> Create LayerNorm for hidden size 768.

> Create the Transformer FFN `768 → 3072 → 768`.

> Create an output projection from hidden states to vocabulary logits.

> Shift token sequence into LM inputs and targets.

> Reshape LM logits and targets for CrossEntropyLoss.

Each should include executable validation where reasonable.

---

# 77. Transformer Syntax Flashcards

Create rapid active recall.

Prompt:

```text
Token IDs → embeddings?
```

Expected:

```python
nn.Embedding(vocab_size, hidden_size)
```

Prompt:

```text
Attention scores?
```

Expected:

```python
Q @ K.transpose(-2, -1)
```

Prompt:

```text
Scale?
```

Expected:

```python
/ math.sqrt(head_dim)
```

Prompt:

```text
Attention probabilities?
```

Expected:

```python
torch.softmax(scores, dim=-1)
```

Prompt:

```text
Block future positions?
```

Expected:

```python
scores.masked_fill(mask, float("-inf"))
```

Create approximately 20 flashcards.

---

# 78. Final Mini Transformer Challenge

Create a complete tiny decoder-style Transformer language model using only PyTorch.

Use:

```text
vocab_size = 128
max_seq_len = 32
hidden_size = 64
num_heads = 4
num_layers = 2
```

Keep it small enough for CPU.

---

# 79. Final Challenge — Input Data

Create synthetic token sequences:

```python
tokens = torch.randint(
    1,
    vocab_size,
    (256, 20)
)
```

Create shifted:

```python
input_ids = tokens[:, :-1]
targets = tokens[:, 1:]
```

Shapes:

```text
input_ids:
(256, 19)

targets:
(256, 19)
```

---

# 80. Final Challenge — Model

Require:

1. Token embedding.
2. Positional embedding.
3. Two Transformer blocks.
4. Final LayerNorm.
5. Vocabulary projection.

Input:

```text
(B, T)
```

Output:

```text
(B, T, vocab_size)
```

---

# 81. Final Challenge — Transformer Block

Require:

```text
LayerNorm
↓
causal self-attention
↓
residual
↓
LayerNorm
↓
FFN
↓
residual
```

Accept either:

- custom manual multi-head attention
- `nn.MultiheadAttention`

Prefer `nn.MultiheadAttention` for the final model after manual attention has already been taught.

---

# 82. Final Challenge — Causal Mask

Generate based on current `T`.

Ensure no hardcoded fixed sequence length is needed for forward execution.

Validate mask shape:

```text
(T, T)
```

---

# 83. Final Challenge — Loss

Use:

```python
nn.CrossEntropyLoss()
```

Given:

```text
logits:
(B,T,V)

targets:
(B,T)
```

reshape to:

```text
(B*T,V)

(B*T)
```

Compute loss.

---

# 84. Final Challenge — Training Step

Ask learner to implement:

```python
optimizer.zero_grad()

logits = model(input_ids)

loss = ...

loss.backward()

optimizer.step()
```

Use:

```python
AdamW
```

with small LR.

Only train a small number of steps to demonstrate the pipeline.

The goal is understanding architecture and tensor flow, not building a useful language model.

---

# 85. Final Challenge — Verify Causality

Create a conceptual or executable test:

Change a future token while keeping earlier tokens identical.

Verify that an earlier position's output does not change because of future-token changes, within numerical tolerance, assuming model is in eval mode and dropout disabled.

This is an excellent advanced validation of the causal mask.

Mark:

```text
🔴 Hard / Optional
```

---

# 86. Final Challenge — Generation

Add a minimal greedy generation loop.

Start with:

```python
context = torch.tensor(
    [[1, 2, 3]]
)
```

Repeatedly:

1. Run model.
2. Take final-position logits.
3. `argmax`.
4. Append token.
5. Stop after a few generated tokens.

Do not claim outputs are meaningful because the model uses synthetic/random training data.

The goal is to understand autoregressive tensor flow.

---

# 87. Final Cheat Sheet

End with a compact Transformer PyTorch cheat sheet.

Include:

```python
# Embeddings
token_embedding = nn.Embedding(
    vocab_size,
    hidden_size
)

x = token_embedding(input_ids)

# Positions
positions = torch.arange(
    input_ids.size(1),
    device=input_ids.device
)

x = x + position_embedding(
    positions
)

# Q K V
Q = q_proj(x)
K = k_proj(x)
V = v_proj(x)

# Attention
scores = (
    Q @ K.transpose(-2, -1)
) / math.sqrt(head_dim)

weights = torch.softmax(
    scores,
    dim=-1
)

output = weights @ V
```

Causal mask:

```python
mask = torch.triu(
    torch.ones(
        T,
        T,
        device=x.device,
        dtype=torch.bool
    ),
    diagonal=1
)

scores = scores.masked_fill(
    mask,
    float("-inf")
)
```

Multi-head shapes:

```text
(B,T,D)
↓ split
(B,H,T,Dh)
↓ attention
(B,H,T,Dh)
↓ combine
(B,T,D)
```

PyTorch module:

```python
attention = nn.MultiheadAttention(
    embed_dim=hidden_size,
    num_heads=num_heads,
    batch_first=True
)

output, weights = attention(
    x,
    x,
    x
)
```

FFN:

```python
ffn = nn.Sequential(
    nn.Linear(
        hidden_size,
        4 * hidden_size
    ),
    nn.GELU(),
    nn.Linear(
        4 * hidden_size,
        hidden_size
    )
)
```

LM head:

```python
lm_head = nn.Linear(
    hidden_size,
    vocab_size
)

logits = lm_head(x)
```

LM loss:

```python
loss = loss_fn(
    logits.reshape(
        -1,
        vocab_size
    ),
    targets.reshape(-1)
)
```

---

# 88. Critical Mental Model

Include:

```text
Token IDs
(B,T)
   ↓
Embedding
(B,T,D)
   ↓
Q K V
(B,T,D)
   ↓
Split heads
(B,H,T,Dh)
   ↓
QKᵀ
(B,H,T,T)
   ↓
Softmax
(B,H,T,T)
   ↓
× V
(B,H,T,Dh)
   ↓
Combine heads
(B,T,D)
   ↓
Residual + LayerNorm + FFN
(B,T,D)
   ↓
LM Head
(B,T,V)
```

Mark:

```text
🔥 Memorize These Shapes
```

---

# 89. Attention Shape Cheat Sheet

Include:

```text
Input:
(B,T,D)

Q/K/V:
(B,T,D)

Split:
(B,H,T,Dh)

Scores:
(B,H,T,T)

Weights:
(B,H,T,T)

Per-head output:
(B,H,T,Dh)

Combined output:
(B,T,D)

LM logits:
(B,T,V)
```

---

# 90. Interview Quick Reference

Add:

```text
Embedding
→ token IDs to vectors

Q/K/V
→ projected attention representations

QKᵀ
→ pairwise attention scores

sqrt(d_k)
→ stabilizes score scale

softmax
→ convert scores to attention weights

causal mask
→ hide future tokens

multi-head attention
→ multiple attention subspaces

residual connection
→ input + sublayer output

LayerNorm
→ normalize hidden features

FFN
→ per-token nonlinear transformation

LM head
→ hidden states to vocabulary logits

KV cache
→ reuse previous keys/values during decoding
```

---

# 91. Completion Checklist

End with:

```text
## Before Moving to 11_pytorch_interview_challenge.ipynb
```

Add:

- [ ] I understand `(batch, sequence, hidden_size)`.
- [ ] I can use `nn.Embedding`.
- [ ] I can add positional embeddings.
- [ ] I understand Q, K, and V.
- [ ] I can compute QKᵀ.
- [ ] I understand attention score shapes.
- [ ] I know why attention is scaled by `sqrt(d_k)`.
- [ ] I know which dimension softmax uses.
- [ ] I can create a causal mask.
- [ ] I understand padding masks.
- [ ] I can explain self-attention.
- [ ] I can explain cross-attention.
- [ ] I understand multi-head attention.
- [ ] I can calculate head dimension.
- [ ] I can split and combine attention heads.
- [ ] I can use `nn.MultiheadAttention`.
- [ ] I understand residual connections.
- [ ] I understand LayerNorm in Transformers.
- [ ] I can build a Transformer FFN.
- [ ] I can build a simple Transformer block.
- [ ] I understand language-model logits `(B,T,V)`.
- [ ] I can shift LM inputs and targets.
- [ ] I can compute token-level CrossEntropyLoss.
- [ ] I understand quadratic attention scaling with sequence length.
- [ ] I understand KV cache conceptually.
- [ ] I can build a tiny autoregressive Transformer in PyTorch.

---

# 92. Difficulty Labels

Use:

```text
🟢 Easy
🟡 Medium
🔴 Hard
```

Easy:

- token embeddings
- shape interpretation
- positional embeddings
- output logits

Medium:

- Q/K/V
- attention score calculation
- causal masks
- `nn.MultiheadAttention`
- residual + LayerNorm + FFN

Hard / optional:

- manual multi-head attention
- causal-behavior verification
- tiny language-model generation
- parameter-count reasoning

Do not deeply cover:

- RoPE derivation
- FlashAttention implementation
- grouped-query attention
- mixture of experts
- tensor parallelism
- FSDP
- distributed training
- quantization
- LoRA
- RLHF
- speculative decoding

These can belong in later advanced notebooks.

---

# 93. Quality Requirements

The notebook must:

- be a valid `.ipynb`
- execute successfully from top to bottom
- work entirely on CPU
- require no internet access
- require only PyTorch
- contain no empty placeholder sections
- include runnable tensor examples
- include automatic validation
- contain approximately 80–100 exercises/questions
- heavily emphasize tensor shapes
- heavily emphasize Q/K/V and attention
- heavily emphasize causal masking
- include both manual attention and `nn.MultiheadAttention`
- include a small runnable Transformer block
- include a tiny language-model example
- avoid unnecessary mathematical derivations
- remain focused on AI Engineer interview preparation

The notebook should take approximately **3–4 hours** to study thoroughly.

The most important learning loop is:

```text
See tensor shape
      ↓
predict next shape
      ↓
write PyTorch operation
      ↓
run it
      ↓
inspect result
      ↓
build attention manually
      ↓
use PyTorch module
      ↓
explain it verbally
```

By the end of this notebook, the learner should be able to look at code such as:

```python
scores = (
    Q @ K.transpose(-2, -1)
) / math.sqrt(head_dim)

weights = torch.softmax(
    scores,
    dim=-1
)

output = weights @ V
```

and immediately explain:

- what each line does
- why the transpose is needed
- what every tensor shape is
- why scaling is used
- what softmax dimension means
- how this becomes multi-head attention

Finally save the notebook as:

```text
10_transformer_pytorch.ipynb
```