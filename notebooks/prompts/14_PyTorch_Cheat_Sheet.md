Create a Jupyter Notebook named:

```text
PyTorch_Cheat_Sheet.ipynb
```

This notebook should be a **compact but comprehensive PyTorch cheat sheet for AI Engineer / Machine Learning Engineer interviews**.

It is not meant to be a long tutorial.

Its purpose is:

```text
quick revision
+
syntax recall
+
shape reference
+
interview preparation
+
debugging reference
```

Assume the learner has already studied PyTorch fundamentals and wants one notebook they can open before interviews to quickly review the most important syntax and concepts.

The notebook should cover the most useful material from:

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
10_transformer_pytorch.ipynb
13_model_saving_loading_and_inference.ipynb
```

Do not turn this into another multi-hour teaching notebook.

Target study/review time:

```text
30–60 minutes
```

---

# Main Design Goal

The notebook should answer questions such as:

```text
How do I create this tensor?

What shape does this operation produce?

What loss should I use?

How do I write a training loop?

How do I create a Dataset?

How do I move a model to GPU?

How do I debug a shape/device/gradient problem?

How does PyTorch attention syntax work?

How do I save/load a model?
```

Everything should be optimized for **rapid recall**.

Use:

```text
🔥 Must Know
⭐ Useful
🟡 Interview Useful
```

to indicate priority.

Keep explanations short.

Prefer:

```text
syntax
→ one-line meaning
→ tiny example
→ important shape/result
```

---

# Notebook Structure

Use approximately:

```text
# PyTorch Interview Cheat Sheet

## 1. Imports and Setup
## 2. Tensor Creation
## 3. Tensor Properties
## 4. Dtypes
## 5. Devices
## 6. Indexing and Slicing
## 7. Reshaping and Dimensions
## 8. Combining Tensors
## 9. Reductions
## 10. Broadcasting
## 11. Masking and Conditional Operations
## 12. Sorting and Top-K
## 13. Matrix Multiplication
## 14. Autograd
## 15. nn.Module
## 16. Common Layers
## 17. Activation Functions
## 18. Normalization and Dropout
## 19. Embeddings
## 20. CNN Layers
## 21. Loss Functions
## 22. Optimizers
## 23. Standard Training Loop
## 24. Validation / Inference Loop
## 25. Dataset and DataLoader
## 26. GPU and Device Handling
## 27. Mixed Precision
## 28. Debugging
## 29. Model Saving and Loading
## 30. Transformer / Attention Syntax
## 31. Common Shape Reference
## 32. Task → Output → Loss Reference
## 33. Common Mistakes
## 34. Interview Quick Answers
## 35. Final One-Page Memory Section
```

---

# 1. Imports and Setup

Start with:

```python
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    TensorDataset,
    DataLoader,
    random_split,
)

torch.manual_seed(42)
```

Device setup:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device)
```

Mark:

```text
🔥 Must Know
```

---

# 2. Tensor Creation

Create a compact table containing:

| Goal | Syntax |
|---|---|
| From values | `torch.tensor([1, 2, 3])` |
| Zeros | `torch.zeros(2, 3)` |
| Ones | `torch.ones(2, 3)` |
| Constant | `torch.full((2, 3), 5)` |
| Range | `torch.arange(0, 10, 2)` |
| Even spacing | `torch.linspace(0, 1, 5)` |
| Uniform random | `torch.rand(2, 3)` |
| Normal random | `torch.randn(2, 3)` |
| Random integers | `torch.randint(0, 10, (2, 3))` |
| Identity | `torch.eye(4)` |
| Like another tensor | `torch.zeros_like(x)` |
| Ones like | `torch.ones_like(x)` |

Then include one executable example block.

---

# 3. Tensor Properties

Include:

```python
x.shape
x.size()
x.ndim
x.numel()
x.dtype
x.device
x.requires_grad
```

Explain in one line each.

Example:

```python
x = torch.randn(32, 3, 224, 224)

print(x.shape)
print(x.ndim)
print(x.numel())
```

---

# 4. Dtypes

Reference table:

```text
torch.float32
torch.float64
torch.float16
torch.bfloat16

torch.int64
torch.long

torch.int32

torch.bool
```

Conversions:

```python
x.float()
x.long()
x.int()
x.bool()

x.to(torch.float32)
```

Critical note:

```text
CrossEntropyLoss class-index targets
→ usually torch.long
```

and:

```text
BCEWithLogitsLoss targets
→ usually floating point
```

Mark these:

```text
🔥 Must Know
```

---

# 5. Devices

Include:

```python
x = x.to(device)
model = model.to(device)
```

Check:

```python
x.device
next(model.parameters()).device
```

Move back:

```python
x = x.cpu()
```

Safe NumPy conversion:

```python
array = x.detach().cpu().numpy()
```

Important note:

```text
model device
=
input device
=
target device
```

for normal training operations.

---

# 6. Indexing and Slicing

Use:

```python
x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90],
])
```

Show:

```python
x[0]
x[:, 0]
x[1, 2]
x[1:]
x[:, 1:]
x[-1]
```

Boolean:

```python
x[x > 50]
```

Multiple conditions:

```python
x[(x > 20) & (x < 80)]
```

Important:

```text
Use:
&
|
~

not Python:
and
or
```

---

# 7. Reshaping and Dimensions

This should be one of the most important cheat-sheet sections.

Include:

```python
x.reshape(...)
x.view(...)
x.flatten(...)
x.squeeze(...)
x.unsqueeze(...)
x.transpose(...)
x.permute(...)
```

Examples:

```python
x = torch.randn(32, 3, 28, 28)

x.reshape(32, -1)
x.flatten(1)
```

Expected:

```text
(32, 2352)
```

---

Unsqueeze:

```python
x = torch.randn(3, 4)

x.unsqueeze(0)   # (1,3,4)
x.unsqueeze(1)   # (3,1,4)
```

---

Squeeze:

```python
x = torch.randn(1, 3, 1, 4)

x.squeeze()      # (3,4)
```

Warn:

> `squeeze()` without a dimension removes every dimension of size 1.

---

Transpose:

```python
x.transpose(1, 2)
```

Permute:

```python
x = torch.randn(32, 3, 224, 224)

x = x.permute(
    0,
    2,
    3,
    1
)
```

NCHW → NHWC:

```text
(32,3,224,224)
→
(32,224,224,3)
```

---

# 8. `reshape()` vs `view()`

Concise reference:

```text
view()
→ requires compatible memory layout

reshape()
→ more flexible; may return a view or copy
```

Interview answer:

> Use `reshape` when I only care about the resulting shape; use `view` when I specifically want view-like semantics and the tensor layout supports it.

---

# 9. Combining Tensors

Include:

```python
torch.cat()
torch.stack()
```

Example:

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)
```

Then:

```text
cat([a,b], dim=0)
→ (6,4)

cat([a,b], dim=1)
→ (3,8)

stack([a,b], dim=0)
→ (2,3,4)
```

Mental model:

```text
cat
→ existing dimension

stack
→ new dimension
```

Mark:

```text
🔥 Must Know
```

---

# 10. Splitting

Include:

```python
torch.chunk(x, chunks=4, dim=0)

torch.split(x, split_size_or_sections=...)
```

Concise:

```text
chunk
→ specify number of chunks

split
→ specify chunk size or exact sizes
```

---

# 11. Reductions

Include:

```python
x.sum()
x.mean()
x.max()
x.min()
x.argmax()
x.argmin()

torch.any(...)
torch.all(...)
```

By dimension:

```python
x.sum(dim=0)
x.mean(dim=1)

x.argmax(dim=1)
```

Mark:

```text
🔥 dim is Interview Essential
```

Mental model:

```text
Reduction over dim=k
→ dimension k disappears

unless:
keepdim=True
```

Example:

```text
x.shape = (2,3)

x.sum(dim=0)
→ (3,)

x.sum(dim=1)
→ (2,)

x.sum(dim=1, keepdim=True)
→ (2,1)
```

---

# 12. Broadcasting

Include the core rule:

```text
Compare dimensions from the RIGHT.

Two dimensions are compatible when:

same size
OR
one is 1
```

Examples:

```text
(32,10)
+
(10,)
→ (32,10)
```

```text
(5,1)
+
(1,7)
→ (5,7)
```

Invalid:

```text
(3,4)
+
(3,)
```

Mark:

```text
🔥 Interview Essential
```

Include feature normalization:

```python
mean = x.mean(
    dim=0,
    keepdim=True
)

std = x.std(
    dim=0,
    keepdim=True
)

x = (
    x - mean
) / (std + 1e-8)
```

---

# 13. Masking and Conditional Operations

Include:

```python
mask = x > 0

positive = x[mask]
```

`where`:

```python
torch.where(
    condition,
    value_if_true,
    value_if_false
)
```

Example:

```python
x = torch.where(
    x < 0,
    torch.tensor(0.0),
    x
)
```

Better simple clipping:

```python
x = torch.clamp(
    x,
    min=0
)
```

---

# 14. Sorting and Top-K

Include:

```python
values, indices = torch.sort(
    x,
    dim=-1
)

indices = torch.argsort(
    x,
    dim=-1
)

top_values, top_indices = torch.topk(
    x,
    k=5,
    dim=-1
)
```

Mental model:

```text
sort
→ values + indices

argsort
→ indices only

topk
→ top K values + indices
```

---

# 15. Matrix Multiplication

Include:

```python
a @ b

torch.matmul(a, b)

torch.mm(a, b)

torch.bmm(a, b)
```

Reference:

```text
*
→ element-wise multiplication

@
→ matrix multiplication
```

Example:

```text
(32,128)
@
(128,64)

→
(32,64)
```

Batch:

```text
A:
(16,20,64)

B:
(16,64,32)

torch.bmm(A,B)

→
(16,20,32)
```

---

# 16. Advanced Index Selection

Class score selection:

```python
scores = torch.randn(
    32,
    10
)

labels = torch.randint(
    0,
    10,
    (32,)
)

selected = scores[
    torch.arange(
        scores.size(0)
    ),
    labels
]
```

Shape:

```text
(32,)
```

Also include:

```python
torch.gather(...)
```

with one concise example.

---

# 17. Autograd

Make this compact but high priority.

```python
x = torch.tensor(
    2.0,
    requires_grad=True
)

y = x ** 2

y.backward()

print(x.grad)
```

Expected:

```text
4
```

Core APIs:

```python
requires_grad=True
loss.backward()
x.grad
x.grad.zero_()

x.detach()

with torch.no_grad():
    ...

x.requires_grad_(True)
```

Mental model:

```text
forward
→ build computation

backward
→ calculate gradients
```

---

# 18. Gradient Accumulation

Include:

```text
PyTorch gradients ACCUMULATE.
```

Therefore:

```python
optimizer.zero_grad()

loss.backward()

optimizer.step()
```

Standard order:

```text
zero_grad
↓
forward
↓
loss
↓
backward
↓
step
```

Mark:

```text
🔥 Memorize
```

---

# 19. `detach()` vs `no_grad()`

Concise:

```text
detach()
→ disconnect one tensor from graph

no_grad()
→ disable gradient tracking for operations in a block
```

Example:

```python
y = x.detach()
```

and:

```python
with torch.no_grad():
    y = model(x)
```

---

# 20. `nn.Module`

Include minimal model template:

```python
class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(
            10,
            3
        )

    def forward(self, x):
        return self.fc(x)
```

Then:

```python
model = Model()
```

Critical notes:

```text
__init__
→ define layers

forward
→ define computation

super().__init__()
→ initialize nn.Module machinery
```

Use:

```python
model(x)
```

not normally:

```python
model.forward(x)
```

---

# 21. Common Layers

Reference table:

| Layer | Purpose |
|---|---|
| `nn.Linear(a,b)` | fully connected |
| `nn.ReLU()` | common hidden activation |
| `nn.GELU()` | common Transformer activation |
| `nn.Dropout(p)` | regularization |
| `nn.BatchNorm1d(n)` | batch normalization |
| `nn.LayerNorm(n)` | feature/layer normalization |
| `nn.Flatten()` | flatten feature dimensions |
| `nn.Embedding(v,d)` | IDs → vectors |
| `nn.Conv2d(...)` | image convolution |
| `nn.MaxPool2d(...)` | spatial downsampling |

---

# 22. `nn.Linear`

Include:

```python
layer = nn.Linear(
    128,
    64
)
```

Input:

```text
(B,128)
```

Output:

```text
(B,64)
```

Weight:

```text
(64,128)
```

Bias:

```text
(64,)
```

Parameter count:

```text
128 × 64 + 64
```

---

# 23. Activations

Reference:

```python
nn.ReLU()
nn.LeakyReLU()
nn.GELU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=1)
```

Important usage:

```text
ReLU
→ common MLP/CNN hidden layers

GELU
→ Transformers

Sigmoid
→ binary probability conversion

Softmax
→ multiclass probabilities
```

Critical:

```text
Do NOT manually softmax before CrossEntropyLoss.

Do NOT manually sigmoid before BCEWithLogitsLoss.
```

Mark both:

```text
🔥 Common Interview Mistake
```

---

# 24. Logits

Define:

```text
logits
=
raw model scores before sigmoid/softmax
```

Multiclass:

```text
(B,C)
```

Binary:

```text
(B,1)
```

Language model:

```text
(B,T,V)
```

---

# 25. Dropout and Normalization

Dropout:

```python
nn.Dropout(0.5)
```

```text
model.train()
→ active

model.eval()
→ disabled
```

BatchNorm:

```text
training
→ batch statistics + update running stats

eval
→ stored running statistics
```

LayerNorm:

```text
common in Transformers
independent of batch statistics
```

Quick comparison:

```text
BatchNorm
→ batch-dependent
→ common CNN

LayerNorm
→ per-sample/token feature normalization
→ common Transformer
```

---

# 26. Embeddings

Include:

```python
embedding = nn.Embedding(
    vocab_size,
    hidden_size
)
```

Input:

```text
(B,T)
```

dtype:

```text
torch.long
```

Output:

```text
(B,T,D)
```

Example:

```text
(8,128)
→ Embedding(...,768)
→ (8,128,768)
```

---

# 27. CNN Quick Reference

Input convention:

```text
NCHW

(batch, channels, height, width)
```

Conv:

```python
nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    padding=1
)
```

For stride 1 / padding 1 / kernel 3:

```text
spatial size unchanged
```

Pool:

```python
nn.MaxPool2d(2)
```

Example:

```text
(B,32,64,64)
→
(B,32,32,32)
```

Conv parameter count:

```text
out_channels
×
in_channels
×
kernel_height
×
kernel_width
+
out_channels
```

---

# 28. Loss Functions

Create a prominent table:

| Task | Model Output | Target | Loss |
|---|---|---|---|
| Regression | continuous | float | `MSELoss` / `L1Loss` |
| Binary | `(B,1)` logits | float | `BCEWithLogitsLoss` |
| Multiclass | `(B,C)` logits | long `(B,)` | `CrossEntropyLoss` |
| Multilabel | `(B,C)` logits | float `(B,C)` | `BCEWithLogitsLoss` |
| LM | `(B,T,V)` logits | long `(B,T)` | `CrossEntropyLoss` |

Mark:

```text
🔥 Memorize
```

---

# 29. MSE / L1

Include:

```python
nn.MSELoss()
nn.L1Loss()
```

Mental model:

```text
MSE
→ squares errors
→ more sensitive to large errors

L1
→ absolute errors
```

---

# 30. BCEWithLogitsLoss

Include:

```python
loss_fn = nn.BCEWithLogitsLoss()

logits = model(X)

loss = loss_fn(
    logits,
    targets
)
```

Inference:

```python
probs = torch.sigmoid(
    logits
)

pred = (
    probs >= 0.5
).long()
```

---

# 31. CrossEntropyLoss

Include:

```python
loss_fn = nn.CrossEntropyLoss()

logits = model(X)

loss = loss_fn(
    logits,
    targets
)
```

Expected:

```text
logits:
(B,C)

targets:
(B,) long
```

Prediction:

```python
pred = logits.argmax(
    dim=1
)
```

Accuracy:

```python
accuracy = (
    pred == targets
).float().mean()
```

---

# 32. Optimizers

Reference:

```python
optim.SGD(
    model.parameters(),
    lr=0.01
)
```

```python
optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

```python
optim.Adam(
    model.parameters(),
    lr=1e-3
)
```

```python
optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2
)
```

Quick comparison:

```text
SGD
→ simple

SGD + momentum
→ smoother momentum-based updates

Adam
→ adaptive

AdamW
→ adaptive + decoupled weight decay
→ common modern AI default
```

---

# 33. Training Loop

This should be one of the largest visible reference blocks.

```python
model.train()

for X, y in train_loader:

    X = X.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    logits = model(X)

    loss = loss_fn(
        logits,
        y
    )

    loss.backward()

    optimizer.step()
```

Mark:

```text
🔥 Must Be Able To Write From Memory
```

---

# 34. Training Metrics

Multiclass:

```python
pred = logits.argmax(
    dim=1
)

correct = (
    pred == y
).sum().item()

batch_size = y.size(0)

total_correct += correct
total_samples += batch_size
```

Correct weighted loss:

```python
total_loss += (
    loss.item()
    * batch_size
)
```

Then:

```python
avg_loss = (
    total_loss
    / total_samples
)

accuracy = (
    total_correct
    / total_samples
)
```

---

# 35. Validation Loop

Include:

```python
model.eval()

with torch.inference_mode():

    for X, y in val_loader:

        X = X.to(device)
        y = y.to(device)

        logits = model(X)

        loss = loss_fn(
            logits,
            y
        )
```

Explicitly show:

```text
NO backward()
NO optimizer.step()
```

---

# 36. `train()` vs `eval()` vs `inference_mode()`

Reference:

```text
model.train()
→ training behavior

model.eval()
→ evaluation behavior for Dropout/BatchNorm

torch.inference_mode()
→ disable Autograd overhead
```

Critical:

```text
model.eval()
≠
torch.inference_mode()
```

Usually use both for inference.

---

# 37. Dataset

Include:

```python
class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y[idx]
        )
```

Mental model:

```text
Dataset
→ one sample

DataLoader
→ batches
```

---

# 38. TensorDataset

Include:

```python
dataset = TensorDataset(
    X,
    y
)
```

Use when data already exists as aligned tensors.

---

# 39. DataLoader

Include:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

Validation:

```python
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)
```

Useful options:

```python
num_workers=4
pin_memory=True
drop_last=True
collate_fn=...
```

Quick meanings:

```text
shuffle
→ randomize sample order

num_workers
→ parallel loading

pin_memory
→ potentially faster CPU→CUDA transfer

drop_last
→ discard partial last batch

collate_fn
→ custom batching
```

---

# 40. Variable-Length Text

Include concise padding example:

```python
from torch.nn.utils.rnn import (
    pad_sequence
)

padded = pad_sequence(
    sequences,
    batch_first=True,
    padding_value=0
)

attention_mask = (
    padded != 0
)
```

Shapes:

```text
tokens:
(B,T)

mask:
(B,T)
```

---

# 41. Device-Aware DataLoader Loop

Include:

```python
for X, y in loader:

    X = X.to(
        device,
        non_blocking=(
            device.type == "cuda"
        )
    )

    y = y.to(
        device,
        non_blocking=(
            device.type == "cuda"
        )
    )
```

Note:

```text
pin_memory
+
non_blocking
can improve CUDA transfer behavior
```

but not automatically.

---

# 42. GPU Memory Reference

Include:

```python
if torch.cuda.is_available():

    torch.cuda.memory_allocated()

    torch.cuda.memory_reserved()

    torch.cuda.max_memory_allocated()
```

Memory contributors:

```text
parameters
gradients
optimizer state
activations
input batch
temporary buffers
```

OOM checklist:

```text
reduce batch size
mixed precision
gradient accumulation
reduce sequence/image size
check graph retention
avoid storing GPU tensors unnecessarily
```

---

# 43. Mixed Precision

Include concept and syntax.

Use current PyTorch API.

Example:

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16
):
    logits = model(X)

    loss = loss_fn(
        logits,
        y
    )
```

Then GradScaler pattern using a modern, non-deprecated PyTorch API supported by the current environment.

Do not include obsolete AMP syntax.

Quick reference:

```text
FP16
→ smaller range

BF16
→ larger exponent range
→ very common modern AI format
```

---

# 44. Gradient Clipping

Include:

```python
loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

optimizer.step()
```

Order:

```text
backward
↓
clip
↓
step
```

---

# 45. Gradient Accumulation

Include compact pattern:

```python
accum_steps = 4

optimizer.zero_grad()

for step, (X, y) in enumerate(
    train_loader
):

    logits = model(X)

    loss = (
        loss_fn(logits, y)
        / accum_steps
    )

    loss.backward()

    if (
        (step + 1)
        % accum_steps
        == 0
    ):
        optimizer.step()
        optimizer.zero_grad()
```

Mental model:

```text
physical batch × accumulation steps
≈ effective batch size
```

---

# 46. Debugging Quick Checklist

Make this prominent.

```text
Model not working?

1. shape
2. dtype
3. device
4. target format
5. loss choice
6. finite values
7. gradients
8. parameter updates
9. train/eval mode
10. learning rate
```

Mark:

```text
🔥 Memorize
```

---

# 47. Debug Tensor State

Include:

```python
print(x.shape)
print(x.dtype)
print(x.device)
print(x.requires_grad)
```

Finite:

```python
torch.isnan(x).any()
torch.isinf(x).any()
torch.isfinite(x).all()
```

---

# 48. Debug Gradients

Include:

```python
for name, param in (
    model.named_parameters()
):

    if param.grad is None:
        print(
            name,
            "NO GRAD"
        )

    else:
        print(
            name,
            param.grad.norm().item()
        )
```

Explain:

```text
grad=None
→ no gradient path / frozen / detached / unused

huge gradient
→ possible instability

near-zero gradient
→ possible weak gradient flow
```

No universal thresholds.

---

# 49. Anomaly Detection

Include:

```python
torch.autograd.set_detect_anomaly(
    True
)
```

Note:

```text
Useful for debugging backward errors / non-finite gradients.

Slow.
Do not leave enabled for normal training.
```

---

# 50. Common Memory Leak Pattern

Bad:

```python
loss_history.append(
    loss
)
```

Better:

```python
loss_history.append(
    loss.item()
)
```

Outputs:

```python
saved_outputs.append(
    output.detach().cpu()
)
```

when gradients are not needed.

---

# 51. Save Model

Include:

```python
torch.save(
    model.state_dict(),
    "model.pt"
)
```

Load:

```python
model = Model(...)

state = torch.load(
    "model.pt",
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    state
)

model.to(device)
model.eval()
```

Mark:

```text
🔥 Interview Essential
```

---

# 52. Save Checkpoint

Include:

```python
checkpoint = {
    "epoch": epoch,

    "model_state_dict":
        model.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "best_val_loss":
        best_val_loss,
}

torch.save(
    checkpoint,
    "checkpoint.pt"
)
```

Restore:

```python
checkpoint = torch.load(
    "checkpoint.pt",
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

optimizer.load_state_dict(
    checkpoint[
        "optimizer_state_dict"
    ]
)
```

---

# 53. Transformer Shapes

Make this a major compact reference.

Use:

```text
B = batch
T = sequence length
D = hidden size
H = number of heads
Dh = head dimension
V = vocabulary size
```

Main shapes:

```text
Token IDs:
(B,T)

Embeddings:
(B,T,D)

Q/K/V:
(B,T,D)

Split heads:
(B,H,T,Dh)

Attention scores:
(B,H,T,T)

Attention weights:
(B,H,T,T)

Per-head output:
(B,H,T,Dh)

Combined:
(B,T,D)

LM logits:
(B,T,V)
```

Mark:

```text
🔥 Memorize These Shapes
```

---

# 54. Q/K/V Attention

Include:

```python
Q = q_proj(x)
K = k_proj(x)
V = v_proj(x)

scores = (
    Q @ K.transpose(
        -2,
        -1
    )
) / math.sqrt(
    head_dim
)

weights = torch.softmax(
    scores,
    dim=-1
)

output = weights @ V
```

Mental model:

```text
QKᵀ
→ scores

scale
→ stabilize

softmax
→ attention weights

weights × V
→ output
```

---

# 55. Causal Mask

Include:

```python
mask = torch.triu(
    torch.ones(
        T,
        T,
        dtype=torch.bool,
        device=x.device,
    ),
    diagonal=1,
)
```

Apply:

```python
scores = scores.masked_fill(
    mask,
    float("-inf")
)
```

Meaning:

```text
prevent access to future positions
```

---

# 56. Multi-Head Attention

Include:

```python
attention = nn.MultiheadAttention(
    embed_dim=768,
    num_heads=12,
    batch_first=True
)
```

Self-attention:

```python
output, weights = attention(
    x,
    x,
    x
)
```

Important:

```text
hidden_size % num_heads == 0
```

Example:

```text
768 / 12
=
64 head dimension
```

---

# 57. Transformer FFN

Include:

```python
ffn = nn.Sequential(
    nn.Linear(
        D,
        4 * D
    ),
    nn.GELU(),
    nn.Linear(
        4 * D,
        D
    )
)
```

Input/output:

```text
(B,T,D)
→
(B,T,D)
```

---

# 58. Residual + LayerNorm

Compact:

```python
x = x + attention_output
x = norm1(x)

x = x + ffn(x)
x = norm2(x)
```

Also mention pre-norm variants exist:

```python
x = x + attention(
    norm1(x)
)
```

Do not overcomplicate.

---

# 59. Language Model Loss

Given:

```text
logits:
(B,T,V)

targets:
(B,T)
```

Use:

```python
loss = loss_fn(
    logits.reshape(
        -1,
        V
    ),
    targets.reshape(-1)
)
```

Explain:

```text
B × T token positions
→ independent vocabulary-classification examples
```

---

# 60. Next Token

Include:

```python
next_logits = logits[
    :,
    -1,
    :
]

next_token = next_logits.argmax(
    dim=-1
)
```

Sampling:

```python
probs = torch.softmax(
    next_logits,
    dim=-1
)

next_token = torch.multinomial(
    probs,
    num_samples=1
)
```

---

# 61. Common Shape Reference

Create a compact table:

| Data | Shape |
|---|---|
| Tabular | `(B,F)` |
| Binary output | `(B,1)` |
| Multiclass output | `(B,C)` |
| Image | `(B,C,H,W)` |
| Token IDs | `(B,T)` |
| Embeddings | `(B,T,D)` |
| Attention | `(B,H,T,T)` |
| LM logits | `(B,T,V)` |

This should be easy to scan.

---

# 62. Task → Output → Loss

Create the highest-value reference table:

| Task | Output | Target | Loss | Prediction |
|---|---|---|---|---|
| Regression | `(B,1)` | float | MSE/L1 | raw |
| Binary | `(B,1)` | float | BCEWithLogits | sigmoid + threshold |
| Multiclass | `(B,C)` | long `(B,)` | CrossEntropy | argmax |
| Multilabel | `(B,C)` | float `(B,C)` | BCEWithLogits | sigmoid + threshold |
| Language model | `(B,T,V)` | long `(B,T)` | CrossEntropy | next-token selection |

Mark:

```text
🔥 Highest Priority Table
```

---

# 63. Common Mistakes

Create a compact "Wrong → Correct" section.

Example:

```text
❌ Softmax before CrossEntropyLoss
✅ Pass raw logits
```

```text
❌ Sigmoid before BCEWithLogitsLoss
✅ Pass raw logits
```

```text
❌ CrossEntropy targets as float
✅ torch.long class indices
```

```text
❌ Model on GPU, input on CPU
✅ Move model and batch to same device
```

```text
❌ optimizer.step() before backward()
✅ backward() then step()
```

```text
❌ zero_grad() after backward but before step
✅ zero_grad before forward/backward
```

```text
❌ validation with model.train()
✅ model.eval()
```

```text
❌ assuming eval() disables gradients
✅ use inference_mode/no_grad too
```

```text
❌ storing loss tensor every iteration
✅ store loss.item()
```

```text
❌ Conv2d input NHWC
✅ NCHW
```

```text
❌ wrong argmax dim
✅ multiclass usually dim=1
```

```text
❌ x.to(device) without assignment
✅ x = x.to(device)
```

Include approximately 20 common mistakes.

---

# 64. Interview Quick Answers

Create approximately 25 very short Q&A items.

Examples:

### What is Autograd?

> PyTorch's automatic differentiation system. It records tensor operations and computes gradients during backward propagation.

### Why `zero_grad()`?

> PyTorch accumulates gradients by default, so old gradients normally need to be cleared before the next independent training step.

### Why no softmax before CrossEntropyLoss?

> CrossEntropyLoss expects raw logits and applies the required log-softmax-based computation internally in a numerically stable way.

### `model.eval()` vs `torch.no_grad()`?

> `eval()` changes layer behavior such as Dropout and BatchNorm; `no_grad()` disables gradient tracking.

### Dataset vs DataLoader?

> Dataset defines how to retrieve one sample; DataLoader handles batching, shuffling, multiprocessing, and iteration.

### Adam vs AdamW?

> AdamW decouples weight decay from Adam's adaptive gradient update and is common in modern deep learning.

### What is a logit?

> A raw model output score before sigmoid or softmax.

### What is self-attention?

> Attention where queries, keys, and values are derived from the same sequence.

### Why divide by `sqrt(d_k)`?

> To keep attention score magnitudes controlled before softmax.

### What is KV cache?

> Cached keys and values from previous tokens used to avoid recomputing them during autoregressive decoding.

Keep answers concise enough to speak.

---

# 65. Final "Write This From Memory" Section

End with one compact section containing the patterns the learner absolutely must be able to reproduce.

## Device

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
```

## Model

```python
class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.net(x)
```

## Training

```python
model.train()

for X, y in train_loader:

    X = X.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    logits = model(X)

    loss = loss_fn(
        logits,
        y
    )

    loss.backward()

    optimizer.step()
```

## Validation

```python
model.eval()

with torch.inference_mode():

    for X, y in val_loader:

        X = X.to(device)
        y = y.to(device)

        logits = model(X)
```

## Dataset

```python
class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y[idx]
        )
```

## Multiclass

```python
loss_fn = nn.CrossEntropyLoss()

pred = logits.argmax(
    dim=1
)
```

## Binary

```python
loss_fn = (
    nn.BCEWithLogitsLoss()
)

probs = torch.sigmoid(
    logits
)

pred = (
    probs >= 0.5
).long()
```

## Attention

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

## Save / Load

```python
torch.save(
    model.state_dict(),
    path
)

state = torch.load(
    path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    state
)
```

---

# 66. Final Memory Checklist

Add:

```text
I should know these without documentation:
```

- [ ] tensor creation
- [ ] shape / dtype / device inspection
- [ ] indexing and boolean masks
- [ ] reshape / flatten / squeeze / unsqueeze
- [ ] transpose / permute
- [ ] cat vs stack
- [ ] reductions and `dim`
- [ ] broadcasting
- [ ] matrix multiplication
- [ ] Autograd
- [ ] gradient accumulation
- [ ] `nn.Module`
- [ ] Linear / ReLU / GELU / Dropout
- [ ] BatchNorm vs LayerNorm
- [ ] Embedding
- [ ] Conv2d basics
- [ ] MSE / BCEWithLogits / CrossEntropy
- [ ] AdamW
- [ ] training loop
- [ ] validation loop
- [ ] Dataset / DataLoader
- [ ] device-aware training
- [ ] basic GPU OOM debugging
- [ ] mixed precision concept
- [ ] gradient clipping
- [ ] save/load state_dict
- [ ] Transformer shapes
- [ ] Q/K/V attention
- [ ] causal mask
- [ ] language-model output/loss shapes

---

# 67. Quality Requirements

The notebook must:

- be a valid `.ipynb`
- execute successfully from top to bottom
- work on CPU
- optionally support CUDA/MPS
- require no internet access
- use only PyTorch and Python standard library
- avoid long theoretical explanations
- avoid large datasets or lengthy training
- contain concise runnable examples
- use tables heavily for reference material
- keep each section easy to scan
- emphasize the most interview-relevant APIs
- use modern, supported PyTorch syntax
- avoid deprecated APIs
- contain no unfinished placeholder sections

The notebook should be useful in three ways:

```text
1. 30–60 minute full review
2. 5-minute targeted lookup
3. last-minute interview revision
```

The most important principle is:

```text
This is not a textbook.

It is a fast-access memory aid.
```

The learner should be able to search the notebook for:

```text
CrossEntropy
DataLoader
reshape
attention
state_dict
device
```

and immediately find the syntax, expected shapes, and the one or two most important rules.

Finally save the completed notebook as:

```text
PyTorch_Cheat_Sheet.ipynb
```