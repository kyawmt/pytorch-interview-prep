Create a Jupyter Notebook named:

```text
11_pytorch_interview_challenge.ipynb
```

This notebook is the final capstone of a **PyTorch Interview Preparation** course for someone preparing for **AI Engineer / Machine Learning Engineer / Generative AI Engineer interviews**.

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
10_transformer_pytorch.ipynb
```

Do NOT teach these topics again from scratch.

This notebook should test whether the learner can actually **use PyTorch without looking up syntax**, reason about shapes, debug broken code, construct models, write training loops, and answer common PyTorch interview questions.

This notebook should feel like a combination of:

```text
coding interview
+
ML interview
+
PyTorch practical test
+
debugging interview
+
AI Engineer take-home exercise
```

The learner should spend most of the notebook solving problems rather than reading explanations.

---

# Main Goal

By the end of this notebook, the learner should be able to confidently handle practical interview tasks involving:

- tensor manipulation
- tensor shapes
- broadcasting
- indexing
- matrix multiplication
- Autograd
- `nn.Module`
- neural-network construction
- parameter counting
- loss selection
- optimizers
- training loops
- validation loops
- Dataset / DataLoader
- device handling
- GPU-aware code
- debugging
- Transformer tensor operations
- attention
- model interpretation
- code reading
- common PyTorch interview questions

The notebook should expose any remaining weak areas before the learner starts real AI Engineer interviews.

---

# Design Philosophy

Do not structure this as a normal tutorial.

Use:

```text
Question
↓
Learner attempts
↓
Validation
↓
Hint if needed
↓
Solution hidden below
↓
Interview takeaway
```

Prefer active recall.

Avoid immediately showing solutions.

Use collapsible Markdown:

```html
<details>
<summary>Show solution</summary>

Solution here.

</details>
```

where appropriate.

For executable exercises, use:

```python
answer = None
```

or incomplete functions followed by automatic validation.

---

# Difficulty Distribution

Use approximately:

```text
🟢 Easy      25%
🟡 Medium    50%
🔴 Hard      25%
```

The notebook should become progressively harder.

---

# Notebook Structure

Use approximately:

```text
# PyTorch Interview Challenge

## 1. Interview Rules
## 2. Tensor Warm-Up
## 3. Shape Reasoning Challenge
## 4. Tensor Manipulation Challenge
## 5. Broadcasting Challenge
## 6. Matrix Multiplication Challenge
## 7. Autograd Challenge
## 8. Neural Network Construction
## 9. Parameter Counting
## 10. Loss Function Selection
## 11. Optimizer Questions
## 12. Training Loop Challenge
## 13. Validation Loop Challenge
## 14. Dataset and DataLoader Challenge
## 15. Device and GPU Challenge
## 16. Debugging Challenge
## 17. Performance Challenge
## 18. Transformer Shape Challenge
## 19. Attention Coding Challenge
## 20. Code Reading Challenge
## 21. Fix the Broken Model
## 22. Mini ML Coding Interview
## 23. Rapid-Fire Interview Questions
## 24. Mock Interview Round 1
## 25. Mock Interview Round 2
## 26. Final Capstone
## 27. Scoring
## 28. Weak-Area Review
## 29. Final Cheat Sheet
```

---

# 1. Interview Rules

Start with a Markdown section:

```text
Try each problem before opening the solution.

For coding questions:

1. Predict the result first.
2. Write the code.
3. Run the validation.
4. Fix mistakes yourself.
5. Only then open the solution.
```

Encourage completing the first attempt without internet or documentation.

Add suggested timing:

```text
Syntax question:
1–2 minutes

Shape question:
30–60 seconds

Debugging question:
2–5 minutes

Model-building question:
5–10 minutes

Full training challenge:
15–30 minutes
```

These are practice targets, not strict limits.

---

# 2. Setup

Start with:

```python
import math
import time

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

Use CPU-compatible examples.

Choose a portable device:

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

print("Device:", device)
```

---

# 3. Tensor Warm-Up

Create approximately 15 short coding exercises.

Do not provide syntax.

Examples:

### 🟢 Exercise 1

Create:

```text
tensor([1, 2, 3, 4])
```

### 🟢 Exercise 2

Create a random normal tensor:

```text
shape = (32, 10)
```

### 🟢 Exercise 3

Create integers:

```text
0 through 9
```

### 🟢 Exercise 4

Convert `x` to `float32`.

### 🟢 Exercise 5

Find the index of the maximum value.

### 🟡 Exercise 6

Select only positive values.

### 🟡 Exercise 7

Replace negative values with zero.

### 🟡 Exercise 8

Create a `4 × 4` identity matrix.

### 🟡 Exercise 9

Create values:

```text
0, 0.25, 0.5, 0.75, 1
```

### 🟡 Exercise 10

Find the top 3 values and indices.

Every coding exercise should have assertions.

---

# 4. Shape Reasoning Challenge

Create approximately 25 questions.

This should be one of the largest sections.

Examples:

```python
x = torch.randn(32, 3, 224, 224)

y = x.flatten(1)
```

Question:

```text
What is y.shape?
```

Expected:

```text
(32, 150528)
```

---

Example:

```python
x = torch.randn(16, 20, 128)

y = x.mean(dim=1)
```

Expected:

```text
(16, 128)
```

---

Example:

```python
x = torch.randn(8, 12, 64, 64)

y = x.mean(
    dim=(2, 3)
)
```

Expected:

```text
(8, 12)
```

---

Example:

```python
x = torch.randn(32, 10)

y = x.argmax(
    dim=1,
    keepdim=True
)
```

Expected:

```text
(32, 1)
```

---

Include:

- reshape
- flatten
- squeeze
- unsqueeze
- transpose
- permute
- reduction
- `cat`
- `stack`
- matrix multiplication
- embeddings
- Linear
- Conv2d
- Transformer shapes

The learner should answer before running validation.

---

# 5. Shape Trap Questions

Add harder cases designed to catch common misconceptions.

Example:

```python
a = torch.randn(4, 5)
b = torch.randn(4, 5)

torch.stack(
    [a, b],
    dim=1
)
```

Expected:

```text
(4, 2, 5)
```

---

Example:

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)

torch.cat(
    [a, b],
    dim=1
)
```

Expected:

```text
(3, 8)
```

---

Example:

```python
x = torch.randn(8, 3, 32, 32)

x.permute(
    0,
    2,
    3,
    1
)
```

Expected:

```text
(8, 32, 32, 3)
```

Include at least 10 trap questions.

---

# 6. Broadcasting Challenge

Create 15 questions.

Ask:

```text
Will it broadcast?

If yes:
what output shape?
```

Examples:

```text
(32, 10)
+
(10,)
```

Expected:

```text
YES
(32, 10)
```

---

```text
(8, 1, 128)
+
(1, 20, 1)
```

Expected:

```text
(8, 20, 128)
```

---

```text
(3, 4)
+
(3,)
```

Expected:

```text
NO
```

---

Also include practical code:

```python
x = torch.randn(64, 128)

mean = x.mean(
    dim=0,
    keepdim=True
)

normalized = x - mean
```

Ask learner to explain why this broadcasts.

---

# 7. Matrix Multiplication Challenge

Create approximately 12 questions.

Examples:

```text
(32, 128)
@
(128, 64)
```

Expected:

```text
(32, 64)
```

---

```text
A:
(16, 20, 64)

B:
(16, 64, 32)

torch.bmm(A, B)
```

Expected:

```text
(16, 20, 32)
```

---

Transformer case:

```text
Q:
(8, 12, 128, 64)

K:
(8, 12, 128, 64)

Q @ K.transpose(-2, -1)
```

Expected:

```text
(8, 12, 128, 128)
```

---

# 8. Tensor Manipulation Coding Challenges

Create approximately 15 exercises.

Examples:

### Exercise

Input:

```python
x = torch.randn(
    32,
    3,
    28,
    28
)
```

Convert to:

```text
(32, 2352)
```

without manually calculating batch size.

---

### Exercise

Convert:

```text
NCHW
```

to:

```text
NHWC
```

---

### Exercise

Given:

```python
scores = torch.randn(
    16,
    10
)
```

select the score corresponding to:

```python
labels = torch.randint(
    0,
    10,
    (16,)
)
```

without a Python loop.

Expected use may involve:

```python
torch.arange(...)
```

with advanced indexing.

---

### Exercise

Normalize each feature column independently.

---

### Exercise

Return top 5 class indices per sample.

---

# 9. Vectorization Challenge

Give Python-loop implementations and ask learner to replace them using tensor operations.

Create at least 10.

Example:

```python
result = []

for value in x:
    if value > 0:
        result.append(value)
```

Expected vectorized form:

```python
x[x > 0]
```

---

Example:

```python
for i in range(len(x)):
    x[i] = x[i] ** 2
```

Expected:

```python
x ** 2
```

---

Example:

```python
for i in range(logits.size(0)):
    predictions.append(
        logits[i].argmax()
    )
```

Expected:

```python
logits.argmax(dim=1)
```

---

# 10. Autograd Challenge

Create approximately 15 exercises.

Examples:

### Manual Gradient Prediction

```python
x = torch.tensor(
    3.0,
    requires_grad=True
)

y = 2 * x**2 + 5 * x
```

Ask:

```text
dy/dx at x=3?
```

Expected:

```text
17
```

Then validate with Autograd.

---

### Chain Rule

```python
x = torch.tensor(
    2.0,
    requires_grad=True
)

a = x * 4
y = a ** 3
```

Ask learner to predict gradient.

---

### Gradient Accumulation

Run backward twice and predict final `x.grad`.

---

### Detach

Ask whether gradient flows through a detached path.

---

### `no_grad`

Ask learner whether resulting tensor tracks gradients.

---

# 11. Explain the Bug — Autograd

Create debugging scenarios such as:

```python
x = torch.tensor(2.0)

y = x ** 2

y.backward()
```

Ask what is wrong.

---

Another:

```python
loss.backward()

optimizer.zero_grad()

optimizer.step()
```

Ask what happens.

---

Another:

```python
features = backbone(X).detach()

logits = head(features)
```

Ask why backbone gets no gradients.

---

# 12. Neural Network Construction

Create approximately 12 model-building exercises.

### 🟢 MLP

Build:

```text
10
→ 32
→ ReLU
→ 2
```

Input:

```text
(8, 10)
```

Expected:

```text
(8, 2)
```

---

### 🟡 MLP With Dropout

Build:

```text
20
→ 64
→ ReLU
→ Dropout(0.3)
→ 32
→ GELU
→ 5
```

---

### 🟡 Binary Classifier

Input:

```text
100 features
```

Output:

```text
one raw logit
```

Do NOT apply sigmoid inside the model.

---

### 🟡 Embedding Classifier

```text
token IDs
→ Embedding
→ mean pooling
→ Linear
```

---

### 🔴 CNN

Input:

```text
(batch, 3, 32, 32)
```

Output:

```text
10 logits
```

Use two Conv/ReLU/Pool blocks.

---

# 13. `nn.Module` From Memory

Give only:

```python
class MyModel(nn.Module):
    pass
```

Ask learner to implement a complete custom model including:

- `__init__`
- `super().__init__()`
- layers
- `forward()`

Validation should ensure:

```python
isinstance(
    model,
    nn.Module
)
```

and correct output shape.

---

# 14. Sequential vs Custom Module

Show an architecture and ask:

> Would `nn.Sequential` be enough?

Examples:

```text
Linear → ReLU → Linear
```

Expected:

```text
Yes
```

Versus:

```text
Embedding
→ mask-aware mean pooling
→ Linear
```

Likely:

```text
Custom nn.Module is clearer
```

Versus residual connection:

```text
x + block(x)
```

Expected:

```text
Custom module
```

Create several scenarios.

---

# 15. Parameter Counting Challenge

Create approximately 15 questions.

Examples:

```python
nn.Linear(
    128,
    64
)
```

Ask total parameters.

Expected:

```text
128 × 64 + 64
```

---

```python
nn.Embedding(
    50000,
    768
)
```

Expected:

```text
50000 × 768
```

---

```python
nn.Conv2d(
    3,
    16,
    kernel_size=3
)
```

Expected:

```text
16 × 3 × 3 × 3 + 16
```

---

Create a small model and ask total trainable parameters.

Validate using:

```python
sum(
    p.numel()
    for p in model.parameters()
)
```

---

# 16. Loss Function Selection

Create at least 20 scenarios.

For each ask:

```text
1. Output shape?
2. Target shape?
3. Target dtype?
4. Loss?
5. Inference conversion?
```

---

### Regression

Predict house price.

Expected:

```text
output:
(B,1)

target:
(B,1) float

loss:
MSELoss

inference:
raw value
```

---

### Binary Classification

Expected:

```text
output:
(B,1) raw logits

target:
(B,1) float

loss:
BCEWithLogitsLoss

inference:
sigmoid + threshold
```

---

### Multiclass

Expected:

```text
output:
(B,C)

target:
(B,) long

loss:
CrossEntropyLoss

inference:
argmax(dim=1)
```

---

### Multilabel

Expected:

```text
output:
(B,C)

target:
(B,C) float

loss:
BCEWithLogitsLoss

inference:
sigmoid + threshold per label
```

Mark this whole section:

```text
🔥 Interview Essential
```

---

# 17. Loss Function Bug Hunt

Show broken combinations.

Example:

```python
probs = torch.softmax(
    logits,
    dim=1
)

loss = nn.CrossEntropyLoss()(
    probs,
    targets
)
```

Ask learner to fix.

---

Another:

```python
probs = torch.sigmoid(logits)

loss = nn.BCEWithLogitsLoss()(
    probs,
    targets
)
```

---

Another:

```python
targets = targets.float()

loss = nn.CrossEntropyLoss()(
    logits,
    targets
)
```

---

# 18. Optimizer Challenge

Create questions such as:

> Create AdamW with:

```text
lr = 3e-4
weight_decay = 0.01
```

---

> Create SGD with:

```text
lr = 0.01
momentum = 0.9
```

---

> Freeze backbone and optimize classifier only.

Expected:

```python
optimizer = optim.AdamW(
    model.classifier.parameters(),
    lr=...
)
```

or filter by trainable parameters.

---

# 19. Training Loop From Memory

Make this one of the biggest challenges.

Give:

```python
for epoch in range(num_epochs):

    # TODO: training
```

Learner must write:

- `model.train()`
- iterate loader
- move tensors
- zero gradients
- forward
- loss
- backward
- step
- loss tracking
- accuracy tracking

Do not provide skeleton beyond the outer loop.

Validate by running against a small learnable synthetic dataset.

---

# 20. Validation Loop From Memory

Learner must write from scratch:

- `model.eval()`
- `torch.no_grad()` or `torch.inference_mode()`
- data movement
- forward
- loss
- metrics
- no backward
- no optimizer step

Validation should check parameters are unchanged.

---

# 21. Train vs Eval Questions

Create at least 10.

Example:

> Does `model.eval()` disable gradients?

Expected:

```text
No.
```

> Does `torch.no_grad()` turn off Dropout?

Expected:

```text
No.
```

> What should validation normally use?

Expected:

```text
model.eval()
+
torch.no_grad()/inference_mode()
```

---

# 22. Loss Aggregation Challenge

Give:

```python
total_loss += loss.item()
```

and:

```python
epoch_loss = (
    total_loss
    / len(dataset)
)
```

Ask why it may be incorrect.

Learner should implement sample-weighted aggregation:

```python
total_loss += (
    loss.item()
    * batch_size
)
```

then:

```python
total_loss / total_samples
```

---

# 23. Dataset Challenge

Ask learner to write from memory:

```python
class CustomDataset(Dataset):
```

Requirements:

- accepts `X` and `y`
- `__len__`
- `__getitem__`
- returns one feature/target pair

Validation should compare exact indices.

---

# 24. DataLoader Challenge

Ask learner to create:

### Training loader

```text
batch_size=32
shuffle=True
```

### Validation loader

```text
batch_size=64
shuffle=False
```

Then ask:

- what happens to partial final batch?
- how would `drop_last=True` change it?
- what does `num_workers` do?
- what does `pin_memory` do?

---

# 25. Variable-Length Data Challenge

Give:

```python
[
    tensor([1,2,3]),
    tensor([4,5]),
    tensor([6,7,8,9])
]
```

Ask learner to write:

```python
collate_fn
```

that returns padded:

```text
(3,4)
```

and attention mask.

Validate:

```text
input_ids shape
attention_mask shape
padding correctness
```

---

# 26. Device Challenge

Create approximately 15 short questions.

Examples:

> Select CUDA, MPS, or CPU.

> Move model.

> Move current batch.

> Convert GPU tensor requiring gradients to NumPy.

> Check model device.

> Explain device mismatch error.

> Explain why `.to(device)` should normally be assigned back.

> Why avoid moving model every batch?

---

# 27. GPU Memory Scenarios

Create practical interview questions.

Example:

```text
Batch 32 works.
Batch 256 causes OOM.
```

Ask what to try.

Expected possibilities:

- reduce batch
- mixed precision
- gradient accumulation
- reduce sequence/image size
- inspect memory retention

---

Example:

```text
Memory increases every training step.
```

Code:

```python
losses.append(loss)
```

Ask learner to diagnose.

---

# 28. Debugging Challenge

Create approximately **25 realistic broken snippets**.

Each should be short.

Learner must identify:

```text
bug
cause
fix
```

Include:

1. Wrong tensor shape.
2. Wrong dtype.
3. Wrong device.
4. Wrong CrossEntropy target shape.
5. Wrong BCE target shape.
6. Wrong argmax dimension.
7. Softmax before CrossEntropyLoss.
8. Sigmoid before BCEWithLogitsLoss.
9. Missing `zero_grad`.
10. Missing `backward`.
11. Missing `step`.
12. Wrong operation order.
13. Validation calling backward.
14. Validation using optimizer step.
15. Missing `model.eval`.
16. Forgetting to return to train mode.
17. Storing graph-connected loss tensors.
18. Accidental detach.
19. Frozen parameters.
20. Optimizer attached to wrong model.
21. In-place operation issue.
22. Invalid labels.
23. NaN input.
24. Learning rate too high.
25. Wrong final Linear size.

Use collapsible solutions.

---

# 29. Debugging Interview Question

Add:

```text
Your PyTorch model's loss is not decreasing.

Explain exactly how you would debug it.
```

Do not immediately provide answer.

After learner section, give model answer approximately:

```text
1. Verify data and labels.
2. Check shapes/dtypes/devices.
3. Check loss/output/target compatibility.
4. Run one batch.
5. Confirm finite loss.
6. Call backward and inspect gradients.
7. Confirm parameters change after optimizer step.
8. Check train/eval mode.
9. Check learning rate.
10. Try to overfit a tiny subset.
11. Inspect preprocessing and normalization.
```

Mark:

```text
🔥 Must Be Able To Answer Verbally
```

---

# 30. Performance Challenge

Create approximately 12 scenarios.

Examples:

### Low GPU utilization

Ask possible causes:

- DataLoader slow
- batch too small
- Python bottleneck
- frequent synchronization
- CPU/GPU transfers

---

### Tiny Python Loop

Ask learner to vectorize.

---

### Repeated concatenation

Ask learner to optimize:

```python
result = torch.cat(
    [result, x],
    dim=0
)
```

inside every iteration.

---

### Frequent `.item()`

Explain possible synchronization overhead.

---

# 31. Transformer Shape Challenge

Create at least 20 questions.

Use:

```text
B
T
D
H
Dh
V
```

Examples:

```text
x:
(8, 128, 768)

H=12
```

After splitting heads:

```text
(8, 12, 128, 64)
```

---

QKᵀ:

```text
(8, 12, 128, 128)
```

---

Weights × V:

```text
(8, 12, 128, 64)
```

---

Combined:

```text
(8, 128, 768)
```

---

LM logits:

```text
(8, 128, 50000)
```

---

# 32. Attention From Memory

Learner must implement:

```python
def attention(
    Q,
    K,
    V,
    mask=None
):
    ...
```

Requirements:

```text
scores = QKᵀ
scale
mask
softmax
multiply V
```

Validation should verify:

- output shape
- attention weights sum to 1
- masked future positions receive ~0 weight

---

# 33. Causal Mask From Memory

Ask learner to generate:

```text
T × T
```

upper triangular future mask using PyTorch.

No hints initially.

Validation:

```python
assert mask.shape == (T, T)
assert ...
```

---

# 34. Multi-Head Attention Reshape Challenge

Given:

```python
x = torch.randn(
    B,
    T,
    D
)
```

Ask learner to transform:

```text
(B,T,D)
→
(B,H,T,Dh)
```

then reverse:

```text
(B,H,T,Dh)
→
(B,T,D)
```

Validate equality after round trip.

---

# 35. Build Transformer Block

Ask learner to build:

```text
LayerNorm
→ MultiheadAttention
→ Residual
→ LayerNorm
→ FFN
→ Residual
```

Expected shape:

```text
(B,T,D)
→
(B,T,D)
```

Use small:

```text
D = 64
H = 4
```

---

# 36. Language Model Loss Challenge

Given:

```text
logits:
(B,T,V)

targets:
(B,T)
```

Ask learner to calculate token-level CrossEntropyLoss.

Expected:

```python
loss_fn(
    logits.reshape(
        -1,
        V
    ),
    targets.reshape(-1)
)
```

---

# 37. Code Reading Challenge

Create approximately 10 snippets from realistic PyTorch code.

Ask learner to explain them verbally.

Example:

```python
optimizer.zero_grad(
    set_to_none=True
)

with torch.autocast(
    device_type="cuda",
    dtype=torch.float16
):
    logits = model(X)
    loss = loss_fn(logits, y)

scaler.scale(
    loss
).backward()

scaler.step(
    optimizer
)

scaler.update()
```

Ask:

- what is happening?
- why autocast?
- why scaler?
- what does zero_grad do?

If CUDA-specific, present as a reading exercise rather than requiring execution.

---

# 38. Code Reading — NLP

Show:

```python
x = embedding(input_ids)

mask = attention_mask.unsqueeze(-1)

x = x * mask

x = x.sum(dim=1)

lengths = mask.sum(
    dim=1
).clamp(min=1)

x = x / lengths
```

Ask learner to explain:

> What pooling operation is being implemented?

Expected:

```text
masked mean pooling
```

---

# 39. Code Reading — CNN

Show architecture.

Ask learner to calculate shape after every layer.

---

# 40. Code Reading — Frozen Backbone

Show:

```python
for p in model.backbone.parameters():
    p.requires_grad = False

optimizer = optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=1e-3
)
```

Ask learner to explain.

---

# 41. Rapid-Fire Interview Questions

Create approximately **50 questions** with hidden answers.

Answers should be short enough to speak in an interview.

Include questions such as:

1. Tensor vs NumPy array?
2. `view()` vs `reshape()`?
3. `cat()` vs `stack()`?
4. `squeeze()` vs `unsqueeze()`?
5. `transpose()` vs `permute()`?
6. What is broadcasting?
7. `max()` vs `argmax()`?
8. What is Autograd?
9. What does `requires_grad` do?
10. Why zero gradients?
11. `detach()` vs `no_grad()`?
12. What is a leaf tensor?
13. What is `nn.Module`?
14. Why `super().__init__()`?
15. What does `forward()` do?
16. Why call `model(x)`?
17. What is a logit?
18. Why ReLU?
19. ReLU vs GELU?
20. BatchNorm vs LayerNorm?
21. What does Dropout do?
22. What does `model.train()` do?
23. What does `model.eval()` do?
24. `eval()` vs `no_grad()`?
25. MSE vs L1?
26. When use BCEWithLogitsLoss?
27. When use CrossEntropyLoss?
28. Why no softmax before CrossEntropyLoss?
29. Why no sigmoid before BCEWithLogitsLoss?
30. What does an optimizer do?
31. SGD vs Adam?
32. Adam vs AdamW?
33. What is weight decay?
34. What is batch size?
35. What is an epoch?
36. Dataset vs DataLoader?
37. Why shuffle training data?
38. What does `num_workers` do?
39. What does `pin_memory` do?
40. Why GPU OOM?
41. What is mixed precision?
42. FP16 vs BF16?
43. What is gradient clipping?
44. What is gradient accumulation?
45. What is self-attention?
46. What are Q/K/V?
47. Why divide attention by `sqrt(d_k)`?
48. What is causal masking?
49. What is KV cache?
50. Why does attention scale quadratically with sequence length?

Use collapsible answers.

---

# 42. Mock Interview Round 1 — PyTorch Fundamentals

Create 10 questions.

Do not provide hints before the learner attempts them.

Suggested mix:

```text
2 syntax questions
2 shape questions
2 model questions
1 loss question
1 training-loop question
1 debugging question
1 verbal question
```

Add a scoring section:

```text
0 = incorrect
1 = partial
2 = correct
```

Maximum:

```text
20 points
```

---

# 43. Mock Interview Round 2 — AI Engineer

Create 10 harder questions.

Suggested mix:

- Dataset/DataLoader
- device handling
- performance debugging
- Transformer shape
- attention implementation
- training bug
- OOM scenario
- parameter freezing
- model architecture
- system-style ML reasoning

Again score:

```text
0 / 1 / 2
```

---

# 44. Mock Live Coding Exercise

Create one interview-style problem:

> Build and train a multiclass classifier from scratch using PyTorch.

Provide only:

```python
torch.manual_seed(42)

X = torch.randn(
    1000,
    20
)

true_w = torch.randn(
    20,
    4
)

y = (
    X @ true_w
).argmax(
    dim=1
)
```

Learner must:

1. split train/validation
2. create DataLoaders
3. build MLP
4. select correct loss
5. select optimizer
6. write train function
7. write evaluate function
8. train
9. report accuracy
10. make code device-aware

Do not provide the architecture.

Learner should choose one.

Validation should confirm:

```text
model output shape = (B,4)
training works
validation accuracy is meaningful
```

---

# 45. Follow-Up Interview Questions for Live Coding

After the coding task ask:

> Why did you choose this loss?

> Why did you not apply softmax?

> Why did you use `model.train()`?

> Why did you use `model.eval()`?

> What would you change if the dataset were highly imbalanced?

> What if the model overfits?

> What if GPU OOM occurs?

> What if validation accuracy stays at random chance?

> How would you make training faster?

> How would you save the model?

Provide hidden model answers.

---

# 46. Final Capstone — Broken AI Pipeline

Create a complete broken PyTorch pipeline with approximately **15 issues**.

Pipeline should contain:

- custom Dataset
- DataLoader
- MLP
- loss
- optimizer
- training
- validation
- device handling
- metric tracking

Intentionally include problems such as:

1. wrong `__len__`
2. wrong target dtype
3. validation shuffled unnecessarily
4. model wrong input size
5. output class count mismatch
6. optimizer on wrong parameters
7. softmax before CrossEntropyLoss
8. missing `model.train()`
9. zero gradients at wrong time
10. incorrect `argmax` dimension
11. graph-connected loss saved
12. validation lacks `eval()`
13. validation lacks no-grad
14. model/data device mismatch
15. excessively high learning rate

The learner should fix the entire pipeline.

---

# 47. Capstone Debugging Procedure

Require learner to diagnose using this sequence:

```text
1. Inspect dataset sample.
2. Inspect batch.
3. Inspect model output shape.
4. Inspect target dtype/shape.
5. Run one forward pass.
6. Confirm finite loss.
7. Run backward.
8. Inspect gradients.
9. Confirm optimizer changes parameters.
10. Overfit 32 examples.
11. Run full training.
12. Run proper validation.
```

This should reinforce disciplined debugging.

---

# 48. Capstone Validation

Automatically verify:

```python
assert ...
```

for:

- dataset length
- batch shapes
- target dtype
- output shape
- finite loss
- gradients present
- finite gradients
- parameter updates
- accuracy range
- validation does not update parameters
- history stores Python numbers rather than graph tensors

Do not require exact final weights.

---

# 49. Timed Syntax Round

Create approximately 25 ultra-short prompts.

Examples:

```text
Create random normal tensor (8, 16)
```

```text
Convert x to long
```

```text
Argmax across classes
```

```text
Flatten except batch
```

```text
Move x to device
```

```text
Detach and move to CPU
```

```text
Create Linear 128→10
```

```text
Create LayerNorm 768
```

```text
Create embedding vocab=50k, dim=768
```

```text
Create AdamW lr=1e-4
```

```text
Disable gradients
```

```text
Set eval mode
```

Goal:

```text
complete in ~10 minutes
```

Provide answers later.

---

# 50. Timed Shape Round

Create 20 questions intended to be completed quickly.

Target:

```text
10 minutes
```

Heavily test:

```text
Linear
Embedding
Conv
Flatten
Reduction
Attention
LM logits
```

---

# 51. Scoring System

Create sections with a points system.

Example:

```text
Tensor fundamentals:       /20
Shape reasoning:           /25
Autograd:                  /15
Models:                    /20
Loss / optimizer:          /20
Training loops:            /25
Data pipeline:             /15
GPU / performance:         /15
Debugging:                 /25
Transformers:              /25

Total:                     /200
```

---

# 52. Score Interpretation

Provide rough guidance:

```text
180–200
Strong practical PyTorch readiness

155–179
Good interview readiness; review a few weak areas

125–154
Functional but several gaps remain

90–124
Needs targeted practice before interviews

Below 90
Repeat core notebooks before relying on PyTorch in interviews
```

Clarify this is a study benchmark, not a scientifically validated hiring score.

---

# 53. Weak-Area Tracker

Create a Markdown table:

| Topic | Score | Confidence | Review Notebook |
|---|---:|---|---|
| Tensors | | | 01–02 |
| Autograd | | | 03 |
| Models | | | 04 |
| Losses | | | 05 |
| Training | | | 06 |
| DataLoader | | | 07 |
| GPU | | | 08 |
| Debugging | | | 09 |
| Transformers | | | 10 |

Ask learner to fill it after finishing.

---

# 54. Mistake Log

Add:

```text
## My PyTorch Mistake Log
```

Create table:

| Mistake | Why I made it | Correct pattern |
|---|---|---|
| | | |
| | | |
| | | |

Encourage recording repeated mistakes.

---

# 55. Interview Answer Practice

For key verbal questions, require answer in:

```text
30 seconds or less
```

Examples:

> Explain `model.eval()` vs `torch.no_grad()`.

> Explain CrossEntropyLoss input format.

> Explain why PyTorch gradients accumulate.

> Explain Dataset vs DataLoader.

> Explain Q/K/V attention.

Then give high-quality short sample answers.

---

# 56. Final "Must Know From Memory" Section

End with a section containing the most important code patterns the learner should be able to write without documentation.

Include:

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

## Multiclass

```python
loss_fn = nn.CrossEntropyLoss()

logits = model(X)

loss = loss_fn(
    logits,
    y
)

pred = logits.argmax(
    dim=1
)
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

## Device

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
```

## Attention

```python
scores = (
    Q @ K.transpose(-2, -1)
) / math.sqrt(head_dim)

scores = scores.masked_fill(
    mask,
    float("-inf")
)

weights = torch.softmax(
    scores,
    dim=-1
)

output = weights @ V
```

---

# 57. Final Concept Checklist

Add:

```text
I should be able to explain all of these without notes:
```

- [ ] Tensor shape reasoning
- [ ] Broadcasting
- [ ] `cat` vs `stack`
- [ ] Matrix multiplication
- [ ] Autograd
- [ ] Gradient accumulation
- [ ] `detach` vs `no_grad`
- [ ] `nn.Module`
- [ ] `forward`
- [ ] logits
- [ ] ReLU / GELU
- [ ] BatchNorm vs LayerNorm
- [ ] CrossEntropyLoss
- [ ] BCEWithLogitsLoss
- [ ] SGD / Adam / AdamW
- [ ] training-loop order
- [ ] train vs eval
- [ ] Dataset vs DataLoader
- [ ] device handling
- [ ] GPU OOM
- [ ] mixed precision basics
- [ ] debugging non-decreasing loss
- [ ] gradient inspection
- [ ] self-attention
- [ ] Q/K/V
- [ ] causal masking
- [ ] multi-head attention
- [ ] Transformer tensor shapes
- [ ] language-model logits

---

# 58. Final Coding Checklist

Add:

```text
I should be able to write these from memory:
```

- [ ] create tensors
- [ ] reshape / permute tensors
- [ ] vectorized masks
- [ ] matrix multiplication
- [ ] custom `nn.Module`
- [ ] MLP
- [ ] embedding classifier
- [ ] basic CNN
- [ ] select loss
- [ ] create AdamW
- [ ] training loop
- [ ] evaluation loop
- [ ] custom Dataset
- [ ] DataLoader
- [ ] custom padding `collate_fn`
- [ ] device-aware code
- [ ] finite-value debugging
- [ ] gradient inspection
- [ ] scaled dot-product attention
- [ ] causal mask
- [ ] Transformer block

---

# 59. Quality Requirements

The notebook must:

- be a valid `.ipynb`
- execute successfully from top to bottom
- work on CPU-only machines
- optionally use CUDA/MPS when available
- require no internet
- require only PyTorch and Python standard library
- contain no empty placeholder sections
- contain meaningful automatic validation
- contain approximately **150–200 total questions/exercises**
- contain at least **25 debugging exercises**
- contain at least **20 shape questions**
- contain at least **20 rapid-fire interview questions**
- include at least **2 mock interview rounds**
- include at least **1 complete live coding challenge**
- include at least **1 full broken-pipeline debugging capstone**
- emphasize active recall rather than explanations
- avoid introducing major new PyTorch topics
- reuse knowledge from notebooks 01–10
- remain focused on AI Engineer interview readiness

The notebook should take approximately **4–6 hours** to complete thoroughly, but individual sections should also work independently for revision.

---

# 60. Most Important Principle

The notebook should enforce:

```text
Can I recognize the code?
```

is NOT enough.

The learner needs to reach:

```text
Can I write it?
Can I explain it?
Can I predict the shape?
Can I debug it?
Can I use it under interview pressure?
```

The core loop is:

```text
Question
   ↓
Recall
   ↓
Write
   ↓
Run
   ↓
Debug
   ↓
Explain
   ↓
Repeat
```

Finally save the completed notebook as:

```text
11_pytorch_interview_challenge.ipynb
```