Create a Jupyter Notebook named:

```text
09_debugging_and_performance.ipynb
```

This notebook is the ninth part of a **PyTorch Interview Preparation** course for someone preparing for **AI Engineer / Machine Learning Engineer interviews**.

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
```

and already understands:

- tensors and tensor shapes
- Autograd
- model construction
- loss functions
- optimizers
- training and validation loops
- Dataset / DataLoader
- CPU / CUDA / MPS device handling
- basic GPU memory concepts
- mixed precision basics

Do not spend significant time reteaching those topics.

The purpose of this notebook is to teach the learner how to **debug real PyTorch training problems and identify performance bottlenecks**.

This notebook should feel closer to debugging an actual ML project than learning individual API syntax.

The learner should repeatedly inspect broken code, identify likely causes, fix it, and explain the reasoning.

---

# Main Learning Goals

By the end of this notebook, the learner should be able to diagnose problems such as:

```text
shape mismatch
dtype mismatch
device mismatch
loss not decreasing
NaN / Inf loss
exploding gradients
vanishing gradients
CUDA out of memory
GPU memory growth
slow DataLoader
low GPU utilization
incorrect train/eval behavior
incorrect gradient flow
accidental graph retention
bad learning rate
wrong loss function
wrong activation / loss combination
wrong target format
in-place Autograd errors
```

The learner should also understand and use:

```python
tensor.shape
tensor.dtype
tensor.device
tensor.requires_grad
tensor.grad

torch.isnan()
torch.isinf()
torch.isfinite()

torch.autograd.set_detect_anomaly()

torch.nn.utils.clip_grad_norm_()

model.train()
model.eval()

torch.no_grad()
torch.inference_mode()

torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()

torch.cuda.synchronize()

time.perf_counter()
```

Also introduce at a practical level:

```python
torch.profiler
```

and optionally:

```python
torch.utils.benchmark
```

without making advanced profiling the main focus.

Mark these as:

```text
🔥 Interview Essential
```

- systematic debugging workflow
- shape / dtype / device inspection
- NaN / Inf debugging
- gradient inspection
- exploding vs vanishing gradients
- loss-not-decreasing diagnosis
- train/eval bugs
- graph retention / memory leaks
- CUDA OOM diagnosis
- GPU utilization diagnosis
- DataLoader bottlenecks
- learning-rate problems

---

# Core Debugging Philosophy

The notebook should teach a repeatable process:

```text
1. Reproduce the problem
        ↓
2. Check shapes
        ↓
3. Check dtypes
        ↓
4. Check devices
        ↓
5. Check values for NaN / Inf
        ↓
6. Check loss and targets
        ↓
7. Check gradients
        ↓
8. Check parameter updates
        ↓
9. Check train/eval modes
        ↓
10. Check memory / performance
```

Emphasize:

> Do not randomly change hyperparameters until the bug disappears. Inspect the pipeline systematically.

---

# Notebook Structure

Use approximately:

```text
# PyTorch Debugging and Performance

## 1. Setup
## 2. A Systematic Debugging Workflow
## 3. Inspecting Tensor State
## 4. Shape Mismatches
## 5. Dtype Mismatches
## 6. Device Mismatches
## 7. Target and Loss Mismatches
## 8. Debugging Loss That Does Not Decrease
## 9. Checking Parameter Updates
## 10. Checking Gradients
## 11. Vanishing Gradients
## 12. Exploding Gradients
## 13. NaN and Inf Debugging
## 14. Anomaly Detection
## 15. In-Place Operation Errors
## 16. train() / eval() Bugs
## 17. Dropout and BatchNorm Debugging
## 18. Memory Leaks and Graph Retention
## 19. CUDA OOM Debugging
## 20. DataLoader Bottlenecks
## 21. Low GPU Utilization
## 22. Timing Code Correctly
## 23. Basic PyTorch Profiling
## 24. Vectorization and Python Loops
## 25. Efficient Metric Logging
## 26. Common Performance Anti-Patterns
## 27. Debugging Scenarios
## 28. Interview Questions
## 29. Knowledge Check Quiz
## 30. Write From Memory
## 31. Final Debugging Challenge
## 32. Debugging Cheat Sheet
```

---

# 1. Setup

Start with:

```python
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
```

Use small synthetic datasets.

All exercises must:

- work on CPU
- require no internet
- avoid intentionally allocating dangerous amounts of memory
- conditionally execute CUDA-specific code

---

# 2. Build a Reusable Debug Helper

Create a function such as:

```python
def inspect_tensor(name, x):
    print(f"{name}")
    print("  shape:", x.shape)
    print("  dtype:", x.dtype)
    print("  device:", x.device)
    print("  requires_grad:", x.requires_grad)

    if x.numel() > 0 and x.is_floating_point():
        print("  min:", x.min().item())
        print("  max:", x.max().item())
        print("  mean:", x.mean().item())
        print("  finite:", torch.isfinite(x).all().item())
```

Explain this is a simple debugging utility, not a production logging system.

Use it throughout the notebook.

---

# 3. First Rule: Check Shape, Dtype, Device

Mark:

```text
🔥 Interview Essential
```

Teach the three fastest checks:

```python
print(x.shape)
print(x.dtype)
print(x.device)
```

Explain:

> A very large fraction of PyTorch runtime errors come from incompatible shapes, dtypes, or devices.

Show a debugging mental checklist:

```text
What shape is coming in?
What shape does the layer expect?

What dtype is coming in?
What dtype does the loss expect?

What device is the model on?
What device is the tensor on?
```

---

# 4. Shape Mismatch — Linear Layer

Show broken code:

```python
model = nn.Linear(10, 3)

X = torch.randn(32, 8)

logits = model(X)
```

Do not execute the failing cell directly unless wrapped safely.

Ask:

> What is wrong?

Expected:

```text
The model expects 10 input features, but X has 8.
```

Correct:

```python
X = torch.randn(32, 10)
```

or redesign the layer.

---

# 5. Shape Mismatch — CrossEntropyLoss

Show:

```python
logits = torch.randn(32, 10)
targets = torch.randint(0, 10, (32, 1))
```

Explain common expected class-index shape:

```text
logits:
(32, 10)

targets:
(32,)
```

Ask learner to fix:

```python
targets = targets.squeeze(1)
```

when appropriate.

Make clear not to blindly `squeeze()` unknown dimensions.

---

# 6. Shape Mismatch — BCE

Show:

```text
logits:
(32, 1)

targets:
(32,)
```

Explain that BCE-style losses commonly expect matching shapes.

Possible correction:

```python
targets = targets.unsqueeze(1)
```

or design model/targets consistently.

Create several exercises comparing:

```text
(batch,)
(batch, 1)
(batch, classes)
```

---

# 7. Shape Mismatch — CNN

Show:

```python
conv = nn.Conv2d(3, 16, kernel_size=3)

X = torch.randn(8, 224, 224, 3)
```

Ask why it fails.

Expected:

> PyTorch `Conv2d` expects NCHW, but the tensor is NHWC.

Possible correction:

```python
X = X.permute(0, 3, 1, 2)
```

Validate shape:

```text
(8, 3, 224, 224)
```

---

# 8. Dtype Mismatch

Mark:

```text
🔥 Interview Essential
```

Show:

```python
embedding = nn.Embedding(1000, 64)

tokens = torch.tensor([
    [1.0, 2.0, 3.0]
])
```

Ask why this fails.

Expected:

> Embedding indices should use an integer index dtype such as `torch.long`.

Fix:

```python
tokens = tokens.long()
```

---

# 9. CrossEntropy Target Dtype

Show:

```python
logits = torch.randn(16, 5)

targets = torch.tensor(
    [0., 1., 4., 2., ...]
)
```

Explain that normal class-index targets should be:

```python
targets = targets.long()
```

Create a debugging exercise.

---

# 10. Model Precision Mismatch

Introduce conceptually:

```text
model parameters → float32
input           → float64
```

This may cause errors or unwanted conversions depending on operation.

Teach:

```python
X = X.float()
```

and:

```python
next(model.parameters()).dtype
```

as a quick diagnostic.

---

# 11. Device Mismatch

Review with debugging emphasis.

Use:

```python
print(next(model.parameters()).device)
print(X.device)
print(y.device)
```

Show a scenario:

```text
Model device: cuda:0
Input device: cpu
Target device: cuda:0
```

Ask learner to identify the problem immediately.

---

# 12. Build `inspect_model()`

Create helper:

```python
def inspect_model(model):
    total = 0
    trainable = 0

    for name, param in model.named_parameters():
        total += param.numel()

        if param.requires_grad:
            trainable += param.numel()

        print(
            name,
            param.shape,
            param.dtype,
            param.device,
            param.requires_grad
        )

    print("Total parameters:", total)
    print("Trainable parameters:", trainable)
```

Use later for frozen-layer bugs.

---

# 13. Wrong Loss Function

Create scenarios.

### Scenario A

Model output:

```text
(batch, 10)
```

one class among 10.

Using:

```python
nn.BCEWithLogitsLoss()
```

Ask whether this is the usual correct formulation.

Expected:

```text
No. Standard single-label multiclass classification usually uses CrossEntropyLoss.
```

---

### Scenario B

Multilabel output:

```text
(batch, 10)
```

where multiple labels can be 1.

Using:

```python
CrossEntropyLoss
```

Ask what is likely wrong.

Expected:

```text
Multilabel commonly uses BCEWithLogitsLoss with float multi-hot targets.
```

---

# 14. Wrong Activation Before Loss

Mark:

```text
🔥 Very Common Interview Bug
```

Show:

```python
logits = model(X)

probs = torch.softmax(
    logits,
    dim=1
)

loss = nn.CrossEntropyLoss()(
    probs,
    y
)
```

Ask:

> What should be changed?

Expected:

```python
loss = nn.CrossEntropyLoss()(
    logits,
    y
)
```

Similarly include:

```python
torch.sigmoid(logits)
```

before:

```python
BCEWithLogitsLoss
```

as another bug.

---

# 15. Loss Does Not Decrease

Make this one of the largest sections.

Mark:

```text
🔥 Interview Essential
```

Create a checklist:

```text
If loss is not decreasing:

1. Check data / labels.
2. Check output and target shapes.
3. Check loss function.
4. Check learning rate.
5. Check gradients exist.
6. Check parameters update.
7. Check activation / loss pairing.
8. Check train/eval mode.
9. Try to overfit a tiny dataset.
10. Inspect normalization / preprocessing.
```

Explain each briefly.

---

# 16. Tiny-Batch Overfitting Test

Teach an extremely useful debugging technique:

> Try to overfit a very small dataset, such as 8–32 examples.

Why?

If the model cannot memorize a tiny dataset, likely problems include:

- bug in training loop
- wrong loss
- wrong targets
- gradients not flowing
- learning rate issue
- model capacity issue

Build a synthetic example and demonstrate successful overfitting.

Mark:

```text
🔥 Excellent Practical Debugging Technique
```

---

# 17. Exercise — Broken Tiny Dataset Training

Provide a deliberately broken training loop with one issue, such as missing:

```python
loss.backward()
```

Ask learner to identify why loss does not decrease.

Validation should confirm parameter updates after the fix.

---

# 18. Check Whether Parameters Actually Change

Teach:

```python
before = {
    name: param.detach().clone()
    for name, param
    in model.named_parameters()
}
```

Run one training step.

Then:

```python
for name, param in model.named_parameters():

    changed = not torch.equal(
        before[name],
        param.detach()
    )

    print(name, changed)
```

Explain:

> If no parameters change after an optimizer step, inspect gradients, optimizer parameters, frozen flags, and training order.

---

# 19. Common Bug — Optimizer Created With Wrong Parameters

Show conceptual example:

```python
model_a = nn.Linear(10, 3)
model_b = nn.Linear(10, 3)

optimizer = optim.Adam(
    model_a.parameters(),
    lr=1e-3
)

logits = model_b(X)
loss = loss_fn(logits, y)

loss.backward()
optimizer.step()
```

Ask what is wrong.

Expected:

> The optimizer updates `model_a`, while the forward pass uses `model_b`.

This is a realistic debugging problem.

---

# 20. Frozen Parameter Bug

Show:

```python
for param in model.parameters():
    param.requires_grad = False
```

Then later train without re-enabling gradients.

Use:

```python
for name, param in model.named_parameters():
    print(name, param.requires_grad)
```

Teach learner to inspect trainability.

---

# 21. Gradient Inspection

Mark:

```text
🔥 Interview Essential
```

After:

```python
loss.backward()
```

teach:

```python
for name, param in model.named_parameters():

    if param.grad is None:
        print(name, "NO GRAD")
    else:
        print(
            name,
            param.grad.norm().item()
        )
```

Explain interpretations:

```text
grad is None
→ parameter did not receive a gradient

grad norm ≈ 0
→ possibly vanishing / inactive path

very large grad norm
→ possible instability / exploding gradient
```

Do not use hard universal thresholds.

---

# 22. Build `gradient_report()`

Create helper:

```python
def gradient_report(model):

    for name, param in model.named_parameters():

        if not param.requires_grad:
            print(name, "FROZEN")
            continue

        if param.grad is None:
            print(name, "NO GRAD")
            continue

        grad = param.grad

        print(
            name,
            "norm=",
            grad.norm().item(),
            "finite=",
            torch.isfinite(grad).all().item()
        )
```

Use it in exercises.

---

# 23. Vanishing Gradients

Explain simply:

> Vanishing gradients occur when gradients become extremely small in earlier layers, making learning very slow.

Possible contributors:

- very deep networks
- saturating activations such as sigmoid/tanh
- poor initialization
- repeated multiplicative effects

Do not imply these are the only causes.

Show a small educational example if feasible.

---

# 24. Exploding Gradients

Explain:

> Exploding gradients occur when gradient magnitudes become extremely large, causing unstable parameter updates.

Symptoms:

```text
loss suddenly becomes huge
NaN loss
very large grad norms
unstable parameters
```

Potential responses:

- reduce learning rate
- gradient clipping
- better initialization
- normalization
- inspect data
- architecture adjustments

---

# 25. Gradient Clipping

Review:

```python
loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

optimizer.step()
```

Show before/after gradient norm.

Make clear:

> Clipping may control symptoms, but if gradients explode because of a deeper bug, diagnose the root cause too.

---

# 26. NaN and Inf Detection

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
torch.isnan(x)
torch.isinf(x)
torch.isfinite(x)
```

Examples:

```python
x = torch.tensor([
    1.0,
    float("nan"),
    float("inf")
])

print(torch.isnan(x))
print(torch.isinf(x))
print(torch.isfinite(x))
```

---

# 27. NaN Check Pattern

Teach:

```python
if not torch.isfinite(loss):
    print("Non-finite loss!")
```

For tensors:

```python
assert torch.isfinite(X).all()
```

Use diagnostic checks rather than silently continuing.

---

# 28. Common Sources of NaN

Discuss:

- learning rate too high
- divide by zero
- log of invalid values
- overflow
- unstable FP16 computation
- exploding gradients
- bad input data containing NaN/Inf
- invalid normalization
- invalid target values

Give examples.

---

# 29. Division by Zero

Show:

```python
x = torch.tensor([1.0, 2.0])

std = torch.tensor(0.0)

normalized = x / std
```

Explain resulting non-finite values.

Fix pattern:

```python
normalized = x / (std + 1e-8)
```

but explain epsilon is appropriate only when mathematically justified.

---

# 30. Invalid Log

Show:

```python
x = torch.tensor([
    1.0,
    0.0,
    -1.0
])

torch.log(x)
```

Explain:

- log(1) finite
- log(0) → `-inf`
- log(negative) → `nan`

This is a useful source-of-NaN example.

---

# 31. Check Input Data Before Training

Teach:

```python
print(
    "X finite:",
    torch.isfinite(X).all().item()
)

print(
    "y finite:",
    torch.isfinite(y.float()).all().item()
)
```

Also inspect:

```python
X.min()
X.max()
X.mean()
X.std()
```

Explain wildly scaled features can contribute to optimization difficulty.

---

# 32. `torch.autograd.set_detect_anomaly`

Mark:

```text
🟡 Interview Useful
```

Teach:

```python
torch.autograd.set_detect_anomaly(True)
```

or scoped usage where supported.

Explain:

> Anomaly detection can help identify the backward operation that generated NaN or invalid gradients.

Important:

> It slows execution significantly, so use it for debugging rather than normal training.

---

# 33. Anomaly Detection Exercise

Create a safe example involving a problematic operation and demonstrate how anomaly detection helps identify the source.

Do not intentionally make the full notebook execution fail.

Use `try/except` if needed.

---

# 34. In-Place Operation Errors

Teach PyTorch underscore convention:

```python
x.add_()
x.relu_()
```

Explain:

> In-place operations modify the existing tensor and can interfere with Autograd when the original value is needed for backward.

Show conceptual error:

```text
one of the variables needed for gradient computation
has been modified by an inplace operation
```

Ask learner what to inspect.

Expected:

> Look for operations ending in `_` or assignments that overwrite tensors needed for backward.

---

# 35. `model.train()` / `model.eval()` Bug

Mark:

```text
🔥 Interview Essential
```

Create model with Dropout:

```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(20, 2)
)
```

Show repeated forward passes in:

```python
model.train()
```

vs:

```python
model.eval()
```

Explain differing outputs due to Dropout.

---

# 36. Forgetting to Return to `train()`

Show pattern:

```python
for epoch in range(num_epochs):

    # training
    ...

    model.eval()
    with torch.no_grad():
        ...

    # next epoch begins
```

Ask:

> What may be missing?

Expected:

```python
model.train()
```

at the start of each training phase.

Explain why Dropout/BatchNorm behavior would otherwise remain incorrect.

---

# 37. BatchNorm Debugging

Explain common issues:

- evaluating while still in train mode
- tiny batch sizes
- using incorrect feature/channel dimension
- frozen/eval behavior misunderstood

Show:

```text
model.eval()
```

is particularly important for BatchNorm's running statistics.

---

# 38. Memory Leak — Saving Loss Tensors

Review:

```python
losses.append(loss)
```

Explain graph retention.

Fix:

```python
losses.append(
    loss.item()
)
```

or:

```python
losses.append(
    loss.detach().cpu()
)
```

depending on need.

---

# 39. Memory Leak — Saving Predictions

Show:

```python
all_predictions.append(
    predictions
)
```

During training this may retain GPU tensors and graphs.

Safer for metric collection:

```python
all_predictions.append(
    predictions.detach().cpu()
)
```

if gradients are unnecessary.

---

# 40. Memory Growth — `retain_graph=True`

Show:

```python
loss.backward(
    retain_graph=True
)
```

Explain:

> `retain_graph=True` keeps the graph for additional backward passes and can increase memory usage.

Do not use it casually to suppress a repeated-backward error.

Mark:

```text
🔥 Common Misuse
```

---

# 41. CUDA OOM Debugging Workflow

Create a checklist:

```text
1. Confirm actual batch size.
2. Check tensor / sequence / image sizes.
3. Check whether outputs or losses are being stored on GPU.
4. Check retain_graph usage.
5. Check whether multiple models are in memory.
6. Reduce batch size.
7. Try mixed precision.
8. Consider gradient accumulation.
9. Inspect CUDA memory.
10. Profile before more advanced changes.
```

---

# 42. Safe CUDA Memory Helper

Only on CUDA:

```python
def cuda_memory_report():

    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return

    print(
        "allocated MB:",
        torch.cuda.memory_allocated()
        / 1024**2
    )

    print(
        "reserved MB:",
        torch.cuda.memory_reserved()
        / 1024**2
    )

    print(
        "peak allocated MB:",
        torch.cuda.max_memory_allocated()
        / 1024**2
    )
```

---

# 43. Reset Peak Memory Stats

Optionally teach:

```python
torch.cuda.reset_peak_memory_stats()
```

Use when measuring peak memory for a specific code section.

Mark intermediate.

---

# 44. DataLoader Bottleneck

Mark:

```text
🔥 Interview Essential for ML Engineering
```

Scenario:

```text
GPU utilization: 20%
GPU repeatedly waits for next batch
```

Possible causes:

- disk too slow
- preprocessing expensive
- too few workers
- batch too small
- CPU saturated
- excessive Python work in Dataset
- no overlap in transfer/compute

Explain how to distinguish compute-bound vs input-bound at a basic level.

---

# 45. Measure Data Loading Time

Show:

```python
start = time.perf_counter()

for i, batch in enumerate(loader):

    if i == 100:
        break

elapsed = time.perf_counter() - start

print(elapsed)
```

Explain this isolates data iteration more than full model training.

---

# 46. Measure Compute Separately

Create synthetic tensor batches already in memory and time forward/backward independently.

The point:

```text
Data loading time
vs
model compute time
```

Compare to identify bottlenecks.

---

# 47. `num_workers` Experiment

If safe, compare:

```text
num_workers=0
```

and perhaps a small positive number when not in environments where multiprocessing is problematic.

Because Jupyter/macOS/Windows behavior can vary, make the actual benchmark optional.

Explain conceptually:

> Benchmark rather than assuming more workers are faster.

---

# 48. Low GPU Utilization

Create reasons table:

| Cause | Possible Sign |
|---|---|
| DataLoader slow | gaps between batches |
| Batch too small | low compute occupancy |
| Model tiny | GPU overhead dominates |
| CPU work in loop | GPU waiting |
| Frequent `.item()` | synchronization |
| Frequent prints | synchronization / overhead |
| CPU-GPU transfers | transfer-bound |
| Python loops | kernel-launch overhead |

Do not overstate that any single sign proves the cause.

---

# 49. Frequent `.item()` Calls

Show inefficient:

```python
for X, y in loader:
    ...
    print(loss.item())
```

every iteration.

Explain potential issues:

- CPU/GPU synchronization
- printing overhead

Better:

```python
if step % 100 == 0:
    print(loss.item())
```

depending on logging needs.

---

# 50. Timing CPU Code

Teach:

```python
start = time.perf_counter()

y = operation(x)

elapsed = (
    time.perf_counter()
    - start
)
```

Use multiple repetitions for more reliable results.

---

# 51. Timing CUDA Correctly

Review:

```python
if torch.cuda.is_available():

    torch.cuda.synchronize()

    start = time.perf_counter()

    y = model(X)

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )
```

Explain:

> CUDA launches can be asynchronous, so without synchronization the timer may only measure dispatch overhead.

---

# 52. Warmup

Teach benchmarking pattern:

```python
for _ in range(10):
    _ = model(X)

# then measure
```

For CUDA, synchronize appropriately.

Explain one-time startup effects.

---

# 53. Average Over Repetitions

Teach:

```python
times = []

for _ in range(100):
    ...
```

Then:

```python
sum(times) / len(times)
```

or use a benchmarking utility.

Explain single-run timings are noisy.

---

# 54. `torch.utils.benchmark`

Briefly introduce:

```python
import torch.utils.benchmark as benchmark
```

Example:

```python
timer = benchmark.Timer(
    stmt="x @ y",
    globals={
        "x": x,
        "y": y
    }
)

print(
    timer.timeit(100)
)
```

Only use if available in the installed PyTorch build, which it generally is.

Explain it is more robust than ad hoc timing for small operations.

---

# 55. `torch.profiler`

Introduce practical basics.

Mark:

```text
🟡 Interview Useful
```

Teach concept:

> Profiler shows where execution time and memory are spent across PyTorch operations.

Use a minimal CPU example:

```python
from torch.profiler import (
    profile,
    ProfilerActivity
)
```

Then:

```python
with profile(
    activities=[
        ProfilerActivity.CPU
    ]
) as prof:

    y = model(X)

print(
    prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=10
    )
)
```

If CUDA exists, optionally include:

```python
ProfilerActivity.CUDA
```

Do not turn this into a full profiler tutorial.

---

# 56. What to Look for in a Profiler

Explain:

- expensive operators
- many tiny operations
- unexpected CPU time
- repeated copies
- data-transfer operations
- large memory operations

Keep concise.

---

# 57. Python Loop vs Vectorization

Mark:

```text
🔥 Interview Essential
```

Bad:

```python
result = torch.empty_like(x)

for i in range(len(x)):
    result[i] = x[i] * 2
```

Better:

```python
result = x * 2
```

Explain:

- fewer Python operations
- optimized PyTorch kernels
- better GPU utilization

Create at least 8 exercises converting loops into tensor operations.

---

# 58. Repeated Concatenation Anti-Pattern

Show bad:

```python
result = torch.empty(
    0,
    10
)

for batch in batches:

    result = torch.cat(
        [result, batch],
        dim=0
    )
```

Explain repeated reallocation/copying can be expensive.

Better:

```python
chunks = []

for batch in batches:
    chunks.append(batch)

result = torch.cat(
    chunks,
    dim=0
)
```

For inference outputs, combine with:

```python
batch.detach().cpu()
```

when appropriate.

---

# 59. Creating Tensors Inside Hot Loops

Show:

```python
for X, y in loader:
    scale = torch.tensor(
        0.5,
        device=device
    )
```

If constant, create once outside.

Explain this is a small example of avoiding repeated setup/allocation in hot loops.

---

# 60. Calling `.to(device)` Repeatedly on Model

Bad:

```python
for X, y in loader:
    model.to(device)
```

Better:

```python
model.to(device)

for X, y in loader:
    ...
```

Explain model transfer should normally occur once.

---

# 61. Repeated CPU-GPU Transfers

Show:

```python
X = X.cpu()
X = X.cuda()
```

inside a critical path.

Ask learners to remove unnecessary transfers.

---

# 62. Inference Performance

Teach:

```python
model.eval()

with torch.inference_mode():
    ...
```

Possible additions when supported:

- mixed precision
- appropriate batch size
- avoid CPU/GPU round trips

Do not introduce deployment frameworks yet.

---

# 63. Overfitting vs Bug

Teach important distinction.

Example:

```text
Train loss ↓ strongly
Train accuracy ↑

Val loss ↑
Val accuracy ↓
```

Likely:

```text
overfitting
```

not necessarily a code bug.

Whereas:

```text
Train loss flat from first step
parameters never change
```

likely indicates implementation/optimization problem.

---

# 64. Underfitting vs Training Bug

If:

```text
train loss high
validation loss high
```

could be:

- insufficient model capacity
- poor optimization
- bad features
- insufficient training

But before changing architecture, verify:

- gradients exist
- parameters update
- loss/targets correct

Teach systematic ordering.

---

# 65. Learning Rate Too High

Create synthetic training example demonstrating unstable loss with a deliberately high but safe LR.

Symptoms:

- loss oscillates
- loss may increase
- parameters may become very large
- potentially NaN

Ask learner to reduce LR and compare.

---

# 66. Learning Rate Too Low

Demonstrate:

```text
loss decreases
but extremely slowly
```

Explain possible cause.

Make clear slow learning can also come from many other factors.

---

# 67. Gradient Norm Tracking

Create:

```python
def global_grad_norm(model):

    norms = []

    for p in model.parameters():

        if p.grad is not None:
            norms.append(
                p.grad.detach().norm()
            )

    if not norms:
        return 0.0

    return torch.stack(norms).norm().item()
```

Use across several training steps.

Do not claim one ideal gradient norm.

---

# 68. Parameter Norm Tracking

Create optional helper:

```python
def global_parameter_norm(model):
    ...
```

Explain comparing:

```text
parameter norm
gradient norm
update magnitude
```

can aid deeper debugging.

Keep intermediate.

---

# 69. Verify Gradients Are Finite

Teach:

```python
for name, param in model.named_parameters():

    if param.grad is not None:

        assert torch.isfinite(
            param.grad
        ).all(), (
            f"Non-finite gradient: {name}"
        )
```

Useful temporary debugging assertion.

---

# 70. Verify Parameters Are Finite

After optimizer step:

```python
for name, param in model.named_parameters():

    assert torch.isfinite(
        param
    ).all(), (
        f"Non-finite parameter: {name}"
    )
```

Explain such assertions are useful during debugging, not necessarily kept in high-performance production loops.

---

# 71. Sanity Check Labels

For multiclass classification:

```python
print(y.min())
print(y.max())
print(y.dtype)
```

Check:

```text
0 <= labels < num_classes
```

Create a broken example with label equal to `num_classes`.

Ask learner to diagnose the out-of-range target.

---

# 72. Sanity Check Class Distribution

Teach:

```python
values, counts = torch.unique(
    y,
    return_counts=True
)

print(values)
print(counts)
```

Explain:

> If one class dominates, accuracy alone may be misleading.

Do not turn this into a full metrics notebook.

---

# 73. Check Input Distribution

Use:

```python
print(X.mean())
print(X.std())
print(X.min())
print(X.max())
```

Explain this helps catch:

- scaling bugs
- all-zero features
- extremely large values
- invalid preprocessing

---

# 74. All-Zero Input Bug

Create:

```python
X = torch.zeros(128, 10)
```

Ask why model may struggle to learn meaningful feature relationships.

Use as a data sanity-check scenario.

---

# 75. Label Leakage Awareness

Briefly mention:

> Extremely suspiciously high validation performance can sometimes indicate data leakage rather than an excellent model.

Examples:

- target accidentally included as feature
- train/validation overlap
- preprocessing fitted using all data

Mark:

```text
🔥 ML Engineering Interview Awareness
```

Do not deeply cover ML methodology here.

---

# 76. Training/Validation Leakage Scenario

Give:

```python
train_dataset = full_dataset
val_dataset = full_dataset
```

Ask what is wrong.

Expected:

> Training and validation sets are identical, so validation does not measure generalization.

---

# 77. Debugging Scenario Format

Create at least 25 standalone debugging scenarios.

Each should include:

```text
Symptoms
Code
Question
Hint
Answer
Why
```

Use collapsible answers.

The learner should attempt diagnosis before revealing.

---

# 78. Scenario — Loss Is Constant

Example:

```python
for X, y in loader:

    optimizer.zero_grad()

    logits = model(X)

    loss = loss_fn(logits, y)

    optimizer.step()
```

Ask what is missing.

Expected:

```python
loss.backward()
```

---

# 79. Scenario — No Parameter Update

Example:

```python
loss.backward()

optimizer.zero_grad()

optimizer.step()
```

Expected:

> Gradients are erased before the optimizer uses them.

---

# 80. Scenario — Gradients Grow Every Batch

Code missing:

```python
optimizer.zero_grad()
```

Expected:

> Unintentional gradient accumulation.

---

# 81. Scenario — Validation Changes Model

Validation contains:

```python
optimizer.step()
```

Ask why this invalidates evaluation.

---

# 82. Scenario — Validation Results Random Every Call

Model contains Dropout and remains:

```python
model.train()
```

Ask likely cause.

---

# 83. Scenario — GPU Memory Increases Every Epoch

Code:

```python
epoch_outputs.append(output)
```

while outputs remain graph-connected GPU tensors.

Ask fix.

---

# 84. Scenario — CUDA OOM Only During Validation

Possible code:

```python
model.eval()

for X, y in val_loader:
    output = model(X)
```

without no-grad, while storing outputs.

Ask for likely fixes:

```python
with torch.inference_mode():
```

and avoid storing GPU graphs.

---

# 85. Scenario — Low GPU Utilization

Code contains expensive Python preprocessing inside each loop before transfer.

Ask whether the GPU or input pipeline is likely bottlenecking.

---

# 86. Scenario — Loss Becomes NaN After 10 Steps

Give:

- LR = 10
- very large gradients

Ask first suspects.

Expected:

- learning rate too high
- exploding gradients
- inspect finite values

Do not assume LR alone is definitely responsible.

---

# 87. Scenario — `grad=None`

Show:

```python
features = backbone(X).detach()

logits = classifier(features)
```

Then expect backbone gradients.

Ask why backbone `.grad` is absent.

Expected:

> `.detach()` breaks gradient flow to the backbone.

---

# 88. Scenario — Frozen Backbone Not Training

Show:

```python
for p in backbone.parameters():
    p.requires_grad = False
```

Then user expects fine-tuning.

Ask what to change.

---

# 89. Scenario — Inference Is Very Slow

Code:

```python
for sample in dataset:
    output = model(
        sample.unsqueeze(0)
    )
```

Ask improvement.

Expected possibilities:

- batch inference
- `inference_mode`
- move model/data once
- mixed precision where appropriate

---

# 90. Interview Questions

Create approximately 40–45 interview questions.

Include:

1. What is your first step when debugging a PyTorch shape error?
2. What tensor properties do you inspect first?
3. How do you debug a device mismatch?
4. How do you debug a dtype mismatch?
5. What would you check if loss is not decreasing?
6. Why is overfitting a tiny dataset a useful debugging test?
7. How do you verify parameters are actually updating?
8. How do you check whether a parameter received a gradient?
9. What does `param.grad is None` mean?
10. What is a gradient norm?
11. What are vanishing gradients?
12. What are exploding gradients?
13. How can gradient clipping help?
14. What commonly causes NaN loss?
15. How do you detect NaN / Inf values in PyTorch?
16. What does `torch.isfinite()` do?
17. What is Autograd anomaly detection?
18. When would you enable anomaly detection?
19. Why can in-place operations break Autograd?
20. What does the underscore suffix mean in PyTorch methods?
21. How can forgetting `model.eval()` affect validation?
22. How can forgetting `torch.no_grad()` affect validation?
23. How can saving loss tensors create a memory leak?
24. Why can `retain_graph=True` increase memory usage?
25. What would you check after CUDA OOM?
26. What is the difference between allocated and reserved CUDA memory?
27. Does `empty_cache()` fix memory leaks?
28. What can cause low GPU utilization?
29. How can a slow DataLoader affect GPU utilization?
30. How would you separate data-loading time from model-compute time?
31. Why can Python loops be slow in PyTorch?
32. Why is vectorization usually better?
33. Why can repeated concatenation be inefficient?
34. Why can `.item()` affect GPU performance?
35. Why should CUDA timing use synchronization?
36. What is a warmup run?
37. What does `torch.profiler` help identify?
38. What is the tiny-batch overfit test?
39. How would you diagnose an incorrect loss function?
40. How do you verify multiclass target values are valid?
41. What is label leakage?
42. How would you distinguish overfitting from a broken training loop?
43. What symptoms suggest learning rate is too high?
44. What symptoms suggest learning rate may be too low?
45. How would you debug GPU memory increasing every iteration?

For each provide:

### Short Interview Answer

A concise answer suitable for speaking.

### Detailed Explanation

A slightly deeper answer.

Use collapsible sections.

---

# 91. Knowledge Check Quiz

Create approximately 30 multiple-choice questions.

Example:

```text
Loss remains exactly constant and parameters never change.

Which should you inspect first?

A. Change model architecture immediately
B. Check gradients and optimizer step
C. Increase DataLoader workers
D. Enable Dropout
```

Correct:

```text
B
```

Another:

```text
What does torch.isfinite(x) test?

A. Whether x is on GPU
B. Whether values are neither NaN nor ±Inf
C. Whether x requires gradients
D. Whether x is contiguous
```

Correct:

```text
B
```

Another:

```text
Why can losses.append(loss) cause GPU memory growth during training?

A. It changes batch size
B. It can keep computation graphs alive
C. It disables gradients
D. It changes dtype
```

Correct:

```text
B
```

Put quiz answers in a separate answer section.

---

# 92. Write From Memory

Create approximately 20 prompts.

Examples:

> Print tensor shape, dtype, and device.

> Check whether all values in `x` are finite.

> Check whether `x` contains any NaN values.

> Print every model parameter's gradient norm.

> Enable Autograd anomaly detection.

> Clip gradient norm to 1.0.

> Check model parameter device.

> Check whether parameters changed after one optimizer step.

> Measure CPU execution time.

> Synchronize CUDA before and after timing.

> Print allocated CUDA memory safely.

> Convert a saved prediction to detached CPU form.

> Replace a Python element-wise loop with vectorized PyTorch.

> Create a tiny-dataset overfitting test.

Include executable validation where possible.

---

# 93. Debugging Flashcards

Create rapid recall prompts such as:

```text
Symptom:
Expected all tensors to be on the same device

First check?
```

Expected:

```text
model device, input device, target device
```

---

```text
Symptom:
CrossEntropyLoss complains about target dtype.

Likely expected dtype?
```

Expected:

```text
torch.long
```

---

```text
Symptom:
GPU memory grows every step.

What code pattern should you search for?
```

Expected ideas:

```text
storing graph-connected tensors
retain_graph=True
large GPU outputs kept in lists
```

Create approximately 15 flashcards.

---

# 94. Final Debugging Challenge

Create one intentionally broken training pipeline containing approximately **10 bugs**.

Use a synthetic multiclass dataset.

Example components:

```python
X = torch.randn(1000, 20)
y = ...
```

Model:

```text
20 → 64 → ReLU → Dropout → 4 logits
```

Training and validation loops should intentionally contain bugs.

The learner's task is to diagnose and fix all of them.

---

# 95. Possible Final Challenge Bugs

Include bugs such as:

1. Labels accidentally converted to float for `CrossEntropyLoss`.
2. Model output has wrong number of classes.
3. Training loop does not call `model.train()`.
4. Gradients are cleared after backward but before step.
5. Learning rate is extremely high.
6. Validation forgets `model.eval()`.
7. Validation runs with gradient tracking.
8. Training loss tensor is appended directly to a history list.
9. Accuracy uses `argmax(dim=0)` instead of `dim=1`.
10. Data is not moved to model device.
11. Optimizer may be attached to wrong parameters.
12. A layer may be unintentionally frozen.

Use approximately 10, not necessarily all 12.

---

# 96. Final Challenge Workflow

Ask learner to:

### Step 1

Run diagnostic inspection.

Print:

```text
X shape
y shape
y dtype
model output shape
model device
X device
target min/max
```

### Step 2

Run one batch only.

Verify:

```text
loss finite
gradients present
gradients finite
parameters change
```

### Step 3

Overfit a tiny subset.

For example:

```text
32 samples
```

Require training accuracy to become very high.

### Step 4

Restore full train/validation training.

### Step 5

Confirm:

```text
training loss generally decreases
validation behaves deterministically in eval mode
GPU memory does not grow continuously from graph retention
```

The goal is not exact final accuracy but proving the pipeline is mechanically correct.

---

# 97. Final Challenge Validation Helpers

Create functions such as:

```python
def assert_finite_tensor(name, x):
    assert torch.isfinite(x).all(), (
        f"{name} contains non-finite values"
    )
```

and:

```python
def assert_gradients_finite(model):

    for name, param in model.named_parameters():

        if param.grad is not None:

            assert torch.isfinite(
                param.grad
            ).all(), (
                f"Non-finite gradient: {name}"
            )
```

Use them in challenge validation.

---

# 98. Debugging Cheat Sheet

End with a concise, high-value cheat sheet.

Include:

```python
# Inspect tensor
print(x.shape)
print(x.dtype)
print(x.device)
print(x.requires_grad)

# Check model device
print(
    next(model.parameters()).device
)

# Check finite values
torch.isnan(x).any()
torch.isinf(x).any()
torch.isfinite(x).all()

# Check target classes
print(y.dtype)
print(y.min())
print(y.max())

# Check gradients
for name, param in model.named_parameters():

    if param.grad is None:
        print(name, "NO GRAD")
    else:
        print(
            name,
            param.grad.norm().item()
        )

# Clip gradients
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

# Anomaly detection
torch.autograd.set_detect_anomaly(
    True
)

# Validation
model.eval()

with torch.inference_mode():
    ...

# CUDA memory
if torch.cuda.is_available():

    print(
        torch.cuda.memory_allocated()
    )

    print(
        torch.cuda.memory_reserved()
    )

# Timing CUDA
if torch.cuda.is_available():

    torch.cuda.synchronize()

    start = time.perf_counter()

    # operation

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )
```

---

# 99. High-Level Debugging Checklist

End with this quick-reference sequence:

```text
Training is broken?

1. Can one batch run?
2. Are shapes correct?
3. Are dtypes correct?
4. Are devices correct?
5. Are targets valid?
6. Is loss finite?
7. Does backward create gradients?
8. Are gradients finite?
9. Does optimizer.step change parameters?
10. Can model overfit 32 samples?
11. Is train/eval mode correct?
12. Is validation really no-grad?
13. Is memory growing?
14. Is DataLoader too slow?
15. Only then tune architecture/hyperparameters.
```

Mark:

```text
🔥 Memorize This Workflow
```

---

# 100. Performance Quick Reference

Add:

```text
GPU slow?
↓
Check DataLoader
↓
Check batch size
↓
Check CPU-GPU transfers
↓
Check Python loops
↓
Check synchronization
↓
Profile
```

And:

```text
GPU memory growing?
↓
Check stored loss/output tensors
↓
Check retain_graph
↓
Check unnecessary references
↓
Inspect allocated memory
```

---

# 101. Completion Checklist

End with:

```text
## Before Moving to 10_transformer_pytorch.ipynb
```

Add:

- [ ] I have a systematic PyTorch debugging workflow.
- [ ] I inspect shape, dtype, and device first.
- [ ] I can diagnose Linear/CNN shape mismatches.
- [ ] I can diagnose target dtype problems.
- [ ] I can diagnose loss/activation mismatches.
- [ ] I know what to check when loss does not decrease.
- [ ] I can verify that parameters actually update.
- [ ] I can inspect gradients.
- [ ] I understand `grad=None`.
- [ ] I understand exploding and vanishing gradients.
- [ ] I can detect NaN and Inf values.
- [ ] I know common causes of NaN loss.
- [ ] I understand Autograd anomaly detection.
- [ ] I understand in-place Autograd errors.
- [ ] I can debug train/eval mode mistakes.
- [ ] I can identify graph-retention memory leaks.
- [ ] I can investigate CUDA OOM.
- [ ] I can diagnose a DataLoader bottleneck.
- [ ] I understand causes of low GPU utilization.
- [ ] I can time CPU and CUDA code correctly.
- [ ] I understand basic `torch.profiler` usage.
- [ ] I understand why vectorization matters.
- [ ] I can use the tiny-dataset overfitting test.
- [ ] I can debug a broken training pipeline methodically.

---

# 102. Difficulty Labels

Use:

```text
🟢 Easy
🟡 Medium
🔴 Hard
```

Easy:

- shape debugging
- dtype debugging
- device debugging
- finite-value checking
- basic train/eval issues

Medium:

- gradient inspection
- NaN diagnosis
- memory leak diagnosis
- profiling
- learning-rate debugging
- DataLoader bottlenecks

Hard / optional:

- deeper profiler analysis
- gradient distribution analysis
- advanced CUDA-memory diagnosis

Do not yet deeply cover:

- DDP debugging
- FSDP
- NCCL
- multi-node issues
- distributed deadlocks
- `torch.compile` graph breaks
- custom CUDA kernels

Those are out of scope.

---

# 103. Quality Requirements

The notebook must:

- be a valid `.ipynb`
- execute successfully from top to bottom
- work on CPU-only systems
- conditionally use CUDA when available
- require no internet access
- use only PyTorch and Python standard library
- avoid deliberately crashing the whole notebook
- use `try/except` or non-executed Markdown snippets for expected-error examples
- contain automatic `assert` validation
- contain approximately 80–100 exercises/questions
- contain at least 25 realistic debugging scenarios
- heavily emphasize diagnosis rather than memorization
- heavily emphasize real AI Engineer interview situations
- use small datasets and models so it runs quickly
- avoid introducing unnecessary third-party profiling tools

The notebook should take approximately **2.5–3.5 hours** to study thoroughly.

The most important learning loop is:

```text
See symptom
    ↓
form hypothesis
    ↓
inspect evidence
    ↓
identify bug
    ↓
fix one thing
    ↓
rerun
    ↓
verify behavior
```

Do not teach debugging as:

```text
"Try random changes until it works."
```

By the end of this notebook, the learner should be able to answer an interview question such as:

> "Your PyTorch model's loss isn't decreasing. How would you debug it?"

with a structured, technically sound process.

Finally save the completed notebook as:

```text
09_debugging_and_performance.ipynb
```