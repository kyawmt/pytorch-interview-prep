Create a Jupyter Notebook named:

```text
08_gpu_and_devices.ipynb
```

This notebook is the eighth part of a **PyTorch Interview Preparation** course for someone preparing for **AI Engineer / Machine Learning Engineer interviews**.

Assume the learner has already completed:

```text
01_tensor_basics.ipynb
02_tensor_operations.ipynb
03_autograd.ipynb
04_neural_networks.ipynb
05_losses_optimizers.ipynb
06_training_loop.ipynb
07_dataset_dataloader.ipynb
```

and already understands:

- tensors
- models
- Autograd
- losses and optimizers
- training loops
- Dataset and DataLoader
- basic `.to(device)` usage

Do not spend significant time reteaching those topics.

The purpose of this notebook is to teach **PyTorch device handling, GPU usage, memory awareness, efficient data transfer, mixed precision basics, and practical debugging of CPU/GPU issues**.

The notebook must work correctly even when the learner has **no CUDA GPU**.

If CUDA is unavailable, GPU-specific exercises should still teach the syntax and concepts without failing.

Also support Apple Silicon MPS where appropriate.

---

# Main Learning Goals

By the end of this notebook, the learner should confidently understand and use:

```python
torch.cuda.is_available()

torch.backends.mps.is_available()

torch.device(...)

tensor.device

tensor.to(device)

model.to(device)

tensor.cpu()

tensor.cuda()
```

and understand practical concepts such as:

- CPU vs GPU
- CUDA
- Apple MPS
- device placement
- device mismatch
- moving models and tensors
- why model and input must be on the same device
- moving outputs back to CPU
- CPU ↔ GPU transfer cost
- GPU memory
- allocated vs reserved CUDA memory
- clearing references vs `empty_cache`
- `pin_memory`
- `non_blocking=True`
- batch size and GPU memory
- out-of-memory errors
- inference memory
- `torch.no_grad()`
- mixed precision
- float32 / float16 / bfloat16
- autocast
- gradient scaling
- basic performance measurement
- avoiding unnecessary synchronization
- common GPU debugging techniques

Mark these as:

```text
🔥 Interview Essential
```

- device selection
- `.to(device)`
- model/input device consistency
- CUDA availability
- CPU ↔ GPU transfer cost
- GPU OOM
- batch size
- `torch.no_grad()`
- mixed precision
- `float16` vs `bfloat16`
- `pin_memory`
- `non_blocking`
- inference vs training memory usage

---

# Notebook Structure

Use approximately:

```text
# PyTorch GPU and Device Handling

## 1. Setup
## 2. CPU vs GPU
## 3. CUDA and MPS
## 4. Selecting a Device
## 5. Inspecting Tensor Devices
## 6. Moving Tensors Between Devices
## 7. Moving Models to a Device
## 8. Device Mismatch Errors
## 9. Device-Agnostic Code
## 10. Moving Data Back to CPU
## 11. CPU-GPU Transfer Cost
## 12. DataLoader and GPU Transfer
## 13. pin_memory
## 14. non_blocking Transfers
## 15. GPU Memory Basics
## 16. CUDA Memory Inspection
## 17. Out-of-Memory Errors
## 18. Batch Size and Memory
## 19. Training vs Inference Memory
## 20. torch.no_grad and inference
## 21. Floating-Point Precision
## 22. float16
## 23. bfloat16
## 24. Mixed Precision
## 25. autocast
## 26. Gradient Scaling
## 27. Mixed Precision Training Pattern
## 28. Performance Timing
## 29. Synchronization
## 30. Common Performance Mistakes
## 31. Common Device Mistakes
## 32. Debugging Exercises
## 33. Interview Questions
## 34. Knowledge Check Quiz
## 35. Write From Memory
## 36. Final GPU-Aware Training Challenge
## 37. Cheat Sheet
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

Print:

```python
print("PyTorch version:", torch.__version__)
```

Then safely inspect available backends.

```python
print("CUDA available:", torch.cuda.is_available())

mps_available = (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
)

print("MPS available:", mps_available)
```

The notebook must never assume CUDA exists.

---

# 2. CPU vs GPU

Explain simply:

```text
CPU
→ general-purpose processor
→ fewer powerful cores
→ good for control flow and general computation

GPU
→ many parallel compute units
→ good for large tensor operations
→ especially matrix multiplication and neural networks
```

Explain why deep learning often benefits from GPUs:

```text
Large tensor operations
+
massive parallelism
=
GPU-friendly workload
```

Do not go deeply into GPU architecture yet.

---

# 3. What Is CUDA?

Explain:

> CUDA is NVIDIA's platform for GPU computing. PyTorch can use CUDA-enabled NVIDIA GPUs when a compatible PyTorch installation and GPU driver are available.

Important distinction:

```text
GPU
≠ automatically CUDA

CUDA
→ NVIDIA-specific GPU computing platform
```

Mention that Apple Silicon uses:

```text
MPS
```

rather than CUDA.

---

# 4. Selecting a Device

Mark:

```text
🔥 Interview Essential
```

Teach the portable pattern:

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

Explain:

```text
NVIDIA GPU available
→ cuda

Apple Silicon GPU available
→ mps

otherwise
→ cpu
```

Also teach the simpler common pattern:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Explain that this simpler form does not select Apple MPS.

---

# 5. `torch.device`

Teach:

```python
device = torch.device("cpu")
```

and conditionally:

```python
device = torch.device("cuda")
```

Explain:

> `torch.device` represents where tensors and model parameters should live.

Show:

```python
print(device.type)
```

---

# 6. Inspecting Tensor Device

Teach:

```python
x = torch.randn(3, 4)

print(x.device)
```

Expected on normal creation:

```text
cpu
```

Explain:

> Tensors are normally created on CPU unless another device is explicitly specified.

---

# 7. Creating a Tensor Directly on a Device

Teach:

```python
x = torch.randn(
    3,
    4,
    device=device
)
```

Show:

```python
print(x.device)
```

Explain two common approaches:

```text
Create on CPU
→ move later with .to(device)

or

Create directly on device
```

---

# 8. Moving a Tensor

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
x = x.to(device)
```

Important subtlety:

Explain that:

```python
x.to(device)
```

returns a tensor on the requested device.

Therefore use:

```python
x = x.to(device)
```

rather than assuming `x.to(device)` always modifies `x` in place.

Create an exercise specifically testing this.

---

# 9. Exercise — Move Tensor

Starter:

```python
x = torch.randn(5, 10)

# Move x to the selected device.
answer = None
```

Validation:

```python
assert isinstance(answer, torch.Tensor)
assert answer.device.type == device.type

print("✅ Correct!")
```

---

# 10. Moving a Model

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
model = nn.Linear(10, 3)

model = model.to(device)
```

Explain:

> `model.to(device)` moves model parameters and registered buffers to the selected device.

Show:

```python
for param in model.parameters():
    print(param.device)
```

Explain that the model is usually moved once before training:

```python
model.to(device)

for epoch in ...:
    for X, y in loader:
        ...
```

not repeatedly inside every batch.

---

# 11. Device Mismatch

Make this a major section.

Mark:

```text
🔥 Very Common Interview / Debugging Issue
```

Explain this situation:

```text
model
→ cuda

X
→ cpu
```

Then:

```python
model(X)
```

fails.

Explain:

> The model parameters and input tensors participating in an operation generally need to be on the same device.

Show safe example conditionally rather than deliberately crashing the notebook.

Use conceptual error text such as:

```text
Expected all tensors to be on the same device
```

Ask learners to diagnose it.

---

# 12. Correct Training Pattern

Teach:

```python
model = model.to(device)

for X, y in train_loader:

    X = X.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    logits = model(X)

    loss = loss_fn(logits, y)

    loss.backward()

    optimizer.step()
```

Mark:

```text
🔥 Interview Essential
```

---

# 13. Why Move the Target?

Explain that target placement depends on the operation.

For common GPU training:

```python
logits = model(X)
loss = loss_fn(logits, y)
```

if logits are on GPU, the target should generally also be on the GPU.

Thus:

```python
y = y.to(device)
```

is normally needed.

---

# 14. Device-Agnostic Functions

Show a reusable training function:

```python
def train_step(
    model,
    X,
    y,
    loss_fn,
    optimizer,
    device
):

    model.train()

    X = X.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    logits = model(X)
    loss = loss_fn(logits, y)

    loss.backward()
    optimizer.step()

    return loss.item()
```

Explain:

> Passing `device` explicitly makes the function portable between CPU and GPU.

---

# 15. `.cpu()`

Teach:

```python
x_cpu = x.cpu()
```

Explain:

> `.cpu()` moves a tensor to CPU memory.

Common use:

```python
predictions = predictions.cpu()
```

before:

- NumPy conversion
- CPU-only libraries
- some logging / serialization workflows

---

# 16. `.cuda()`

Teach:

```python
x.cuda()
```

but explain:

> `.cuda()` is NVIDIA-specific. `.to(device)` is generally more portable because it can support CPU, CUDA, and MPS.

Preferred course style:

```python
x = x.to(device)
```

rather than hardcoding:

```python
x = x.cuda()
```

---

# 17. NumPy Conversion

Connect to earlier material.

Teach:

```python
x.detach().cpu().numpy()
```

Explain each:

```text
detach()
→ remove Autograd history

cpu()
→ ensure CPU memory

numpy()
→ convert to ndarray
```

Mark:

```text
🔥 Interview Useful
```

---

# 18. CPU ↔ GPU Transfer Cost

Make this important.

Explain:

> Moving tensors between CPU and GPU is not free.

Visualize:

```text
CPU memory
    ↓ transfer
GPU memory
    ↓ computation
GPU memory
    ↓ transfer
CPU memory
```

Explain that excessive movement can make code slower even if GPU computation itself is fast.

Bad:

```python
for layer in layers:
    x = x.cpu()
    x = x.to(device)
```

Better:

```text
move batch to GPU
↓
do as much computation as possible
↓
move results back only when needed
```

---

# 19. Avoid Repeated Transfers

Show bad pattern:

```python
for X, y in loader:

    X = X.to(device)

    output = model(X)

    output = output.cpu()

    output = output.to(device)

    ...
```

Ask what is inefficient.

Expected:

> Unnecessary CPU/GPU transfers.

---

# 20. DataLoader and Device

Explain:

> A normal DataLoader usually yields CPU tensors.

Typical pipeline:

```text
Dataset
↓
DataLoader
↓
CPU batch
↓
.to(device)
↓
GPU model
```

Show:

```python
for X, y in train_loader:

    X = X.to(device)
    y = y.to(device)

    ...
```

---

# 21. `pin_memory`

Mark:

```text
🔥 Interview Useful
```

Teach:

```python
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=True
)
```

Explain:

> Pinned CPU memory can improve CPU-to-CUDA transfer performance.

Important caveat:

> It is primarily relevant for CUDA workflows and does not automatically move data to the GPU.

---

# 22. `non_blocking=True`

Teach:

```python
X = X.to(
    device,
    non_blocking=True
)
```

Explain:

> With suitable pinned-memory CUDA transfers, `non_blocking=True` can allow data transfer to overlap with other work.

Do not imply that simply adding `non_blocking=True` always improves performance.

Mark as intermediate.

---

# 23. Typical CUDA Data Transfer Pattern

Show:

```python
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
)
```

Then:

```python
for X, y in train_loader:

    X = X.to(
        device,
        non_blocking=torch.cuda.is_available()
    )

    y = y.to(
        device,
        non_blocking=torch.cuda.is_available()
    )
```

This should run on CPU as well.

---

# 24. GPU Memory Basics

Mark:

```text
🔥 Interview Essential
```

Explain GPU memory contains things such as:

```text
model parameters
+
activations
+
gradients
+
optimizer state
+
temporary tensors
+
input batches
```

Show conceptual breakdown:

```text
Training GPU memory

Parameters
Gradients
Optimizer state
Activations
Batch
Temporary buffers
```

Explain why training usually consumes more memory than inference.

---

# 25. Parameters vs Activations

Explain:

> Model size is not the only source of GPU-memory usage.

For large batches or deep networks:

```text
activations
```

can consume a large amount of memory.

Connect:

```text
larger batch size
→ more activations
→ more GPU memory
```

---

# 26. Training vs Inference Memory

Mark:

```text
🔥 Interview Essential
```

Explain:

### Training

Needs:

- model parameters
- activations for backward
- gradients
- optimizer state

### Inference

Usually needs:

- model parameters
- forward activations
- no gradients
- no optimizer state

Therefore inference usually needs less memory.

---

# 27. Why `torch.no_grad()` Reduces Memory

Teach:

```python
model.eval()

with torch.no_grad():
    output = model(X)
```

Explain:

> Without Autograd graph tracking, PyTorch does not need to store many intermediate values for backward computation.

This reduces:

- memory use
- computation overhead

---

# 28. `torch.inference_mode()`

Introduce:

```python
with torch.inference_mode():
    output = model(X)
```

Explain:

> `inference_mode()` is an even stronger inference-oriented mode that can reduce additional Autograd overhead.

Compare:

```text
no_grad()
→ disables gradient calculation

inference_mode()
→ optimized specifically for inference
```

Keep this intermediate and practical.

Do not claim they are interchangeable in every edge case.

---

# 29. CUDA Memory Inspection

Only execute these cells if CUDA is available.

Teach:

```python
if torch.cuda.is_available():

    print(
        torch.cuda.memory_allocated()
    )

    print(
        torch.cuda.memory_reserved()
    )
```

Explain:

```text
memory_allocated
→ memory currently used by live tensors

memory_reserved
→ memory held by PyTorch's CUDA allocator
```

Also show:

```python
torch.cuda.max_memory_allocated()
```

when useful.

---

# 30. Convert Bytes to MB

Create helper:

```python
def bytes_to_mb(x):
    return x / 1024**2
```

Then:

```python
if torch.cuda.is_available():

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    print(
        "Allocated MB:",
        bytes_to_mb(allocated)
    )

    print(
        "Reserved MB:",
        bytes_to_mb(reserved)
    )
```

---

# 31. `torch.cuda.empty_cache()`

Teach carefully:

```python
torch.cuda.empty_cache()
```

Explain:

> `empty_cache()` releases unused cached CUDA memory held by PyTorch back to the CUDA allocator/other applications.

Important:

> It does not magically free memory still referenced by live tensors.

Example:

```python
x = huge_tensor
```

If `x` still exists, `empty_cache()` does not delete `x`.

Typical sequence:

```python
del x

if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

Mark:

```text
🔥 Common Misconception
```

---

# 32. Python References and GPU Memory

Explain:

```python
x = torch.randn(..., device="cuda")
```

GPU memory remains in use while `x` is still referenced.

Possible release:

```python
del x
```

after which memory becomes eligible for reuse.

Do not teach manual memory management as something normally required for every tensor.

---

# 33. CUDA Out-of-Memory Error

Mark:

```text
🔥 Interview Essential
```

Explain typical error:

```text
CUDA out of memory
```

Common causes:

- batch too large
- model too large
- activations too large
- accidental retained computation graphs
- saving GPU tensors every iteration
- forgotten gradients
- multiple models in memory

Create a debugging checklist.

---

# 34. First Response to OOM

Teach practical approaches:

1. Reduce batch size.
2. Use mixed precision.
3. Use `torch.no_grad()` / `inference_mode()` for inference.
4. Avoid storing unnecessary GPU tensors.
5. Delete unused large objects.
6. Use gradient accumulation if a large effective batch is required.
7. Consider gradient checkpointing for large models.
8. Reduce sequence/image size if appropriate.
9. Check for memory leaks.

Do not make advanced distributed techniques central.

---

# 35. Memory Leak Pattern

Show bad code:

```python
loss_history = []

for X, y in loader:

    logits = model(X)

    loss = loss_fn(logits, y)

    loss_history.append(loss)
```

Explain:

> Saving the loss tensor itself may keep its computation graph alive.

Preferred:

```python
loss_history.append(
    loss.item()
)
```

or:

```python
loss_history.append(
    loss.detach().cpu()
)
```

depending on purpose.

Mark:

```text
🔥 Important Debugging Pattern
```

---

# 36. Another Graph-Retention Mistake

Show:

```python
all_outputs = []

for X in loader:
    output = model(X)
    all_outputs.append(output)
```

During training this can retain large GPU tensors/graphs.

Possible solution when gradients are not needed for saved outputs:

```python
all_outputs.append(
    output.detach().cpu()
)
```

Explain context matters.

---

# 37. Batch Size and Memory

Create a conceptual relationship:

```text
larger batch
→ more samples processed simultaneously
→ more activations
→ more GPU memory
```

Ask:

> If batch size 128 causes OOM, what is the first easy thing to try?

Expected:

```text
reduce batch size
```

---

# 38. Sequence Length and Memory

Because this is AI Engineer preparation, mention:

```text
longer sequence
→ more activation memory
```

For transformers, attention can be particularly expensive as sequence length grows.

Do not dive deeply into attention complexity yet.

Mark:

```text
🔥 Interview Useful for LLM roles
```

---

# 39. Precision Basics

Teach:

```python
torch.float32
torch.float16
torch.bfloat16
```

Create comparison:

| dtype | Bits | Memory vs FP32 | General Note |
|---|---:|---:|---|
| float32 | 32 | 1× | standard precision |
| float16 | 16 | ~0.5× | faster/smaller but narrower numeric range |
| bfloat16 | 16 | ~0.5× | larger exponent range, popular in modern AI |

Explain that actual speed gains depend on hardware.

---

# 40. Float32

Explain:

> FP32 is a common default floating-point dtype in PyTorch.

Show:

```python
x = torch.randn(10)

print(x.dtype)
```

Typically:

```text
torch.float32
```

---

# 41. Float16

Teach:

```python
x_half = x.to(torch.float16)
```

Explain benefits:

- lower memory
- can increase speed on appropriate GPUs

Potential issue:

- smaller numerical range
- underflow / overflow risk

---

# 42. bfloat16

Teach:

```python
x_bf16 = x.to(torch.bfloat16)
```

Explain:

> bfloat16 uses fewer precision bits than float32 but keeps an exponent range similar to float32, making it useful for deep-learning workloads.

Mention:

```text
modern transformers
→ bfloat16 is very common
```

when supported by hardware.

---

# 43. float16 vs bfloat16

Mark:

```text
🔥 Interview Essential for modern AI
```

Explain simply:

```text
float16
→ more mantissa precision than bfloat16
→ smaller exponent range

bfloat16
→ less mantissa precision
→ much larger exponent range
→ often more numerically robust for deep learning
```

Avoid overly detailed IEEE-754 discussion.

---

# 44. Why Not Convert Everything Manually to float16?

Explain:

> Some operations are less numerically stable in low precision.

Therefore modern training often uses:

```text
mixed precision
```

rather than converting every operation blindly.

---

# 45. Mixed Precision

Mark:

```text
🔥 Interview Essential
```

Explain:

> Mixed precision uses lower precision for operations that are safe and higher precision where needed.

Benefits:

- lower GPU-memory use
- potentially faster training
- larger possible batch size

Potential concern:

- numerical stability

---

# 46. Autocast

Teach the modern pattern conceptually.

For CUDA:

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16
):
    logits = model(X)
    loss = loss_fn(logits, y)
```

For suitable hardware, bfloat16 may be used:

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16
):
    ...
```

Do not execute CUDA-only code unless CUDA exists.

Build a helper so the notebook runs everywhere.

---

# 47. Portable Autocast Example

Create conditional code such as:

```python
if device.type == "cuda":
    autocast_device = "cuda"
elif device.type == "cpu":
    autocast_device = "cpu"
else:
    autocast_device = None
```

But do not force autocast on unsupported combinations.

Simpler notebook policy:

- execute mixed-precision training only when CUDA is available
- otherwise explain the code in Markdown and skip execution gracefully

Example:

```python
if torch.cuda.is_available():
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16
    ):
        ...
else:
    print(
        "CUDA unavailable — showing mixed-precision syntax only."
    )
```

---

# 48. Gradient Scaling

Teach why FP16 can create very small gradients.

Introduce:

```python
torch.amp.GradScaler
```

Use current, modern PyTorch syntax where supported.

Before generating final notebook code, verify which GradScaler API is appropriate for the installed/current PyTorch version used by the environment. Prefer modern non-deprecated APIs and avoid teaching obsolete syntax when possible.

The conceptual pattern is:

```text
forward under autocast
↓
scaled backward
↓
optimizer step through scaler
↓
scaler update
```

Explain:

> Gradient scaling helps prevent small FP16 gradients from underflowing toward zero.

Mark:

```text
🟡 Medium
```

---

# 49. Mixed Precision Training Pattern

Show a clean pattern appropriate for the PyTorch version.

Conceptually:

```python
optimizer.zero_grad()

with autocast_context:
    logits = model(X)
    loss = loss_fn(logits, y)

scaler.scale(loss).backward()

scaler.step(optimizer)

scaler.update()
```

Explain each step.

Do not make mixed precision mandatory for completing the notebook.

---

# 50. bfloat16 and Gradient Scaling

Explain:

> bfloat16 has a much larger exponent range than float16, so gradient scaling is often less necessary than with FP16.

Keep the wording nuanced because exact practice depends on framework, hardware, and training setup.

---

# 51. Mixed Precision Inference

Show conceptually:

```python
model.eval()

with torch.inference_mode():
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16
    ):
        output = model(X)
```

Explain potential benefits:

- less memory
- faster inference on suitable hardware

---

# 52. Measuring Execution Time

Teach basic CPU timing:

```python
start = time.perf_counter()

# operation

elapsed = time.perf_counter() - start

print(elapsed)
```

Then explain GPU timing is different because GPU execution can be asynchronous.

---

# 53. CUDA Synchronization

Mark:

```text
🟡 Interview Useful
```

Teach:

```python
torch.cuda.synchronize()
```

Explain:

> CUDA operations can be queued asynchronously. If you benchmark with normal Python timers, synchronize before reading elapsed time.

Example:

```python
if torch.cuda.is_available():

    torch.cuda.synchronize()

    start = time.perf_counter()

    y = model(x)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(elapsed)
```

Explain not to call synchronize unnecessarily during normal training because it can reduce performance.

---

# 54. CUDA Events

Optionally introduce better GPU timing:

```python
start_event = torch.cuda.Event(
    enable_timing=True
)

end_event = torch.cuda.Event(
    enable_timing=True
)
```

Use only in a short optional section.

Do not make it core material.

---

# 55. Warm-Up Runs

Explain:

> The first GPU execution can include one-time setup overhead.

When benchmarking:

```text
warm up
↓
then measure repeated iterations
```

This is useful for performance interviews.

---

# 56. GPU Utilization Concept

Explain that owning a GPU does not guarantee high utilization.

Possible reasons for low GPU utilization:

- DataLoader too slow
- batch too small
- excessive CPU/GPU transfers
- Python overhead
- frequent synchronization
- model too small
- storage bottleneck

Mention that external profiling tools are used in real systems, but do not require them in this notebook.

---

# 57. CPU-GPU Synchronization Mistake: `.item()`

Explain:

```python
loss.item()
```

on a CUDA tensor can require the CPU to wait for the GPU result.

This is usually fine for occasional logging, but doing synchronization-heavy operations excessively can hurt performance.

Do not tell learners to avoid `.item()` entirely.

Explain:

> Use it when needed for logging, but understand it can synchronize GPU work.

Mark as intermediate.

---

# 58. Another Synchronization Pattern

Mention printing GPU tensors frequently:

```python
print(x)
```

can trigger synchronization/data transfer.

Explain why excessive per-batch printing can slow training.

---

# 59. Common Device Mistakes

Include at least these.

## Mistake 1 — Model on CUDA, input on CPU

## Mistake 2 — Input on CUDA, target on CPU when the loss requires both on same device

## Mistake 3 — Calling `.to(device)` without assigning the result

Example:

```python
X.to(device)

logits = model(X)
```

Explain that `X` may still refer to the original CPU tensor.

Preferred:

```python
X = X.to(device)
```

## Mistake 4 — Moving model inside every batch

## Mistake 5 — Hardcoding `"cuda"` without checking availability

## Mistake 6 — Using `.cuda()` in code intended to support MPS/CPU

## Mistake 7 — Calling `.numpy()` on a CUDA tensor

## Mistake 8 — Calling `.numpy()` on a tensor requiring gradients

## Mistake 9 — Repeated CPU/GPU transfers

## Mistake 10 — Assuming `pin_memory` moves data to GPU

## Mistake 11 — Assuming `empty_cache()` frees live tensors

## Mistake 12 — Saving graph-connected GPU tensors every iteration

## Mistake 13 — Using training Autograd during inference unnecessarily

## Mistake 14 — Increasing batch size until OOM without measuring throughput

## Mistake 15 — Assuming FP16 is always faster

## Mistake 16 — Using low precision without checking hardware support

## Mistake 17 — Benchmarking CUDA without synchronization

---

# 60. OOM Debugging Exercise

Give scenario:

```text
Model trains successfully with batch_size=32.

Changing to batch_size=256 produces:

CUDA out of memory
```

Ask:

1. Why?
2. First thing to try?
3. How could gradient accumulation help?

Expected concepts:

```text
larger batch
→ more activation memory

first try:
reduce batch size

gradient accumulation:
small physical batches
but larger effective batch
```

---

# 61. Device Debugging Exercises

Create at least 20.

Example:

```python
model = model.to(device)

for X, y in loader:
    logits = model(X)
```

Ask what may be wrong when `device="cuda"`.

---

Another:

```python
X.to(device)

logits = model(X)
```

Ask why `X` may still be CPU.

---

Another:

```python
pred = model(X)

array = pred.numpy()
```

Ask why this may fail on GPU.

Expected:

```python
pred.detach().cpu().numpy()
```

as appropriate.

---

Another:

```python
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model.cuda()
```

Ask why the code is not actually device-agnostic.

---

Another:

```python
losses.append(loss)
```

inside a long training loop.

Ask why GPU memory may continuously increase.

---

# 62. Device State Inspection Exercise

Teach a useful debugging pattern:

```python
print("Model device:")
print(next(model.parameters()).device)

print("Input device:")
print(X.device)

print("Target device:")
print(y.device)
```

Ask learner to memorize this as a quick check for device mismatch.

---

# 63. Parameter Device Helper

Optionally define:

```python
def get_model_device(model):
    return next(model.parameters()).device
```

Explain caveat that parameterless modules are possible, but this is useful for normal trainable models.

---

# 64. Memory Estimation Exercise

Teach basic tensor-memory calculation.

Example:

```text
Tensor:
(1000, 1000)
float32

elements:
1,000,000

bytes per element:
4

memory:
4,000,000 bytes
≈ 3.81 MiB
```

Show PyTorch calculation:

```python
x = torch.empty(
    1000,
    1000,
    dtype=torch.float32
)

memory_bytes = (
    x.numel()
    * x.element_size()
)

print(memory_bytes)
```

Mark:

```text
🔥 Interview Useful
```

---

# 65. `element_size()`

Teach:

```python
x.element_size()
```

Examples:

```text
float32 → usually 4 bytes
float16 → usually 2 bytes
bfloat16 → usually 2 bytes
int64 → usually 8 bytes
```

Let learners calculate memory for several tensor shapes.

---

# 66. Model Parameter Memory

Create helper:

```python
def parameter_memory_bytes(model):

    return sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
```

Explain this only measures parameter tensors, not:

- gradients
- activations
- optimizer state

Ask learners why total training memory is larger.

---

# 67. Adam Optimizer Memory Concept

Explain at a conceptual level:

> Adam/AdamW maintains additional per-parameter optimizer state, so optimizer memory can be significant.

Do not give a simplistic exact multiplier as universal because precision and implementation can vary.

Explain:

```text
parameters
+
gradients
+
optimizer states
```

all contribute.

This is useful for large-model interviews.

---

# 68. Inference Memory Reduction Exercise

Ask:

> Which changes generally reduce inference memory?

Options such as:

- use `torch.inference_mode()`
- smaller batch
- lower precision
- avoid storing intermediate GPU outputs
- remove optimizer state from the inference process

Create multiple-choice and explanation.

---

# 69. GPU Performance Mental Model

Include:

```text
Fast training requires more than a GPU.

CPU / storage
    ↓
DataLoader
    ↓
CPU batch
    ↓
transfer
    ↓
GPU
    ↓
model compute
```

A bottleneck anywhere can reduce overall throughput.

---

# 70. Interview Questions

Create approximately 35–40 interview questions.

Include:

1. How do you check whether CUDA is available?
2. How do you select CPU or GPU dynamically?
3. What does `.to(device)` do?
4. Why do model and input need to be on the same device?
5. How do you move a model to GPU?
6. How do you move a tensor back to CPU?
7. `.cuda()` vs `.to(device)`?
8. Why is `.to(device)` more portable?
9. What is MPS in PyTorch?
10. What causes "Expected all tensors to be on the same device"?
11. Why are CPU-to-GPU transfers expensive?
12. Why should unnecessary device transfers be avoided?
13. Where are DataLoader batches usually created?
14. What does `pin_memory=True` do?
15. Does pinning memory move data to GPU?
16. What does `non_blocking=True` do?
17. What consumes GPU memory during training?
18. Why does training need more GPU memory than inference?
19. What does `torch.no_grad()` do for inference memory?
20. What is `torch.inference_mode()`?
21. What causes CUDA OOM?
22. What is the first thing you might try after OOM?
23. How does batch size affect GPU memory?
24. How does sequence length affect memory?
25. What does `torch.cuda.empty_cache()` do?
26. Does `empty_cache()` free tensors that are still referenced?
27. What is `memory_allocated()`?
28. What is `memory_reserved()`?
29. What is mixed precision?
30. Why use mixed precision?
31. float16 vs bfloat16?
32. Why can FP16 have numerical stability problems?
33. What does autocast do?
34. Why is gradient scaling used with FP16 training?
35. Is gradient scaling normally as important for bfloat16?
36. Why can CUDA timing with `time.time()` be misleading?
37. What does `torch.cuda.synchronize()` do?
38. Why can frequent `.item()` calls slow GPU training?
39. How would you diagnose low GPU utilization?
40. How would you estimate a tensor's memory size?

For each provide:

### Short Interview Answer

and:

### Detailed Explanation

Use collapsible `<details>` sections.

---

# 71. Knowledge Check Quiz

Create approximately 30 multiple-choice questions.

Example:

```text
Which is the most portable way to move a tensor?

A. x.cuda()
B. x.gpu()
C. x.to(device)
D. torch.move(x)
```

Correct:

```text
C
```

Another:

```text
A model is on CUDA but input X is on CPU.

What happens?

A. PyTorch automatically moves X
B. Model runs on CPU
C. Device mismatch error is likely
D. Gradients are disabled
```

Correct:

```text
C
```

Another:

```text
What usually uses more GPU memory?

A. Inference
B. Training
```

Correct:

```text
B
```

Another:

```text
Which generally uses about half the storage of float32?

A. int64
B. float16
C. float64
D. bool only
```

Correct:

```text
B
```

Put answers in a separate answer section.

---

# 72. Write From Memory

Create approximately 20 prompts.

Examples:

> Check if CUDA is available.

> Select CUDA, MPS, or CPU.

> Move model to `device`.

> Move `X` and `y` to `device`.

> Print a tensor's current device.

> Move a tensor to CPU.

> Convert a GPU tensor that requires gradients into NumPy safely.

> Build a DataLoader with pinned memory.

> Move a batch using `non_blocking=True`.

> Put a model into inference mode.

> Use `torch.no_grad()`.

> Use `torch.inference_mode()`.

> Calculate tensor memory using `numel()` and `element_size()`.

> Inspect allocated CUDA memory.

> Clear unused CUDA cache conditionally.

> Write an autocast forward pass.

> Write the standard mixed-precision scaling sequence.

Every executable task must run safely even without CUDA.

Use conditional validation where necessary.

---

# 73. GPU Syntax Flashcards

Create a rapid active-recall section.

Prompt:

```text
Check CUDA availability.
```

Answer cell should eventually contain:

```python
torch.cuda.is_available()
```

Prompt:

```text
Move model to device.
```

Expected:

```python
model.to(device)
```

Prompt:

```text
Move tensor X to device.
```

Expected:

```python
X = X.to(device)
```

Prompt:

```text
Move tensor back to CPU.
```

Expected:

```python
X = X.cpu()
```

Prompt:

```text
Disable gradients for inference.
```

Expected:

```python
with torch.no_grad():
```

Create approximately 15 of these.

---

# 74. Performance Debugging Scenarios

Create approximately 10 realistic scenarios.

### Scenario 1

```text
GPU utilization is 20%.
GPU memory is mostly unused.
GPU waits between batches.
```

Ask for possible causes.

Expected possibilities:

- DataLoader bottleneck
- batch too small
- preprocessing too slow
- storage too slow

---

### Scenario 2

```text
GPU utilization high
but CUDA OOM occurs after several hundred iterations.
```

Ask what might indicate a memory leak.

Possible:

```python
history.append(loss)
```

instead of:

```python
history.append(loss.item())
```

---

### Scenario 3

```text
Training is slower on GPU than CPU for a tiny model.
```

Explain possible reason:

> GPU launch/transfer overhead can dominate for very small workloads.

---

### Scenario 4

```text
Batch size 16 works.
Batch size 256 OOMs.
```

Ask for solutions.

---

### Scenario 5

```text
Inference uses much more memory than expected.
```

Ask whether gradient tracking may still be enabled.

---

# 75. Final GPU-Aware Training Challenge

Create a complete device-aware multiclass training task.

Use synthetic data:

```python
torch.manual_seed(42)

X = torch.randn(2000, 64)

true_weights = torch.randn(64, 5)

scores = X @ true_weights

y = scores.argmax(dim=1)
```

Split train / validation.

Create DataLoaders.

---

# 76. Final Challenge — Device Selection

Learner must write:

```python
if ...:
    device = ...
elif ...:
    device = ...
else:
    device = ...
```

Requirements:

```text
CUDA first
MPS second
CPU otherwise
```

Validate `device.type`.

---

# 77. Final Challenge — DataLoaders

Use:

```text
train batch size = 64
val batch size = 128
```

Use:

```python
pin_memory=torch.cuda.is_available()
```

Training should shuffle.

Validation should not.

---

# 78. Final Challenge — Model

Build:

```text
64
↓
128
↓ ReLU
↓
64
↓ ReLU
↓
5 logits
```

Move model to:

```python
device
```

---

# 79. Final Challenge — Training Function

Implement:

```python
def train_one_epoch(...):
```

Requirements:

- `model.train()`
- transfer batch to device
- optional non-blocking transfer when CUDA
- `zero_grad()`
- forward
- loss
- backward
- optimizer step
- loss tracking
- accuracy

---

# 80. Final Challenge — Evaluation

Implement:

```python
def evaluate(...):
```

Requirements:

- `model.eval()`
- `torch.inference_mode()` or `torch.no_grad()`
- transfer data to device
- no backward
- no optimizer update
- calculate loss
- calculate accuracy

---

# 81. Final Challenge — Optional Mixed Precision

If:

```text
device.type == "cuda"
```

allow learner to add mixed precision.

Use a clearly isolated optional block so CPU/MPS users can complete everything without it.

The mixed-precision solution should use modern PyTorch APIs.

Before writing the final notebook, check the installed PyTorch API to avoid deprecated AMP syntax.

---

# 82. Final Challenge — Report Device and Memory

Print:

```text
Using device: ...
```

If CUDA:

print:

- allocated memory
- reserved memory
- peak allocated memory

If not CUDA:

print a friendly message:

```text
CUDA memory metrics unavailable on this device.
```

No cell should fail.

---

# 83. Final Challenge — Tensor Memory Calculation

Ask learner to calculate how much memory this input batch uses:

```python
batch = torch.randn(
    64,
    3,
    224,
    224
)
```

Use:

```python
batch.numel() * batch.element_size()
```

Display MiB.

Then ask what happens approximately if dtype changes from:

```text
float32
→
float16
```

Expected:

> Tensor storage is approximately halved.

---

# 84. Optional Hard Challenge — Find Safe Batch Size

Add:

```text
🔴 Hard / Optional
```

On CUDA only:

Experimentally increase synthetic batch size until a predefined limit or OOM is approached.

But do NOT intentionally crash the notebook.

Instead create a safe demonstration or explain how one could benchmark candidate batch sizes with exception handling.

Avoid allocating dangerously huge tensors.

---

# 85. Final Cheat Sheet

End with a compact cheat sheet.

Include:

```python
# Device selection
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

Then:

```python
# Move model
model = model.to(device)

# Move tensors
X = X.to(device)
y = y.to(device)

# Check device
print(X.device)

# CPU
X = X.cpu()

# Safe NumPy conversion
array = X.detach().cpu().numpy()
```

DataLoader:

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
)
```

Training:

```python
for X, y in loader:

    X = X.to(
        device,
        non_blocking=torch.cuda.is_available()
    )

    y = y.to(
        device,
        non_blocking=torch.cuda.is_available()
    )

    optimizer.zero_grad()

    logits = model(X)

    loss = loss_fn(logits, y)

    loss.backward()

    optimizer.step()
```

Inference:

```python
model.eval()

with torch.inference_mode():
    output = model(X)
```

CUDA memory:

```python
if torch.cuda.is_available():

    print(
        torch.cuda.memory_allocated()
    )

    print(
        torch.cuda.memory_reserved()
    )

    print(
        torch.cuda.max_memory_allocated()
    )
```

Tensor memory:

```python
memory_bytes = (
    x.numel()
    * x.element_size()
)
```

Mixed precision concept:

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16
):
    output = model(X)
```

Only show GradScaler syntax using a current, non-deprecated API compatible with the PyTorch environment.

---

# 86. Critical Mental Models

Include:

```text
MODEL DEVICE
=
INPUT DEVICE
=
TARGET DEVICE
```

for normal training operations.

And:

```text
Dataset / DataLoader
      ↓
CPU batch
      ↓
.to(device)
      ↓
GPU/MPS model
      ↓
compute
```

And:

```text
Training memory
=
parameters
+
activations
+
gradients
+
optimizer state
+
batch
+
temporary buffers
```

And:

```text
OOM?
↓
reduce batch size
↓
mixed precision
↓
gradient accumulation
↓
reduce unnecessary tensors / graph retention
↓
consider more advanced techniques
```

---

# 87. Interview Quick Reference

Add:

```text
torch.cuda.is_available()
→ check CUDA

torch.backends.mps.is_available()
→ check Apple MPS

.to(device)
→ move tensor/model

.cpu()
→ move to CPU

pin_memory
→ potentially faster CPU→CUDA transfer

non_blocking
→ allow asynchronous transfer when conditions permit

torch.no_grad()
→ disable Autograd

torch.inference_mode()
→ inference-optimized no-grad mode

float16
→ lower-memory 16-bit floating point

bfloat16
→ 16-bit format with larger exponent range

autocast
→ automatically choose lower/higher precision operations

GradScaler
→ helps FP16 gradients avoid underflow

memory_allocated
→ live tensor CUDA memory

memory_reserved
→ CUDA memory reserved by PyTorch allocator

empty_cache
→ releases unused cached CUDA memory, not live tensors
```

---

# 88. Completion Checklist

End with:

```text
## Before Moving to 09_debugging_and_performance.ipynb
```

Add:

- [ ] I can check whether CUDA is available.
- [ ] I can detect Apple MPS.
- [ ] I can select a portable device.
- [ ] I can move a tensor to a device.
- [ ] I can move a model to a device.
- [ ] I understand device mismatch errors.
- [ ] I can move outputs back to CPU.
- [ ] I know how to convert a GPU tensor to NumPy safely.
- [ ] I understand CPU ↔ GPU transfer cost.
- [ ] I understand `pin_memory`.
- [ ] I understand `non_blocking=True`.
- [ ] I understand what consumes GPU memory.
- [ ] I understand why training uses more memory than inference.
- [ ] I understand CUDA OOM causes.
- [ ] I know the first steps for reducing OOM.
- [ ] I understand `torch.cuda.empty_cache()`.
- [ ] I understand why storing graph-connected tensors can leak memory.
- [ ] I can calculate tensor memory from `numel()` and `element_size()`.
- [ ] I understand float32, float16, and bfloat16.
- [ ] I understand mixed precision.
- [ ] I understand autocast.
- [ ] I understand the purpose of gradient scaling.
- [ ] I know CUDA benchmarking may require synchronization.
- [ ] I can write a fully device-agnostic training loop.

---

# 89. Difficulty Labels

Use:

```text
🟢 Easy
🟡 Medium
🔴 Hard
```

Easy:

- device checking
- `.to(device)`
- `.cpu()`
- device mismatch
- basic OOM understanding

Medium:

- pinned memory
- non-blocking transfers
- CUDA allocator concepts
- mixed precision
- GradScaler
- CUDA timing

Hard / optional:

- profiling
- manual memory analysis
- advanced performance tuning

Do not yet deeply cover:

- DistributedDataParallel
- FSDP
- tensor parallelism
- pipeline parallelism
- DeepSpeed
- ZeRO
- custom CUDA kernels
- `torch.compile`
- CUDA graphs

Those belong in later/advanced material.

---

# 90. Quality Requirements

The notebook must:

- be a valid `.ipynb`
- execute from top to bottom
- run successfully with CPU only
- support CUDA when available
- support Apple MPS where reasonable
- never crash merely because CUDA is unavailable
- require no internet
- use only PyTorch plus Python standard library
- contain automatic `assert` validation
- contain approximately 70–90 exercises/questions
- heavily emphasize device debugging
- heavily emphasize GPU-memory reasoning
- heavily emphasize practical AI Engineer interview knowledge
- use modern, non-deprecated PyTorch APIs
- clearly separate CUDA-only examples using safe conditional execution

The notebook should take approximately **2–3 hours** to study thoroughly.

The core learning principle is:

```text
Do not memorize only:

model.to("cuda")

Understand the complete flow:

detect hardware
      ↓
choose device
      ↓
move model
      ↓
load CPU batch
      ↓
move batch
      ↓
compute
      ↓
manage memory
      ↓
move results only when needed
      ↓
debug performance / OOM
```

By the end of the notebook, the learner should be able to look at a PyTorch training script and quickly diagnose:

```text
device mismatch
unnecessary transfer
GPU memory issue
incorrect inference mode
low-precision issue
data-transfer bottleneck
```

Finally save the notebook as:

```text
08_gpu_and_devices.ipynb
```