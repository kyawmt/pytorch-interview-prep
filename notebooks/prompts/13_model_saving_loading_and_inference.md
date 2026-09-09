Create a Jupyter Notebook named:

```text
13_model_saving_loading_and_inference.ipynb
```

This notebook is the next part of a **PyTorch Interview Preparation** course for someone preparing for **AI Engineer / Machine Learning Engineer / Generative AI Engineer interviews**.

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
11_pytorch_interview_challenge.ipynb
12_pytorch_mini_projects.ipynb
```

Do **not** reteach tensors, Autograd, model construction, losses, optimizers, or basic training loops.

The purpose of this notebook is to teach what happens **after or around training**:

```text
train model
↓
save model
↓
save checkpoint
↓
load model
↓
resume training
↓
run inference
↓
batch inference
↓
manage devices
↓
handle model versions
↓
debug loading problems
```

This is highly relevant for practical AI Engineer interviews because a model that can be trained but cannot be reliably saved, restored, and used for inference is incomplete.

---

# Main Learning Goals

By the end of the notebook, the learner should confidently understand and use:

```python
model.state_dict()
model.load_state_dict(...)

optimizer.state_dict()
optimizer.load_state_dict(...)

torch.save(...)
torch.load(...)

model.train()
model.eval()

torch.no_grad()
torch.inference_mode()
```

The learner should also understand:

- what a `state_dict` is
- saving model weights
- loading model weights
- why model architecture must exist before loading weights
- full training checkpoints
- optimizer state
- epoch / step restoration
- scheduler state concept
- resuming training
- `map_location`
- loading CUDA checkpoints on CPU
- strict vs non-strict state-dict loading
- missing keys
- unexpected keys
- architecture mismatches
- frozen vs trainable models after loading
- inference mode
- single-example inference
- batch inference
- logits → predictions
- binary vs multiclass inference
- preprocessing consistency
- output postprocessing
- moving outputs to CPU
- saving predictions
- model reproducibility
- checkpoint naming/versioning
- "best model" checkpointing
- deployment-oriented model preparation
- TorchScript and `torch.compile` only at a high-level awareness level

Mark these as:

```text
🔥 Interview Essential
```

- `state_dict`
- `torch.save`
- `torch.load`
- loading weights into the same architecture
- checkpoint vs weights-only saving
- `map_location`
- `model.eval()`
- `torch.inference_mode()`
- resuming training
- saving the best validation model
- preprocessing consistency at inference

---

# Notebook Structure

Use approximately:

```text
# PyTorch Model Saving, Loading, and Inference

## 1. Setup
## 2. Why Saving Models Matters
## 3. What Is a state_dict?
## 4. Inspecting model.state_dict()
## 5. Saving Model Weights
## 6. Loading Model Weights
## 7. Architecture + Weights
## 8. Verify Loaded Model
## 9. Saving Full Training Checkpoints
## 10. Optimizer State
## 11. Resume Training
## 12. Saving the Best Model
## 13. map_location
## 14. Loading GPU Checkpoints on CPU
## 15. strict=True vs strict=False
## 16. Missing and Unexpected Keys
## 17. Architecture Mismatch
## 18. Inference Mode
## 19. Single-Sample Inference
## 20. Batch Inference
## 21. Binary Classification Inference
## 22. Multiclass Inference
## 23. Regression Inference
## 24. Text-Model Inference
## 25. Preprocessing Consistency
## 26. Device-Aware Inference
## 27. Efficient Large-Batch Inference
## 28. Saving Predictions
## 29. Checkpoint Management
## 30. Common Mistakes
## 31. Debugging Exercises
## 32. Interview Questions
## 33. Knowledge Check Quiz
## 34. Write From Memory
## 35. Final Checkpoint + Inference Challenge
## 36. Cheat Sheet
```

---

# 1. Setup

Start with:

```python
import os
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
```

Choose device:

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

Use temporary directories or notebook-local temporary files so the notebook does not clutter the learner's filesystem.

For example:

```python
tmpdir = tempfile.mkdtemp()
```

All examples must run offline and on CPU.

---

# 2. Why Save a Model?

Explain simply:

```text
Training may take:
minutes
hours
days
weeks

You do not want to retrain every time you need predictions.
```

Typical lifecycle:

```text
train
↓
save
↓
close program
↓
later
↓
load
↓
inference / resume training
```

Explain two common goals:

```text
Goal A:
save trained weights for inference

Goal B:
save full checkpoint so training can resume
```

Mark:

```text
🔥 Interview Essential
```

---

# 3. Create a Small Model

Use:

```python
class SimpleClassifier(nn.Module):

    def __init__(
        self,
        input_dim=10,
        hidden_dim=32,
        num_classes=3
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)
```

Create:

```python
model = SimpleClassifier()
```

---

# 4. What Is a `state_dict`?

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
state = model.state_dict()
```

Explain:

> A `state_dict` is a Python dictionary mapping parameter/buffer names to tensors.

Show:

```python
for key, value in model.state_dict().items():
    print(key, value.shape)
```

Expected names similar to:

```text
net.0.weight
net.0.bias
net.2.weight
net.2.bias
```

Explain:

```text
key
→ parameter name

value
→ tensor containing that parameter
```

---

# 5. `state_dict()` Is Not the Model Architecture

Make this very clear.

Explain:

```text
state_dict
=
parameter values

NOT
=
Python model class definition
```

Therefore to load:

```python
new_model = SimpleClassifier()

new_model.load_state_dict(...)
```

The architecture must already exist.

Mark:

```text
🔥 Very Common Interview Point
```

---

# 6. Save Model Weights

Teach:

```python
path = os.path.join(
    tmpdir,
    "model_weights.pt"
)

torch.save(
    model.state_dict(),
    path
)
```

Explain common extensions:

```text
.pt
.pth
```

Both are conventions.

Do not imply a functional difference.

---

# 7. Exercise — Save Weights

Give:

```python
save_path = os.path.join(
    tmpdir,
    "exercise_weights.pt"
)

# Save only model weights here.
```

Validation:

```python
assert os.path.exists(save_path)

loaded = torch.load(
    save_path,
    map_location="cpu",
    weights_only=True
)

assert isinstance(loaded, dict)

print("✅ Correct!")
```

Use current PyTorch-compatible `torch.load` arguments.

If `weights_only=True` behavior depends on the installed PyTorch version, ensure the generated notebook uses APIs supported by that environment.

Prefer secure/current PyTorch loading practices.

---

# 8. Load Model Weights

Teach:

```python
new_model = SimpleClassifier()

state_dict = torch.load(
    path,
    map_location="cpu",
    weights_only=True
)

new_model.load_state_dict(
    state_dict
)
```

Then:

```python
new_model.eval()
```

Explain:

```text
1. Re-create architecture
2. Load saved parameter dictionary
3. Put model into inference mode
```

---

# 9. Verify Models Match

Create:

```python
model.eval()
new_model.eval()

X = torch.randn(8, 10)

with torch.inference_mode():
    output1 = model(X)
    output2 = new_model(X)
```

Validate:

```python
assert torch.allclose(
    output1,
    output2
)

print("✅ Loaded model matches")
```

Explain this is a strong sanity check after loading.

---

# 10. Check Parameter Equality

Show:

```python
for (
    name1,
    param1
), (
    name2,
    param2
) in zip(
    model.named_parameters(),
    new_model.named_parameters()
):

    assert name1 == name2

    assert torch.equal(
        param1,
        param2
    )
```

Explain loaded weights should exactly match when using the same checkpoint.

---

# 11. Weights-Only vs Full Checkpoint

Create comparison:

| Weights-only | Training checkpoint |
|---|---|
| model state | model state |
| simpler | includes more training state |
| good for inference | good for resuming training |
| smaller | larger |
| architecture still required | architecture still generally required |

Explain full checkpoint may include:

```text
model_state_dict
optimizer_state_dict
epoch
global_step
validation_loss
scheduler_state_dict
scaler_state_dict
configuration
```

depending on the project.

---

# 12. Optimizer State

Mark:

```text
🔥 Interview Essential
```

Create optimizer:

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3
)
```

Explain:

```python
optimizer.state_dict()
```

contains:

- parameter-group configuration
- optimizer internal state

For Adam-style optimizers this includes running statistics used by future updates.

Therefore:

> Loading model weights alone does not fully restore an interrupted training run.

---

# 13. Full Training Checkpoint

Teach:

```python
checkpoint = {
    "epoch": 5,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_loss": 0.25,
    "val_loss": 0.31,
}
```

Save:

```python
checkpoint_path = os.path.join(
    tmpdir,
    "checkpoint.pt"
)

torch.save(
    checkpoint,
    checkpoint_path
)
```

---

# 14. Load Full Checkpoint

Show:

```python
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)
```

Then:

```python
model = SimpleClassifier().to(device)

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3
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

start_epoch = (
    checkpoint["epoch"] + 1
)
```

Explain the security implications of loading arbitrary serialized Python objects and why you should only load trusted checkpoint files.

Do not overcomplicate the security discussion, but clearly distinguish trusted local checkpoints from untrusted files.

---

# 15. Resume Training

Mark:

```text
🔥 Interview Essential
```

Show conceptual sequence:

```text
load architecture
↓
load model state
↓
create optimizer
↓
load optimizer state
↓
restore epoch / step
↓
model.train()
↓
continue training
```

Give a small runnable example:

1. Train for 3 epochs.
2. Save checkpoint.
3. Create a new model.
4. Load checkpoint.
5. Continue for another 2 epochs.

Validate that training continues normally.

---

# 16. Exercise — Resume Training

Learner must fill:

```python
restored_model = ...
restored_optimizer = ...

checkpoint = ...

...
```

Validation should ensure:

```python
assert start_epoch == saved_epoch + 1
```

and at least one optimizer state entry exists after relevant training has occurred.

---

# 17. Why Optimizer State Matters

Explain intuitively.

For SGD without momentum:

```text
optimizer state may be minimal
```

For:

```text
SGD + momentum
Adam
AdamW
```

the optimizer remembers information from previous gradients.

If optimizer state is lost:

```text
same model weights
but
optimization history resets
```

Training can continue, but it is not exactly the same continuation.

---

# 18. Save Scheduler State

Briefly introduce if scheduler exists:

```python
"scheduler_state_dict":
    scheduler.state_dict()
```

Do not deeply teach LR schedulers yet if they have not been covered.

Explain the general rule:

> Anything with training state needed to reproduce/resume the run should usually be checkpointed.

---

# 19. Save AMP Scaler State

For FP16 CUDA mixed precision, mention:

```python
"scaler_state_dict":
    scaler.state_dict()
```

when using a gradient scaler.

Keep optional.

No CUDA requirement.

---

# 20. Save Best Model

Make this a major practical section.

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
best_val_loss = float("inf")

for epoch in range(num_epochs):

    train(...)

    val_loss = evaluate(...)

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            best_model_path
        )
```

Explain:

> The final epoch is not necessarily the best model.

Use validation metric to choose the checkpoint.

---

# 21. Best Model vs Last Model

Explain:

```text
last checkpoint
→ state at end of training

best checkpoint
→ best according to selected validation metric
```

Sometimes both should be saved.

For example:

```text
checkpoint_last.pt
checkpoint_best.pt
```

---

# 22. Saving Based on Accuracy

Show:

```python
if val_accuracy > best_val_accuracy:
    ...
```

Explain whether higher or lower is "better" depends on the metric:

```text
loss
→ lower better

accuracy
→ higher better
```

---

# 23. `map_location`

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
state_dict = torch.load(
    path,
    map_location="cpu",
    weights_only=True
)
```

Explain:

> `map_location` controls where saved tensors are reconstructed.

Useful when:

```text
checkpoint created on CUDA
↓
later loaded on CPU-only machine
```

Typical portable pattern:

```python
torch.load(
    path,
    map_location=device,
    ...
)
```

---

# 24. GPU Checkpoint → CPU

Explain scenario:

```text
Training machine:
CUDA

Inference machine:
CPU
```

Solution:

```python
torch.load(
    path,
    map_location="cpu",
    weights_only=True
)
```

Then:

```python
model.load_state_dict(...)
model.to("cpu")
```

---

# 25. CPU Checkpoint → GPU

Show:

```python
state = torch.load(
    path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state)
model.to(device)
```

Explain either loading then moving or mapping appropriately can be valid depending on workflow.

---

# 26. `strict=True`

Teach default:

```python
model.load_state_dict(
    state_dict,
    strict=True
)
```

Explain:

> Every expected key must match, and unexpected keys are not allowed.

This is safest when architecture is expected to be exactly the same.

---

# 27. `strict=False`

Teach:

```python
result = model.load_state_dict(
    state_dict,
    strict=False
)
```

Inspect:

```python
print(result.missing_keys)
print(result.unexpected_keys)
```

Explain use case:

- load pretrained backbone
- architecture changed slightly
- classifier head replaced
- transfer learning

Mark:

```text
🟡 Interview Useful
```

---

# 28. Missing Keys

Explain:

```text
Checkpoint does not contain parameters
the current model expects.
```

Example:

Old model:

```text
backbone
```

New model:

```text
backbone
+
classifier
```

New classifier parameters become:

```text
missing keys
```

---

# 29. Unexpected Keys

Explain:

```text
Checkpoint contains parameters
that the current model does not have.
```

Example:

Old checkpoint includes:

```text
classifier
```

but current architecture removed it.

---

# 30. Shape Mismatch

Important distinction:

Even:

```python
strict=False
```

does not automatically solve incompatible parameter tensor shapes for matching keys.

Example:

Checkpoint:

```text
classifier.weight:
(10, 128)
```

New model:

```text
classifier.weight:
(5, 128)
```

Ask learner what happened.

Expected:

> The number of classes changed, so the classifier layer shape changed.

A common solution is to load only compatible backbone weights or remove/reinitialize classifier keys.

---

# 31. Transfer Learning Loading Pattern

Show conceptually:

```text
Pretrained model:
backbone + 1000-class head

New task:
backbone + 10-class head
```

Pattern:

1. Build new architecture.
2. Load compatible backbone weights.
3. Ignore/reinitialize the new head.
4. Train the new head and optionally fine-tune backbone.

Do not require downloading a pretrained model.

Use a synthetic example.

---

# 32. Exercise — New Classifier Head

Create:

```python
class BackboneModel(nn.Module):
    ...
```

Save an original 5-class model.

Then create a 3-class version.

Ask learner to load only compatible backbone parameters while leaving the new head randomly initialized.

Validate:

- backbone weights match old model
- classifier shape is `(3, hidden_size)`
- classifier is not incorrectly loaded from old 5-class head

---

# 33. Inference Mode

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
model.eval()

with torch.inference_mode():
    logits = model(X)
```

Explain two separate roles:

```text
model.eval()
→ changes Dropout/BatchNorm behavior

torch.inference_mode()
→ disables Autograd overhead for inference
```

Review without overexplaining.

---

# 34. Why Both?

Ask:

> Is `model.eval()` enough?

Expected:

```text
No.
```

It does not disable Autograd.

Ask:

> Is `inference_mode()` enough?

Expected:

> It disables gradient tracking, but does not switch Dropout/BatchNorm into evaluation mode.

Therefore common inference pattern:

```python
model.eval()

with torch.inference_mode():
    ...
```

---

# 35. Single-Sample Inference

Suppose classifier expects:

```text
(B,10)
```

A single feature vector:

```python
x = torch.randn(10)
```

Ask learner to add a batch dimension:

```python
x = x.unsqueeze(0)
```

Shape:

```text
(1,10)
```

Then:

```python
logits = model(x)
```

Expected:

```text
(1,3)
```

---

# 36. Multiclass Inference

Teach:

```python
model.eval()

with torch.inference_mode():

    logits = model(X)

    predictions = logits.argmax(
        dim=1
    )
```

If probabilities are actually needed:

```python
probabilities = torch.softmax(
    logits,
    dim=1
)
```

Explain:

```text
training loss
→ raw logits

user-facing probabilities
→ softmax when required
```

---

# 37. Top-K Inference

Teach:

```python
probs = torch.softmax(
    logits,
    dim=1
)

top_probs, top_classes = torch.topk(
    probs,
    k=3,
    dim=1
)
```

Use case:

- image-classification top predictions
- recommendation/ranking-like output
- candidate classes

---

# 38. Binary Inference

Teach:

```python
with torch.inference_mode():
    logits = model(X)

probs = torch.sigmoid(logits)

pred = (
    probs >= 0.5
).long()
```

Reinforce:

```text
Training:
raw logits → BCEWithLogitsLoss

Inference:
logits → sigmoid → threshold
```

---

# 39. Regression Inference

Teach:

```python
with torch.inference_mode():
    predictions = model(X)
```

No classification probability conversion.

Explain outputs may need domain-specific inverse scaling/postprocessing.

---

# 40. Preprocessing Consistency

Make this a major section.

Mark:

```text
🔥 Interview Essential
```

Explain:

> Inference must use the same input preprocessing assumptions as training.

Example:

Training used:

```text
x_normalized =
(x - train_mean) / train_std
```

Inference must use the same:

```text
train_mean
train_std
```

not recompute statistics from one new sample.

This is a common ML system bug.

---

# 41. Save Preprocessing State

Example checkpoint:

```python
checkpoint = {
    "model_state_dict":
        model.state_dict(),

    "feature_mean":
        train_mean,

    "feature_std":
        train_std,
}
```

Explain preprocessing metadata may be part of the artifact needed for reliable inference.

---

# 42. Class Name Mapping

For classification, save:

```python
class_names = [
    "cat",
    "dog",
    "bird"
]
```

or:

```python
class_to_idx = {
    "cat": 0,
    "dog": 1,
    "bird": 2,
}
```

Explain:

> A numeric output class is useless if the index-to-label mapping is lost.

Include it in checkpoint metadata where appropriate.

---

# 43. Model Configuration

Save architecture-related metadata:

```python
config = {
    "input_dim": 10,
    "hidden_dim": 32,
    "num_classes": 3,
}
```

Then reconstruction can use:

```python
model = SimpleClassifier(
    **config
)
```

Explain this reduces architecture/version mistakes.

---

# 44. Build a Reproducible Checkpoint

Example:

```python
checkpoint = {
    "model_state_dict":
        model.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "epoch":
        epoch,

    "config":
        config,

    "class_names":
        class_names,

    "best_val_loss":
        best_val_loss,
}
```

Explain real production systems may track much more:

- code version
- dataset version
- hyperparameters
- tokenizer information
- feature schema

Keep these as ML engineering awareness.

---

# 45. Batch Inference

Create:

```python
inference_loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False
)
```

Then:

```python
model.eval()

all_predictions = []

with torch.inference_mode():

    for X_batch in inference_loader:

        X_batch = X_batch.to(device)

        logits = model(X_batch)

        pred = logits.argmax(dim=1)

        all_predictions.append(
            pred.cpu()
        )

all_predictions = torch.cat(
    all_predictions
)
```

Explain why outputs are moved to CPU before long-term storage.

---

# 46. Why Not Store All GPU Outputs?

Explain:

```text
GPU memory is limited.
```

Bad:

```python
all_logits.append(logits)
```

Better when gradients not needed:

```python
all_logits.append(
    logits.cpu()
)
```

inside `inference_mode()`.

For training-time outputs, use `detach().cpu()` if appropriate.

---

# 47. Preserve Input Order

Explain:

For inference:

```python
shuffle=False
```

is often important when predictions must correspond exactly to the original input ordering.

Mark:

```text
🔥 Practical ML Engineering Point
```

---

# 48. Batch Size During Inference

Explain:

```text
larger batch
→ potentially higher throughput
→ more memory
```

and:

```text
smaller batch
→ lower memory
→ potentially lower throughput
```

Inference batch size should be benchmarked.

Do not assume largest possible is always best.

---

# 49. Latency vs Throughput

Introduce interview concept.

```text
Latency
→ time for one request/sample

Throughput
→ samples processed per unit time
```

Batching often:

```text
improves throughput
but
may increase request latency
```

This is important for AI Engineer interviews.

---

# 50. Inference Timing

Use:

```python
import time
```

CPU:

```python
start = time.perf_counter()

with torch.inference_mode():
    output = model(X)

elapsed = (
    time.perf_counter()
    - start
)
```

If CUDA, synchronize before/after measurement as learned previously.

---

# 51. Warm-Up for Inference Benchmarking

Explain:

```python
with torch.inference_mode():
    for _ in range(10):
        _ = model(X)
```

then measure repeated runs.

Do not focus heavily on microbenchmarking.

---

# 52. Model File Naming

Introduce practical naming:

```text
model_best.pt
model_last.pt
checkpoint_epoch_010.pt
checkpoint_step_050000.pt
```

Explain avoid ambiguous:

```text
model_final_final_v2_REAL.pt
```

Use structured naming/versioning.

---

# 53. Keep Best and Last

Recommend pattern:

```text
best
→ deployment/evaluation candidate

last
→ exact most recent training state
```

This is especially useful with early stopping or interrupted training.

---

# 54. Checkpoint Frequency

Ask:

> Save every training step?

Explain trade-offs:

```text
very frequent
→ more I/O and storage

too infrequent
→ more work lost after interruption
```

Frequency depends on training duration/cost.

No universal interval.

---

# 55. Atomic/Robust Saving Awareness

Briefly mention production systems may:

1. write a temporary checkpoint
2. confirm save
3. rename to final checkpoint

to reduce risk of corrupted partial files.

Do not implement full filesystem robustness unless simple.

Mark as ML engineering awareness.

---

# 56. `torch.save(model)` vs `state_dict`

Explain that PyTorch can serialize entire Python objects in some workflows, but **prefer teaching `state_dict`-based saving** for ordinary model checkpoints because it is:

- clearer
- more flexible
- less tied to Python class serialization details
- standard practice

Do not frame full-model serialization as universally forbidden; explain why state-dict workflows are generally preferred.

---

# 57. Loading Untrusted Files

Add a concise safety note:

> Do not blindly load untrusted PyTorch checkpoint files. Python-based serialization can have security implications. Use trusted artifacts and current safer PyTorch loading options where possible.

Keep this short but explicit.

---

# 58. Common Mistake — Forgetting `eval()`

Model contains Dropout.

Load model:

```python
model.load_state_dict(state)
```

Then immediately infer without:

```python
model.eval()
```

Ask learner what might happen.

Expected:

> Dropout stays active because a newly created model defaults to training mode.

---

# 59. Common Mistake — Wrong Architecture

Checkpoint comes from:

```text
10 → 32 → 3
```

New model:

```text
10 → 64 → 3
```

Ask why loading fails.

Expected:

> Parameter tensor shapes differ.

---

# 60. Common Mistake — Wrong Class Count

Checkpoint:

```text
num_classes=10
```

Current model:

```text
num_classes=5
```

Classifier layer mismatch.

Ask how transfer learning might handle it.

---

# 61. Common Mistake — Saving Only Model When You Need Resume

User saves:

```python
torch.save(
    model.state_dict(),
    path
)
```

then expects exact optimizer continuation.

Ask what's missing.

Expected:

```text
optimizer state
epoch/step
possibly scheduler/scaler state
```

---

# 62. Common Mistake — Recomputed Normalization

Training used train-set mean/std.

Inference does:

```python
mean = sample.mean()
std = sample.std()
```

for each sample.

Ask why this is wrong.

Expected:

> Inference preprocessing no longer matches training preprocessing.

---

# 63. Common Mistake — Wrong Label Map

Model output class index `2`.

Old mapping:

```text
2 → bird
```

new deployment mapping:

```text
2 → dog
```

Model is mathematically correct but application output is wrong.

This is an important ML engineering scenario.

---

# 64. Common Mistake — Loading to Missing CUDA

Show:

```python
checkpoint = torch.load(path)
```

on a CPU-only system when checkpoint tensors were saved on CUDA.

Ask learner to fix with:

```python
map_location="cpu"
```

where needed.

---

# 65. Common Mistake — Inference With Gradients

Code:

```python
model.eval()

for X in loader:
    output = model(X)
```

Ask what's missing for efficient inference.

Expected:

```python
torch.inference_mode()
```

or `torch.no_grad()`.

---

# 66. Common Mistake — Storing GPU Predictions

```python
predictions.append(
    logits
)
```

for thousands of batches.

Ask why memory grows.

Expected:

> GPU tensors are being retained.

Fix:

```python
predictions.append(
    logits.cpu()
)
```

during inference.

---

# 67. Debugging Exercises

Create at least **25 debugging scenarios**.

Each should include:

```text
Symptom
Broken code
Question
Hidden answer
Fix
```

Include cases for:

1. missing `eval()`
2. missing `inference_mode`
3. missing architecture before loading
4. wrong hidden size
5. wrong class count
6. missing optimizer state
7. wrong `map_location`
8. missing checkpoint key
9. unexpected checkpoint key
10. strict loading mismatch
11. classifier-head mismatch
12. wrong class mapping
13. wrong normalization
14. wrong device after loading
15. single sample missing batch dimension
16. binary inference missing sigmoid
17. multiclass inference using sigmoid
18. softmax on wrong dimension
19. shuffled inference DataLoader when preserving order matters
20. GPU outputs kept forever
21. checkpoint overwritten accidentally
22. best-model condition reversed
23. validation loss comparison bug
24. resumed epoch starts at wrong number
25. optimizer constructed before/after load incorrectly explained
26. checkpoint created before optimizer has meaningful state
27. model loaded but parameters intentionally/accidentally frozen

Use at least 25.

---

# 68. Save/Load Round-Trip Exercise

Learner should:

1. Create model.
2. Run one optimizer step.
3. Save.
4. Create brand-new model.
5. Load.
6. Compare outputs exactly.

Require:

```python
assert torch.allclose(
    original_output,
    restored_output
)
```

This should be a central exercise.

---

# 69. Checkpoint Round-Trip Exercise

Learner should save:

```text
model
optimizer
epoch
loss
config
```

Then restore all values.

Automatically verify:

```python
assert restored_epoch == original_epoch
assert restored_config == config
```

and model outputs match.

---

# 70. Inference API Function

Ask learner to build:

```python
def predict_multiclass(
    model,
    X,
    device
):
    ...
```

Requirements:

- `model.eval()`
- `torch.inference_mode()`
- move X
- calculate logits
- calculate probabilities
- return CPU predictions and probabilities

Validation:

```text
pred shape:
(B,)

prob shape:
(B,C)

row probability sums:
≈ 1
```

---

# 71. Binary Prediction Function

Ask learner to implement:

```python
def predict_binary(
    model,
    X,
    device,
    threshold=0.5
):
    ...
```

Return:

- probabilities
- predictions

Validate values are in:

```text
probabilities:
[0,1]

predictions:
0 or 1
```

---

# 72. Regression Prediction Function

Ask learner to implement:

```python
def predict_regression(
    model,
    X,
    device
):
    ...
```

No softmax or sigmoid.

---

# 73. Model Metadata Inspection

Show:

```python
checkpoint.keys()
```

and ask learner to inspect without assuming expected keys.

This is useful when receiving an unfamiliar checkpoint.

Also inspect:

```python
state_dict.keys()
```

to understand architecture naming.

---

# 74. Rename Prefix Debugging Awareness

Mention checkpoint keys may sometimes contain prefixes such as:

```text
module.
model.
backbone.
```

from wrappers or different codebases.

Example:

```text
module.layer1.weight
```

vs current model:

```text
layer1.weight
```

Explain this can produce missing/unexpected keys.

Do not deeply cover distributed wrappers yet.

---

# 75. Interview Questions

Create approximately **40 interview questions**.

Include:

1. What is a PyTorch `state_dict`?
2. What does `model.state_dict()` contain?
3. Does a state_dict contain model architecture?
4. How do you save model weights?
5. How do you load model weights?
6. Why must the architecture exist before loading?
7. Why is state-dict saving commonly preferred?
8. What should a training checkpoint contain?
9. Why save optimizer state?
10. What happens if Adam state isn't restored?
11. How do you resume training?
12. Why save the epoch/global step?
13. What is the difference between best and last checkpoint?
14. How do you save the best model?
15. What is `map_location`?
16. How do you load a GPU-trained checkpoint on CPU?
17. What does `strict=True` mean?
18. When might `strict=False` be useful?
19. What are missing keys?
20. What are unexpected keys?
21. Can `strict=False` fix tensor shape mismatches automatically?
22. How would you replace a pretrained classifier head?
23. What does `model.eval()` do?
24. What does `torch.inference_mode()` do?
25. Why use both for inference?
26. How do you run binary classification inference?
27. How do you run multiclass inference?
28. Why use raw logits during training but probabilities for presentation?
29. How do you handle one sample when a model expects a batch?
30. Why must preprocessing match training?
31. Why save class-index mappings?
32. Why might you save model configuration?
33. What would you include in a reproducible checkpoint?
34. Why use `shuffle=False` for ordered inference?
35. What is latency?
36. What is throughput?
37. How can batching affect throughput and latency?
38. Why move inference outputs to CPU?
39. How would you diagnose a checkpoint load failure?
40. What security concern exists when loading untrusted checkpoint files?
41. How would you version checkpoints?
42. Why keep both a best and last checkpoint?

For each provide:

### Short Interview Answer

and:

### Detailed Explanation

Use collapsible answers.

---

# 76. Knowledge Check Quiz

Create approximately **30 multiple-choice questions**.

Example:

```text
What is normally saved by:

torch.save(
    model.state_dict(),
    path
)

A. Model source code
B. Model parameters/buffers
C. Training dataset
D. Python interpreter
```

Correct:

```text
B
```

---

Example:

```text
You trained on CUDA and now need to load on CPU.

Which is useful?

A. shuffle=False
B. map_location="cpu"
C. model.train()
D. requires_grad=False
```

Correct:

```text
B
```

---

Example:

```text
What does model.eval() NOT do?

A. Disable Dropout
B. Change BatchNorm behavior
C. Disable Autograd
D. Switch module to evaluation behavior
```

Correct:

```text
C
```

---

Example:

```text
Why save optimizer.state_dict()?

A. It stores training labels
B. It stores optimizer configuration and internal state
C. It contains model architecture
D. It converts the model to inference mode
```

Correct:

```text
B
```

Put answers in a later section.

---

# 77. Write From Memory

Create approximately **25 prompts**.

Examples:

> Save a model's state_dict.

> Load a state_dict on CPU.

> Re-create a model and load saved weights.

> Put a loaded model in evaluation mode.

> Run inference without Autograd.

> Save model + optimizer + epoch in one checkpoint.

> Restore a checkpoint.

> Restore AdamW state.

> Calculate `start_epoch`.

> Save only when validation loss improves.

> Load with `strict=False`.

> Print missing and unexpected keys.

> Convert one input `(10,)` to batch shape `(1,10)`.

> Convert multiclass logits to predictions.

> Convert multiclass logits to probabilities.

> Convert binary logits to probabilities and predictions.

> Move inference outputs to CPU.

> Build a batch-inference loop.

> Preserve original inference order.

> Save preprocessing mean/std in checkpoint.

> Save class names in checkpoint.

Every coding exercise should have validation where reasonable.

---

# 78. Checkpoint Syntax Flashcards

Create approximately 15 rapid prompts.

Example:

```text
Get model weights?
```

Expected:

```python
model.state_dict()
```

```text
Load model weights?
```

Expected:

```python
model.load_state_dict(state)
```

```text
Save object?
```

Expected:

```python
torch.save(obj, path)
```

```text
Load on CPU?
```

Expected:

```python
torch.load(
    path,
    map_location="cpu"
)
```

```text
Inference mode?
```

Expected:

```python
with torch.inference_mode():
```

---

# 79. Final Challenge — Train, Save, Restore, Infer

Create a complete synthetic multiclass classification task.

Generate:

```python
torch.manual_seed(42)

X = torch.randn(1200, 20)

true_weights = torch.randn(
    20,
    4
)

y = (
    X @ true_weights
).argmax(
    dim=1
)
```

Split:

```text
80% training
20% validation
```

---

# 80. Final Challenge — Model

Require architecture:

```text
20
↓
64
↓ ReLU
↓
32
↓ ReLU
↓
4 logits
```

Use:

```python
class Classifier(nn.Module):
    ...
```

Also define a config dictionary:

```python
config = {
    "input_dim": 20,
    "hidden1": 64,
    "hidden2": 32,
    "num_classes": 4,
}
```

---

# 81. Final Challenge — Train

Use:

```text
CrossEntropyLoss
AdamW
```

Track:

- train loss
- validation loss
- validation accuracy

Train a modest number of epochs.

---

# 82. Final Challenge — Save Best Checkpoint

Checkpoint must include:

```text
model_state_dict
optimizer_state_dict
epoch
config
best_val_loss
class_names
```

Save only when validation loss improves.

Also save a final/last checkpoint separately.

---

# 83. Final Challenge — Simulate New Program

Create completely new objects:

```python
restored_model = ...
restored_optimizer = ...
```

Do not reuse the original model.

Load the best checkpoint.

Use:

```python
map_location=device
```

Restore configuration automatically.

---

# 84. Final Challenge — Verify Restoration

Compare original best model output with restored output on a fixed batch where practical.

Validate:

```python
assert torch.allclose(
    expected_logits,
    restored_logits,
    atol=1e-6
)
```

Make sure the original "best model" output is captured consistently at save time or reloaded from the same saved state rather than incorrectly comparing to the final epoch model.

---

# 85. Final Challenge — Resume Training

From the last checkpoint:

1. Load model.
2. Load optimizer.
3. Restore epoch.
4. Train two more epochs.

Validate:

```text
new start epoch
=
saved epoch + 1
```

and training proceeds without error.

---

# 86. Final Challenge — Batch Inference

Create an inference DataLoader.

Requirements:

- `shuffle=False`
- `model.eval()`
- `torch.inference_mode()`
- device transfer
- collect probabilities
- collect predicted classes
- move results to CPU
- concatenate results

Validate final shape:

```text
predictions:
(num_samples,)

probabilities:
(num_samples,4)
```

and:

```python
assert torch.allclose(
    probabilities.sum(dim=1),
    torch.ones(
        probabilities.size(0)
    ),
    atol=1e-5
)
```

---

# 87. Final Challenge — Prediction Labels

Use:

```python
class_names = [
    "class_A",
    "class_B",
    "class_C",
    "class_D",
]
```

Convert:

```text
prediction index
→ human-readable label
```

Show the first 10 predictions.

---

# 88. Final Challenge — Device Portability

Require saved checkpoint to load correctly with:

```python
map_location="cpu"
```

even if training occurred on CUDA/MPS.

Test restored CPU model with CPU tensor.

No checkpoint should be device-locked.

---

# 89. Final Challenge — Broken Checkpoint Debugging

Provide a second intentionally broken model:

```text
num_classes = 5
```

Attempt to load the 4-class checkpoint.

Ask learner to explain the resulting classifier mismatch.

Then show how to:

- preserve backbone weights
- leave a new 5-class head initialized

This should be optional medium/hard practice.

---

# 90. Final Cheat Sheet

End with a compact cheat sheet.

```python
# Save model weights
torch.save(
    model.state_dict(),
    "model.pt"
)
```

```python
# Load model weights
model = Model(...)

state = torch.load(
    "model.pt",
    map_location=device,
    weights_only=True
)

model.load_state_dict(state)

model.to(device)
model.eval()
```

Checkpoint:

```python
checkpoint = {
    "epoch": epoch,

    "model_state_dict":
        model.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "config":
        config,

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

model = Model(
    **checkpoint["config"]
).to(device)

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3
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

start_epoch = (
    checkpoint["epoch"] + 1
)
```

Inference:

```python
model.eval()

with torch.inference_mode():

    logits = model(X.to(device))

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predictions = logits.argmax(
        dim=1
    )
```

Binary:

```python
probs = torch.sigmoid(logits)

pred = (
    probs >= 0.5
).long()
```

---

# 91. Critical Mental Models

Include:

```text
WEIGHTS-ONLY SAVE

architecture code
+
state_dict
=
usable model
```

and:

```text
RESUME TRAINING

model weights
+
optimizer state
+
epoch/step
+
other training state
=
continuation
```

and:

```text
INFERENCE

load model
↓
model.eval()
↓
torch.inference_mode()
↓
same preprocessing as training
↓
forward pass
↓
postprocess logits
↓
CPU / user-facing output
```

---

# 92. Interview Quick Reference

Add:

```text
state_dict
→ named parameter/buffer tensors

torch.save
→ serialize checkpoint/state

torch.load
→ restore serialized state

load_state_dict
→ copy weights into model

map_location
→ control load device

strict=True
→ exact key matching expected

strict=False
→ allow missing/unexpected keys

model.eval()
→ evaluation behavior

inference_mode()
→ disable Autograd overhead

best checkpoint
→ best validation metric

last checkpoint
→ most recent training state
```

---

# 93. Completion Checklist

End with:

```text
## Before Moving to 14_pytorch_training_techniques.ipynb
```

Add:

- [ ] I understand what `state_dict` contains.
- [ ] I know that `state_dict` does not contain model architecture.
- [ ] I can save model weights.
- [ ] I can restore model weights.
- [ ] I can verify a save/load round trip.
- [ ] I can save optimizer state.
- [ ] I can create a full training checkpoint.
- [ ] I can resume training from a checkpoint.
- [ ] I can save the best validation model.
- [ ] I understand best vs last checkpoint.
- [ ] I understand `map_location`.
- [ ] I can load CUDA-trained weights on CPU.
- [ ] I understand `strict=True`.
- [ ] I understand `strict=False`.
- [ ] I can diagnose missing/unexpected keys.
- [ ] I understand classifier-head shape mismatches.
- [ ] I can perform multiclass inference.
- [ ] I can perform binary inference.
- [ ] I can perform regression inference.
- [ ] I understand `model.eval()` vs `inference_mode()`.
- [ ] I can run batch inference efficiently.
- [ ] I understand latency vs throughput.
- [ ] I know preprocessing must match training.
- [ ] I know class-index mappings should be preserved.
- [ ] I can package useful metadata with a checkpoint.

---

# 94. Difficulty Labels

Use:

```text
🟢 Easy
🟡 Medium
🔴 Hard
```

Easy:

- `state_dict`
- save/load weights
- inference mode
- binary/multiclass predictions

Medium:

- full checkpoints
- optimizer restoration
- `map_location`
- strict loading
- best-model checkpointing

Hard / optional:

- partial model loading
- classifier-head replacement
- checkpoint migration across architecture versions

Do not deeply cover yet:

- ONNX
- TorchServe
- TensorRT
- distributed checkpointing
- FSDP checkpoint formats
- Hugging Face model serialization
- cloud model registries
- production serving frameworks

Those can be covered in later advanced notebooks.

---

# 95. Quality Requirements

The notebook must:

- be a valid `.ipynb`
- execute successfully from top to bottom
- work fully on CPU
- optionally support CUDA/MPS
- require no internet access
- use only PyTorch and Python standard library
- create checkpoint files only in a temporary/notebook-safe directory
- clean up temporary files if appropriate
- use modern supported PyTorch loading APIs
- include concise security guidance around untrusted serialized checkpoints
- contain automatic validation
- contain approximately **70–90 exercises/questions**
- contain at least **25 checkpoint/inference debugging scenarios**
- include at least one full save → restore → resume flow
- include at least one full save → new process-style restore → inference flow
- emphasize practical AI Engineer interview knowledge
- avoid turning into a deployment-framework tutorial

The notebook should take approximately **3–4 hours** to study thoroughly.

The most important learning principle is:

```text
Training a model is not the end.

A practical PyTorch engineer must know how to:

train
↓
save
↓
restore
↓
resume
↓
evaluate
↓
infer
↓
move between devices
↓
preserve preprocessing/config
↓
debug checkpoint mismatches
```

By the end of this notebook, the learner should be able to answer an interview question such as:

> "How would you save a PyTorch model so that you can both deploy the best version and resume training later?"

with a clear explanation of **weights-only checkpoints, full training checkpoints, optimizer state, validation-based best-model saving, `map_location`, and inference mode**.

Finally save the completed notebook as:

```text
13_model_saving_loading_and_inference.ipynb
```