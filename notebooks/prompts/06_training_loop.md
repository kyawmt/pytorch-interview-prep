Create a Jupyter Notebook named:

```text
06_training_loop.ipynb
```

This notebook is the sixth part of a **PyTorch Interview Preparation** course for someone preparing for **AI Engineer / Machine Learning Engineer interviews**.

Assume the learner has already completed:

```text
01_tensor_basics.ipynb
02_tensor_operations.ipynb
03_autograd.ipynb
04_neural_networks.ipynb
05_losses_optimizers.ipynb
```

and already understands:

- tensors and shapes
- Autograd
- `nn.Module`
- neural-network layers
- logits
- loss functions
- SGD / Adam / AdamW
- `optimizer.zero_grad()`
- `loss.backward()`
- `optimizer.step()`

Do not spend significant time reteaching those topics.

The purpose of this notebook is to make the learner able to **write complete PyTorch training and evaluation loops from memory**, understand every line, track metrics correctly, switch between training and evaluation modes, and debug common training-loop problems.

This should be one of the most important notebooks in the entire course.

---

# Main Learning Goals

By the end of this notebook, the learner should be able to write from memory:

```python
model.train()

for X, y in train_loader:
    X = X.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    logits = model(X)

    loss = loss_fn(logits, y)

    loss.backward()

    optimizer.step()
```

and:

```python
model.eval()

with torch.no_grad():
    for X, y in val_loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)

        loss = loss_fn(logits, y)
```

The learner should also understand:

- epoch
- batch
- mini-batch training
- training loop
- validation loop
- `model.train()`
- `model.eval()`
- `torch.no_grad()`
- why validation must not update parameters
- loss aggregation
- accuracy calculation
- `loss.item()`
- batch-weighted average loss
- gradient flow
- device placement
- logging
- learning curves
- early stopping concept
- gradient clipping
- gradient accumulation
- training vs inference behavior
- reproducibility
- common loop bugs

Mark these as:

```text
🔥 Interview Essential
```

- training-loop order
- `model.train()`
- `model.eval()`
- `torch.no_grad()`
- `zero_grad()`
- `backward()`
- `step()`
- loss aggregation
- classification accuracy
- device handling
- train vs validation differences

---

# Notebook Structure

Use approximately:

```text
# PyTorch Training Loops

## 1. Setup
## 2. What Is a Training Loop?
## 3. Epochs and Batches
## 4. The Core Training Step
## 5. Training Mode
## 6. Forward Pass
## 7. Loss Calculation
## 8. Backward Pass
## 9. Parameter Update
## 10. Full Training Loop
## 11. Tracking Training Loss
## 12. Classification Accuracy
## 13. Evaluation Loop
## 14. model.train() vs model.eval()
## 15. torch.no_grad()
## 16. Validation Loss
## 17. Correct Loss Averaging
## 18. Device Handling
## 19. Training for Multiple Epochs
## 20. Logging Progress
## 21. Learning Curves
## 22. Gradient Clipping
## 23. Gradient Accumulation
## 24. Early Stopping
## 25. Reproducibility
## 26. Common Training Patterns
## 27. Common Mistakes
## 28. Debugging Exercises
## 29. Interview Questions
## 30. Knowledge Check Quiz
## 31. Write From Memory
## 32. Final Training Challenge
## 33. Cheat Sheet
```

---

# 1. Setup

Start with:

```python
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
```

Use only small synthetic datasets.

The notebook must:

- run on CPU
- finish quickly
- require no internet access

---

# 2. What Is a Training Loop?

Explain simply:

> A training loop repeatedly passes batches through the model, computes loss, calculates gradients, and updates parameters.

Visualize:

```text
Batch
  ↓
Model
  ↓
Prediction
  ↓
Loss
  ↓
Backward
  ↓
Gradients
  ↓
Optimizer Step
  ↓
Updated Model
```

Then repeat with the next batch.

---

# 3. Epoch vs Batch

Mark:

```text
🔥 Interview Essential
```

Explain:

```text
Dataset
= all training examples

Batch
= a subset processed together

Epoch
= one complete pass through the training dataset
```

Example:

```text
Dataset size = 1000
Batch size   = 100

Batches per epoch ≈ 10
```

Ask several quick questions.

Example:

```text
Dataset = 10,000 samples
Batch size = 250

How many full batches are there?
```

Expected:

```text
40
```

Mention the final batch can be smaller if the dataset size is not divisible by batch size.

---

# 4. Build a Small Dataset

Create:

```python
torch.manual_seed(42)

X = torch.randn(500, 10)

true_w = torch.randn(10, 3)

logits = X @ true_w

y = logits.argmax(dim=1)
```

Create:

```python
dataset = TensorDataset(X, y)
```

and:

```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

Do not deeply explain DataLoader because it will have its own notebook later.

Only explain enough to use it.

---

# 5. Build a Small Model

Use:

```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 3)
)
```

Loss:

```python
loss_fn = nn.CrossEntropyLoss()
```

Optimizer:

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3
)
```

---

# 6. The Core Training Step

Make this a major section.

Mark:

```text
🔥 Interview Essential
```

Show:

```python
optimizer.zero_grad()

logits = model(X_batch)

loss = loss_fn(logits, y_batch)

loss.backward()

optimizer.step()
```

Explain every line.

Use:

```text
zero_grad()
→ clear gradients from previous step

model(...)
→ forward pass

loss_fn(...)
→ calculate error

backward()
→ calculate gradients

step()
→ update parameters
```

---

# 7. Training-Order Exercise

Give these lines in the wrong order:

```python
loss.backward()
optimizer.step()
logits = model(X_batch)
optimizer.zero_grad()
loss = loss_fn(logits, y_batch)
```

Ask learner to reorder them correctly.

Expected:

```python
optimizer.zero_grad()

logits = model(X_batch)

loss = loss_fn(logits, y_batch)

loss.backward()

optimizer.step()
```

Include several variations.

---

# 8. Why `zero_grad()` Comes First

Briefly review gradient accumulation.

Explain:

```text
Without zero_grad():

old gradients
+
new gradients
```

unless accumulation is intentional.

Add one debugging question.

---

# 9. `model.train()`

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
model.train()
```

Explain:

> `model.train()` puts the model into training mode.

Important:

> It does NOT start training automatically.

It only changes the behavior of certain layers such as:

```text
Dropout
BatchNorm
```

This distinction must be emphasized.

Interview answer:

> `model.train()` sets training mode for modules like Dropout and BatchNorm; the actual training still happens through the forward, backward, and optimizer steps.

---

# 10. Full One-Epoch Training Loop

Teach:

```python
model.train()

for X_batch, y_batch in train_loader:
    optimizer.zero_grad()

    logits = model(X_batch)

    loss = loss_fn(logits, y_batch)

    loss.backward()

    optimizer.step()
```

Ask learner to write it from memory before showing the solution.

---

# 11. `loss.item()`

Teach:

```python
loss.item()
```

Explain:

> `loss` is a tensor. `.item()` converts a single-value tensor into a regular Python number.

Example:

```python
print(loss)
print(loss.item())
```

Explain this is useful for:

- logging
- accumulating metrics
- printing progress

Do not use `.item()` before `backward()` as a replacement for the loss tensor.

Show wrong conceptual pattern:

```python
loss_value = loss.item()
loss_value.backward()
```

Explain why it cannot work.

---

# 12. Tracking Training Loss

Start with simple accumulation:

```python
total_loss = 0.0

for X_batch, y_batch in train_loader:
    ...
    total_loss += loss.item()
```

Then explain that this gives average **batch loss** if divided by number of batches:

```python
average_loss = total_loss / len(train_loader)
```

But then introduce the more robust approach for differently sized final batches.

---

# 13. Correct Loss Averaging

Mark:

```text
🔥 Interview Useful
```

Show:

```python
total_loss = 0.0
total_samples = 0

for X_batch, y_batch in train_loader:
    ...
    batch_size = X_batch.size(0)

    total_loss += loss.item() * batch_size
    total_samples += batch_size

average_loss = total_loss / total_samples
```

Explain why:

> Most PyTorch losses use `reduction="mean"`, so `loss.item()` is the average loss for that batch. Multiplying by batch size converts it back to total loss contribution.

This matters when the final batch is smaller.

---

# 14. Classification Accuracy

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
predictions = logits.argmax(dim=1)

correct = (
    predictions == y_batch
).sum().item()
```

Accumulate:

```python
total_correct += correct
total_samples += y_batch.size(0)
```

Then:

```python
accuracy = total_correct / total_samples
```

Explain shape flow:

```text
logits:
(batch, num_classes)

argmax(dim=1):
(batch,)

targets:
(batch,)
```

---

# 15. Complete Training Metrics Loop

Show:

```python
model.train()

total_loss = 0.0
total_correct = 0
total_samples = 0

for X_batch, y_batch in train_loader:
    optimizer.zero_grad()

    logits = model(X_batch)

    loss = loss_fn(logits, y_batch)

    loss.backward()

    optimizer.step()

    batch_size = X_batch.size(0)

    total_loss += loss.item() * batch_size

    predictions = logits.argmax(dim=1)

    total_correct += (
        predictions == y_batch
    ).sum().item()

    total_samples += batch_size

train_loss = total_loss / total_samples
train_accuracy = total_correct / total_samples
```

Ask the learner to recreate this later.

---

# 16. Evaluation / Validation Loop

Mark:

```text
🔥 Interview Essential
```

Teach the standard pattern:

```python
model.eval()

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        logits = model(X_batch)

        loss = loss_fn(logits, y_batch)
```

Explain that evaluation has:

```text
NO
optimizer.zero_grad()

NO
loss.backward()

NO
optimizer.step()
```

This contrast must be very clear.

---

# 17. Training vs Validation Comparison

Create a table:

| Training | Validation |
|---|---|
| `model.train()` | `model.eval()` |
| forward pass | forward pass |
| loss | loss |
| `backward()` | no backward |
| `optimizer.step()` | no update |
| gradients needed | gradients not needed |
| Dropout active | Dropout disabled |
| BatchNorm uses batch behavior | BatchNorm evaluation behavior |

Mark:

```text
🔥 Interview Essential
```

---

# 18. `model.eval()`

Teach:

```python
model.eval()
```

Explain:

> `model.eval()` switches certain modules into evaluation behavior.

Examples:

```text
Dropout
→ disabled

BatchNorm
→ uses stored running statistics
```

Important:

> `model.eval()` does NOT disable gradient tracking.

This should be emphasized.

---

# 19. `torch.no_grad()`

Teach:

```python
with torch.no_grad():
    logits = model(X)
```

Explain:

> `torch.no_grad()` disables gradient tracking inside the block.

Benefits:

- lower memory usage
- less computation
- no unnecessary graph

Make clear:

```text
model.eval()
and
torch.no_grad()
```

solve different problems.

---

# 20. `model.eval()` vs `torch.no_grad()`

Make this a dedicated interview section.

Mark:

```text
🔥 Very Common Interview Question
```

Explain:

```text
model.eval()
→ changes module behavior

torch.no_grad()
→ disables gradient tracking
```

During validation, normally use both:

```python
model.eval()

with torch.no_grad():
    ...
```

Short interview answer:

> `model.eval()` changes layers like Dropout and BatchNorm to evaluation behavior, while `torch.no_grad()` disables Autograd. They do different things, so evaluation usually uses both.

---

# 21. Validation Metrics

Create a complete validation loop:

```python
model.eval()

total_loss = 0.0
total_correct = 0
total_samples = 0

with torch.no_grad():

    for X_batch, y_batch in val_loader:

        logits = model(X_batch)

        loss = loss_fn(logits, y_batch)

        batch_size = X_batch.size(0)

        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        total_correct += (
            predictions == y_batch
        ).sum().item()

        total_samples += batch_size

val_loss = total_loss / total_samples
val_accuracy = total_correct / total_samples
```

---

# 22. Build `train_one_epoch()`

Create a reusable function:

```python
def train_one_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device
):
    ...
```

Ask learner to implement it.

It should return:

```python
return average_loss, accuracy
```

Validation should confirm:

- parameters change
- loss is a Python float
- accuracy is between 0 and 1

---

# 23. Build `evaluate()`

Create:

```python
def evaluate(
    model,
    loader,
    loss_fn,
    device
):
    ...
```

Requirements:

- `model.eval()`
- `torch.no_grad()`
- no optimizer
- no backward
- calculate loss
- calculate accuracy

Return:

```python
return average_loss, accuracy
```

---

# 24. Device Handling

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
```

Then:

```python
model = model.to(device)
```

Inside loop:

```python
X_batch = X_batch.to(device)
y_batch = y_batch.to(device)
```

Explain:

> Model parameters and input tensors must generally be on the same device.

Show common error concept:

```text
model → CUDA
input → CPU
```

causes a device mismatch.

---

# 25. Device-Agnostic Loop

Use:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model.to(device)
```

Then:

```python
for X_batch, y_batch in train_loader:

    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    ...
```

Ensure the notebook still runs on CPU-only systems.

---

# 26. Optional Apple Silicon Support

Briefly mention a more flexible device choice:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

Do not require MPS or CUDA.

---

# 27. Multiple Epochs

Teach:

```python
for epoch in range(num_epochs):

    train_loss, train_acc = train_one_epoch(...)

    val_loss, val_acc = evaluate(...)
```

Explain:

```text
inner loop
→ batches

outer loop
→ epochs
```

Create a diagram:

```text
Epoch 1
 ├─ Batch 1
 ├─ Batch 2
 ├─ Batch 3
 └─ ...

Epoch 2
 ├─ Batch 1
 ├─ Batch 2
 └─ ...
```

---

# 28. Epoch Logging

Show:

```python
print(
    f"Epoch {epoch + 1}/{num_epochs} | "
    f"Train Loss: {train_loss:.4f} | "
    f"Train Acc: {train_acc:.4f} | "
    f"Val Loss: {val_loss:.4f} | "
    f"Val Acc: {val_acc:.4f}"
)
```

Explain that logging helps diagnose training behavior.

---

# 29. Store Learning History

Teach:

```python
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
}
```

Each epoch:

```python
history["train_loss"].append(train_loss)
```

etc.

---

# 30. Plot Learning Curves

Use only standard `matplotlib` if available.

If you want to avoid adding dependencies, check whether it is installed before using it.

Prefer including:

```python
import matplotlib.pyplot as plt
```

and plot:

```text
training loss
validation loss
```

Explain patterns conceptually.

If plotting adds dependency concerns, it is acceptable because Jupyter environments commonly have Matplotlib, but PyTorch remains the only deep-learning dependency.

---

# 31. Interpreting Learning Curves

Teach basic patterns.

### Healthy

```text
train loss ↓
val loss   ↓
```

### Overfitting

```text
train loss ↓
val loss   begins ↑
```

### Underfitting

```text
train loss remains high
val loss remains high
```

### Possible unstable learning rate

```text
loss oscillates strongly
or explodes
```

Do not make strong diagnoses from one metric alone.

---

# 32. Gradient Clipping

Mark:

```text
🔥 Interview Useful
```

Teach:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

Placement:

```python
optimizer.zero_grad()

logits = model(X_batch)

loss = loss_fn(logits, y_batch)

loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

optimizer.step()
```

Explain:

> Gradient clipping limits gradient magnitude before the optimizer update.

Commonly useful for:

- exploding gradients
- RNNs
- some transformer training situations

---

# 33. Gradient Clipping Order

Ask:

```text
Should clipping happen before or after backward()?
```

Answer:

```text
after backward()
before optimizer.step()
```

because gradients must first exist.

---

# 34. Gradient Accumulation

Mark:

```text
🟡 Medium
```

Explain use case:

> Simulate a larger effective batch size when GPU memory cannot fit the full batch.

Normal training:

```text
batch
→ backward
→ step
```

Accumulation:

```text
batch 1 → backward
batch 2 → backward
batch 3 → backward
batch 4 → backward
             ↓
            step
```

---

# 35. Gradient Accumulation Code

Teach:

```python
accumulation_steps = 4

optimizer.zero_grad()

for step, (X_batch, y_batch) in enumerate(train_loader):

    logits = model(X_batch)

    loss = loss_fn(logits, y_batch)

    loss = loss / accumulation_steps

    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Explain why:

```python
loss = loss / accumulation_steps
```

is commonly used to keep gradient scale approximately comparable to a larger averaged batch.

Mention the final partial accumulation window needs handling in robust production code.

---

# 36. Effective Batch Size

Explain:

```text
physical batch size = 16

accumulation steps = 4

effective batch size ≈ 64
```

assuming one device and standard accumulation.

Ask several quick questions.

---

# 37. Early Stopping

Introduce conceptually.

Explain:

> Early stopping stops training when validation performance stops improving for a specified number of epochs.

Important terms:

```text
best validation loss
patience
```

Example pseudocode:

```python
best_val_loss = float("inf")
patience = 3
bad_epochs = 0

for epoch in range(num_epochs):

    ...

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        bad_epochs = 0
    else:
        bad_epochs += 1

    if bad_epochs >= patience:
        break
```

Mention that a real implementation often saves/restores the best checkpoint.

---

# 38. Reproducibility

Teach:

```python
torch.manual_seed(42)
```

and briefly:

```python
torch.cuda.manual_seed_all(42)
```

when CUDA exists.

Explain:

> A seed makes random-number generation more reproducible, but exact reproducibility can still depend on hardware and algorithms.

Do not overpromise determinism.

---

# 39. Binary Classification Training Loop

Create a separate example.

Model output:

```text
(batch, 1)
```

Loss:

```python
nn.BCEWithLogitsLoss()
```

Prediction:

```python
probs = torch.sigmoid(logits)

pred = (probs >= 0.5)
```

Accuracy:

```python
correct = (
    pred.squeeze(1) == targets.bool()
)
```

or use a consistent target shape.

Emphasize binary training differs from multiclass primarily in:

- loss
- target format
- prediction conversion

---

# 40. Multiclass Training Loop

Use:

```python
nn.CrossEntropyLoss()
```

Prediction:

```python
pred = logits.argmax(dim=1)
```

This should be the primary training example.

---

# 41. Regression Training Loop

Create a short example using:

```python
nn.MSELoss()
```

Explain that there may be no accuracy metric.

Possible metrics:

```text
MSE
MAE
RMSE
```

Do not create a full metrics library.

---

# 42. Common Training Pattern Comparison

Create:

| Task | Loss | Prediction |
|---|---|---|
| Regression | MSE | raw model output |
| Binary | BCEWithLogits | sigmoid + threshold |
| Multiclass | CrossEntropy | argmax |

This reinforces Notebook 05.

---

# 43. Common Mistakes

Include at least these.

## Mistake 1 — Forgetting `model.train()`

Especially relevant with Dropout / BatchNorm.

## Mistake 2 — Forgetting `model.eval()`

## Mistake 3 — Forgetting `torch.no_grad()` during validation

## Mistake 4 — Calling `backward()` in validation

## Mistake 5 — Calling `optimizer.step()` in validation

## Mistake 6 — Forgetting `optimizer.zero_grad()`

## Mistake 7 — Wrong order

Wrong:

```python
optimizer.step()
loss.backward()
```

## Mistake 8 — Forgetting device transfer

## Mistake 9 — Model and tensors on different devices

## Mistake 10 — Averaging batch losses incorrectly

## Mistake 11 — Dividing accuracy by number of batches instead of number of samples

## Mistake 12 — Using `.item()` too early

## Mistake 13 — Calculating metrics from wrong dimension

## Mistake 14 — Forgetting to switch back to training mode after validation

## Mistake 15 — Calling `zero_grad()` after `backward()` but before `step()`

Wrong:

```python
loss.backward()

optimizer.zero_grad()

optimizer.step()
```

This clears the gradients before the optimizer can use them.

## Mistake 16 — Applying softmax before CrossEntropyLoss

Review briefly.

## Mistake 17 — Using validation data for parameter updates.

## Mistake 18 — Training indefinitely without tracking validation performance.

---

# 44. Debugging Exercises

Create at least 20 debugging exercises.

Example:

```python
model.eval()

for X, y in val_loader:
    logits = model(X)
    loss = loss_fn(logits, y)

    loss.backward()
```

Ask:

> What is wrong?

Expected:

> Validation should normally not perform backpropagation, and gradient tracking should usually be disabled.

---

Another:

```python
optimizer.zero_grad()

logits = model(X)

loss = loss_fn(logits, y)

loss.backward()

optimizer.zero_grad()

optimizer.step()
```

Ask why parameters do not update correctly.

---

Another:

```python
model.to(device)

for X, y in loader:
    logits = model(X)
```

Ask what may be missing.

Expected:

```python
X = X.to(device)
```

and targets as appropriate.

---

Another:

```python
total_loss += loss.item()

epoch_loss = total_loss / len(dataset)
```

Ask why this calculation may be incorrect when loss is mean-reduced per batch.

---

# 45. Exercise Format

Use cells like:

```python
# Exercise:
# Complete one training step.

optimizer.zero_grad()

logits = ...

loss = ...

...

...
```

Validation should check:

- loss exists
- gradients exist after backward
- parameters change after step

Where possible, compare model parameters before and after the step.

Example:

```python
before = [
    p.detach().clone()
    for p in model.parameters()
]
```

then validate that at least one changed.

---

# 46. Training Loop Fill-in-the-Blanks

Create at least 10.

Example:

```python
model.________()

for X, y in train_loader:

    optimizer.________()

    logits = model(X)

    loss = loss_fn(logits, y)

    loss.________()

    optimizer.________()
```

Expected:

```text
train
zero_grad
backward
step
```

---

# 47. Validation Loop Fill-in-the-Blanks

Example:

```python
model.________()

with torch.________():
    for X, y in val_loader:
        logits = model(X)
```

Expected:

```text
eval
no_grad
```

---

# 48. Interview Questions

Create approximately 35 questions.

Include:

1. What is a training loop?
2. What is an epoch?
3. What is a batch?
4. Why use mini-batches?
5. What does `model.train()` do?
6. Does `model.train()` start training automatically?
7. What does `model.eval()` do?
8. What is the difference between `model.train()` and `model.eval()`?
9. What does `torch.no_grad()` do?
10. `model.eval()` vs `torch.no_grad()`?
11. Why call `optimizer.zero_grad()`?
12. What does `loss.backward()` do?
13. What does `optimizer.step()` do?
14. What is the correct training-step order?
15. Why should validation not call backward?
16. Why should validation not update parameters?
17. Why use `torch.no_grad()` during validation?
18. How do you calculate multiclass predictions?
19. How do you calculate classification accuracy?
20. How do you correctly average loss across batches?
21. Why multiply batch mean loss by batch size?
22. What does `loss.item()` do?
23. Why should you not use `loss.item()` for backpropagation?
24. Why must the model and input be on the same device?
25. How do you move a model to GPU?
26. How do you move a batch to GPU?
27. What is gradient clipping?
28. When is gradient clipping useful?
29. What is gradient accumulation?
30. Why divide loss by accumulation steps?
31. What is effective batch size?
32. What is early stopping?
33. What does patience mean in early stopping?
34. What can training vs validation loss tell you about overfitting?
35. What could cause loss to explode?
36. What could cause training to be extremely slow?
37. Why set a random seed?

For each provide:

### Short Interview Answer

and:

### Detailed Explanation

Use collapsible sections.

---

# 49. Knowledge Check Quiz

Create approximately 30 multiple-choice questions.

Example:

```text
Which is the normal training order?

A.
forward → step → backward → zero_grad

B.
zero_grad → forward → loss → backward → step

C.
backward → zero_grad → forward → step

D.
step → forward → backward
```

Correct:

```text
B
```

Another:

```text
What does model.eval() do?

A. Disables Autograd
B. Updates parameters
C. Changes behavior of layers such as Dropout and BatchNorm
D. Deletes gradients
```

Correct:

```text
C
```

Another:

```text
What does torch.no_grad() do?

A. Switches Dropout off
B. Disables gradient tracking
C. Clears optimizer gradients
D. Updates parameters
```

Correct:

```text
B
```

Put answers in a separate section.

---

# 50. Write From Memory

Create approximately 20 prompts.

Examples:

> Write one complete training step.

> Put a model into training mode.

> Put a model into evaluation mode.

> Disable gradient tracking for validation.

> Calculate multiclass predictions.

> Calculate classification accuracy.

> Move a model to `device`.

> Move input and target to `device`.

> Write a complete one-epoch training loop.

> Write a complete validation loop.

> Accumulate sample-weighted average loss.

> Add gradient clipping before the optimizer update.

> Write basic gradient accumulation for 4 mini-batches.

> Write a basic multi-epoch train + validation loop.

Every practical prompt should include validation where possible.

---

# 51. 10-Minute Training Loop Challenge

Create:

```text
## 10-Minute Training Loop Challenge
```

Give a blank-ish training loop and ask the learner to complete everything without looking at notes.

Use:

```python
for epoch in range(num_epochs):

    # TRAIN
    ...

    # VALIDATE
    ...
```

Requirements:

- training mode
- validation mode
- device transfer
- zero gradients
- forward
- loss
- backward
- optimizer step
- no_grad
- loss tracking
- accuracy tracking

This should be treated as a major active-recall exercise.

---

# 52. Code Reading Exercise

Show:

```python
for epoch in range(10):

    model.train()

    for X, y in train_loader:

        optimizer.zero_grad()

        pred = model(X)

        loss = loss_fn(pred, y)

        loss.backward()

        optimizer.step()

    model.eval()

    with torch.no_grad():

        for X, y in val_loader:

            pred = model(X)

            val_loss = loss_fn(pred, y)
```

Ask the learner to explain each block verbally as if answering an interview question.

Provide a concise model answer later.

---

# 53. Final Training Challenge

Create a complete synthetic multiclass classification task.

Use reproducible data:

```python
torch.manual_seed(42)

num_samples = 1200
num_features = 20
num_classes = 4

X = torch.randn(
    num_samples,
    num_features
)

true_weights = torch.randn(
    num_features,
    num_classes
)

scores = X @ true_weights

y = scores.argmax(dim=1)
```

Split:

```text
80% training
20% validation
```

Do this using tensors only, with no external packages required.

Create DataLoaders.

---

# 54. Final Challenge Model

Require learner to build:

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
nn.Sequential
```

or custom `nn.Module`.

Accept either.

---

# 55. Final Challenge Requirements

Ask learner to implement:

### Part 1 — Device

Choose:

```text
CUDA if available
MPS if available
otherwise CPU
```

### Part 2 — Loss

Use correct multiclass loss.

### Part 3 — Optimizer

Use:

```text
AdamW
lr = 1e-3
weight_decay = 1e-2
```

### Part 4 — `train_one_epoch()`

Must:

- call `model.train()`
- move tensors to device
- zero gradients
- forward
- compute loss
- backward
- optimizer step
- calculate average loss
- calculate accuracy

### Part 5 — `evaluate()`

Must:

- call `model.eval()`
- use `torch.no_grad()`
- move tensors
- calculate loss
- calculate accuracy
- perform no updates

### Part 6 — Train for multiple epochs

Approximately:

```text
10–20 epochs
```

depending on runtime.

### Part 7 — Log results

Example:

```text
Epoch 01 | Train Loss ... | Train Acc ... | Val Loss ... | Val Acc ...
```

### Part 8 — Store history

Save all four metrics.

### Part 9 — Validate

Check:

```python
0 <= train_accuracy <= 1
0 <= val_accuracy <= 1
```

and verify that the training loop produces sensible learning progress.

Do not require an exact final accuracy, but because the labels are generated from a linear mapping, the model should generally learn meaningfully.

---

# 56. Optional Harder Challenge

Add:

```text
🔴 Hard
```

Ask learner to add:

```python
torch.nn.utils.clip_grad_norm_
```

and early stopping.

Do not make these required to complete the notebook.

---

# 57. Final Cheat Sheet

End with a compact cheat sheet.

Include:

```python
# Training
model.train()

for X, y in train_loader:

    X = X.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    logits = model(X)

    loss = loss_fn(logits, y)

    loss.backward()

    optimizer.step()
```

Then:

```python
# Validation
model.eval()

with torch.no_grad():

    for X, y in val_loader:

        X = X.to(device)
        y = y.to(device)

        logits = model(X)

        loss = loss_fn(logits, y)
```

Metrics:

```python
# Multiclass predictions
pred = logits.argmax(dim=1)

# Number correct
correct = (
    pred == y
).sum().item()

# Batch size
batch_size = y.size(0)

# Weighted loss accumulation
total_loss += loss.item() * batch_size

# Epoch average
avg_loss = total_loss / total_samples

accuracy = total_correct / total_samples
```

Device:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)
```

Gradient clipping:

```python
loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

optimizer.step()
```

---

# 58. Critical Mental Model

Include:

```text
TRAINING

model.train()
    ↓
batch
    ↓
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

and:

```text
VALIDATION

model.eval()
    ↓
torch.no_grad()
    ↓
batch
    ↓
forward
    ↓
loss + metrics

NO backward
NO step
```

And explicitly:

```text
model.eval()
≠
torch.no_grad()
```

---

# 59. Interview Quick Reference

Add:

```text
epoch
→ one pass through dataset

batch
→ subset of dataset

model.train()
→ training behavior

model.eval()
→ evaluation behavior

torch.no_grad()
→ disable Autograd

zero_grad()
→ clear previous gradients

backward()
→ calculate gradients

step()
→ update parameters

loss.item()
→ scalar Python value

gradient clipping
→ limit gradient magnitude

gradient accumulation
→ combine gradients across multiple mini-batches

early stopping
→ stop when validation stops improving
```

---

# 60. Completion Checklist

End with:

```text
## Before Moving to 07_dataset_dataloader.ipynb
```

Add:

- [ ] I can explain epoch vs batch.
- [ ] I can write one training step from memory.
- [ ] I can write a full training loop from memory.
- [ ] I understand `model.train()`.
- [ ] I understand `model.eval()`.
- [ ] I understand `torch.no_grad()`.
- [ ] I can explain `model.eval()` vs `torch.no_grad()`.
- [ ] I can calculate training loss.
- [ ] I can calculate validation loss.
- [ ] I can calculate multiclass accuracy.
- [ ] I can average batch loss correctly.
- [ ] I know how to move model and batches to a device.
- [ ] I can train for multiple epochs.
- [ ] I can log training and validation metrics.
- [ ] I understand learning curves.
- [ ] I understand gradient clipping.
- [ ] I understand gradient accumulation.
- [ ] I understand early stopping.
- [ ] I can debug common training-loop mistakes.
- [ ] I can implement reusable `train_one_epoch()` and `evaluate()` functions.

---

# 61. Difficulty Labels

Use:

```text
🟢 Easy
🟡 Medium
🔴 Hard
```

Most content should be Easy or Medium.

Medium:

- correct loss aggregation
- device-agnostic code
- gradient clipping
- learning-curve interpretation

Hard:

- gradient accumulation
- robust early stopping

Do not introduce yet:

- distributed training
- DDP
- FSDP
- DeepSpeed
- custom CUDA
- multi-GPU training
- mixed precision in depth

Those belong later.

---

# 62. Quality Requirements

The notebook must:

- be a valid `.ipynb`
- run successfully from top to bottom
- work on CPU-only systems
- require no internet
- use only small synthetic datasets
- contain real runnable training loops
- include automatic validation with `assert`
- contain approximately 70–90 exercises/questions
- heavily emphasize active recall
- heavily emphasize writing loops from memory
- heavily emphasize train vs evaluation behavior
- heavily emphasize debugging
- avoid excessive theoretical explanations

The notebook should take approximately **2–3 hours** to study thoroughly.

The most important learning loop is:

```text
Look at blank training loop
        ↓
write it from memory
        ↓
run it
        ↓
inspect loss
        ↓
inspect accuracy
        ↓
debug mistakes
        ↓
write it again without notes
```

By the end of this notebook, the learner should be able to write the basic PyTorch training and validation loops in an interview **without needing to search documentation**.

Finally save the notebook as:

```text
06_training_loop.ipynb
```