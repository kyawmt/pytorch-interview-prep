Create a Jupyter Notebook named:

```text
05_losses_optimizers.ipynb
```

This notebook is the fifth part of a **PyTorch Interview Preparation** course for someone preparing for **AI Engineer / Machine Learning Engineer interviews**.

Assume the learner has already completed:

```text
01_tensor_basics.ipynb
02_tensor_operations.ipynb
03_autograd.ipynb
04_neural_networks.ipynb
```

and already understands:

- tensors and shapes
- broadcasting
- Autograd
- `.backward()`
- gradient accumulation
- `nn.Module`
- `nn.Linear`
- activations
- logits
- model parameters
- basic MLP/CNN structure

Do not spend significant time reteaching those topics.

The purpose of this notebook is to teach **loss functions and optimizers**, how to choose them, how their inputs must be shaped, and how they fit into PyTorch training.

---

# Main Learning Goals

By the end of this notebook, the learner should confidently understand and use:

```python
nn.MSELoss()
nn.L1Loss()

nn.BCELoss()
nn.BCEWithLogitsLoss()

nn.CrossEntropyLoss()

torch.optim.SGD()
torch.optim.Adam()
torch.optim.AdamW()
```

The learner should also understand:

- what a loss function does
- regression vs classification losses
- logits vs probabilities
- binary vs multiclass classification
- target shapes and target dtypes
- why `CrossEntropyLoss` expects raw logits
- why `BCEWithLogitsLoss` is preferred over manual sigmoid + BCE
- `reduction="mean"`, `"sum"`, and `"none"`
- class weighting
- learning rate
- momentum
- weight decay
- SGD vs Adam vs AdamW
- `optimizer.zero_grad()`
- `loss.backward()`
- `optimizer.step()`
- parameter groups
- optimizer state
- how to inspect and change learning rate
- common loss/optimizer bugs

Mark these as:

```text
🔥 Interview Essential
```

- MSELoss
- CrossEntropyLoss
- BCEWithLogitsLoss
- logits
- target shape
- target dtype
- learning rate
- SGD vs Adam
- Adam vs AdamW
- weight decay
- `zero_grad() → backward() → step()`

---

# Notebook Structure

Use approximately:

```text
# PyTorch Loss Functions and Optimizers

## 1. Setup
## 2. What Is a Loss Function?
## 3. Regression vs Classification
## 4. MSELoss
## 5. L1Loss
## 6. Binary Classification Losses
## 7. BCELoss
## 8. BCEWithLogitsLoss
## 9. Multiclass Classification
## 10. CrossEntropyLoss
## 11. Target Shapes and Dtypes
## 12. Loss Reduction
## 13. Class Weights
## 14. What Is an Optimizer?
## 15. Learning Rate
## 16. SGD
## 17. Momentum
## 18. Adam
## 19. AdamW
## 20. Weight Decay
## 21. SGD vs Adam vs AdamW
## 22. zero_grad / backward / step
## 23. Optimizer Parameter Groups
## 24. Inspecting Optimizer State
## 25. Changing Learning Rate
## 26. Common Training Patterns
## 27. Common Mistakes
## 28. Debugging Exercises
## 29. Interview Questions
## 30. Knowledge Check Quiz
## 31. Write From Memory
## 32. Final Challenge
## 33. Cheat Sheet
```

---

# 1. Setup

Start with:

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
```

Use CPU-only examples.

---

# 2. What Is a Loss Function?

Explain:

> A loss function measures how wrong the model's predictions are compared with the target.

Show:

```text
model input
    ↓
model
    ↓
prediction
    ↓
loss(prediction, target)
    ↓
single value measuring error
```

Explain that training tries to minimize this loss.

---

# 3. Regression vs Classification

Create a comparison table:

| Task | Typical Output | Common Loss |
|---|---|---|
| Regression | continuous value | `MSELoss` |
| Binary classification | one logit per sample | `BCEWithLogitsLoss` |
| Multiclass classification | one logit per class | `CrossEntropyLoss` |

Mark this:

```text
🔥 Interview Essential
```

---

# 4. MSELoss

Teach:

```python
loss_fn = nn.MSELoss()
```

Example:

```python
pred = torch.tensor([2.0, 4.0, 6.0])
target = torch.tensor([1.0, 5.0, 7.0])

loss = loss_fn(pred, target)

print(loss)
```

Explain:

```text
MSE = mean((prediction - target)^2)
```

Ask learner to manually calculate before running.

Add exercises with:

```python
pred = ...
target = ...
```

and validate using `torch.allclose`.

---

# 5. MSE Shape Requirement

Explain that prediction and target should usually have compatible/matching shapes.

Example:

```text
prediction: (32, 1)
target:     (32, 1)
```

Warn about accidental broadcasting:

```text
prediction: (32, 1)
target:     (32,)
```

Explain that PyTorch may broadcast them in an unintended way.

Create debugging questions around this.

---

# 6. L1Loss

Teach:

```python
nn.L1Loss()
```

Explain:

```text
L1 loss
= mean absolute error
```

Compare:

```text
MSE
→ squares large errors
→ more sensitive to outliers

L1
→ absolute error
→ less sensitive to large outliers
```

Keep this section concise.

---

# 7. Binary Classification

Mark:

```text
🔥 Interview Essential
```

Explain common model output:

```text
(batch, 1)
```

or sometimes:

```text
(batch,)
```

Each sample gets one raw score:

```text
logit
```

Example:

```python
logits = torch.tensor([
    [-2.0],
    [0.5],
    [3.0]
])
```

Explain:

```text
negative large → probability closer to 0
0              → probability 0.5
positive large → probability closer to 1
```

---

# 8. Sigmoid Review

Show:

```python
probs = torch.sigmoid(logits)
```

Explain:

```text
logit
↓ sigmoid
probability between 0 and 1
```

But immediately introduce the preferred training pattern:

```text
raw logits
↓
BCEWithLogitsLoss
```

---

# 9. BCELoss

Teach:

```python
nn.BCELoss()
```

Explain that it expects probabilities rather than raw logits.

Example:

```python
loss_fn = nn.BCELoss()

probs = torch.tensor([
    [0.9],
    [0.2],
    [0.7]
])

targets = torch.tensor([
    [1.0],
    [0.0],
    [1.0]
])

loss = loss_fn(probs, targets)
```

Explain that probabilities should be between 0 and 1.

---

# 10. BCEWithLogitsLoss

Make this one of the biggest sections.

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
loss_fn = nn.BCEWithLogitsLoss()
```

Example:

```python
logits = torch.tensor([
    [2.0],
    [-1.0],
    [0.5]
])

targets = torch.tensor([
    [1.0],
    [0.0],
    [1.0]
])

loss = loss_fn(logits, targets)
```

Explain:

> `BCEWithLogitsLoss` combines sigmoid and binary cross entropy in one numerically stable operation.

Show conceptually:

```text
logits
↓
sigmoid
↓
binary cross entropy
```

but explain PyTorch performs this more safely internally.

---

# 11. Why Not Sigmoid Before BCEWithLogitsLoss?

Show incorrect pattern:

```python
probs = torch.sigmoid(logits)

loss = nn.BCEWithLogitsLoss()(probs, targets)
```

Explain:

> This applies sigmoid twice conceptually and gives the loss the wrong type of input.

Correct:

```python
loss = nn.BCEWithLogitsLoss()(logits, targets)
```

Mark:

```text
🔥 Very Common Interview Question
```

---

# 12. Binary Prediction at Inference

Teach:

```python
probs = torch.sigmoid(logits)

predictions = (probs >= 0.5).long()
```

Explain:

```text
training:
logits → BCEWithLogitsLoss

inference:
logits → sigmoid → threshold
```

Create exercises.

---

# 13. Multiclass Classification

Mark:

```text
🔥 Interview Essential
```

Explain model output:

```text
(batch_size, num_classes)
```

Example:

```text
32 samples
10 classes

logits shape:
(32, 10)
```

Target:

```text
one class index per sample

target shape:
(32,)
```

Example:

```python
targets = torch.tensor([2, 0, 1, 2])
```

---

# 14. CrossEntropyLoss

Teach:

```python
loss_fn = nn.CrossEntropyLoss()
```

Example:

```python
logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3],
    [0.1, 0.2, 3.0]
])

targets = torch.tensor([0, 1, 2])

loss = loss_fn(logits, targets)

print(loss)
```

Explain expected shapes:

```text
logits:
(batch, classes)

targets:
(batch,)
```

and target values:

```text
0 <= target < num_classes
```

---

# 15. CrossEntropyLoss Target Dtype

Mark:

```text
🔥 Interview Essential
```

Teach that class-index targets typically need:

```python
torch.long
```

Example:

```python
targets = torch.tensor(
    [0, 2, 1],
    dtype=torch.long
)
```

Show wrong:

```python
targets = torch.tensor(
    [0.0, 2.0, 1.0]
)
```

Ask why this causes an issue in the normal class-index use case.

---

# 16. Why No Softmax Before CrossEntropyLoss?

Make this very prominent.

Wrong:

```python
probs = torch.softmax(logits, dim=1)

loss = nn.CrossEntropyLoss()(probs, targets)
```

Normal correct pattern:

```python
loss = nn.CrossEntropyLoss()(logits, targets)
```

Explain:

> `CrossEntropyLoss` expects raw logits and internally combines the operations needed for log-softmax and negative log likelihood in a numerically stable way.

Short interview answer:

> Pass raw logits directly to `CrossEntropyLoss`; don't manually softmax them first.

---

# 17. CrossEntropyLoss Prediction

Teach:

```python
predictions = logits.argmax(dim=1)
```

Explain:

```text
logits:
(32, 10)

argmax(dim=1):
(32,)
```

Then:

```python
accuracy = (
    predictions == targets
).float().mean()
```

Add coding exercises.

---

# 18. Binary vs Multiclass Cheat Table

Create:

| Feature | Binary | Multiclass |
|---|---|---|
| Output | `(batch, 1)` | `(batch, classes)` |
| Final raw output | logit | logits |
| Loss | `BCEWithLogitsLoss` | `CrossEntropyLoss` |
| Target dtype | usually float | long/int64 |
| Inference | sigmoid + threshold | argmax |
| Manual sigmoid/softmax before loss? | No | No |

Mark as high priority.

---

# 19. Multilabel Classification

Introduce briefly because it is frequently confused with multiclass classification.

Explain:

```text
Multiclass:
one class out of many

Multilabel:
multiple classes can be true simultaneously
```

For multilabel, commonly:

```python
nn.BCEWithLogitsLoss()
```

with output:

```text
(batch, num_labels)
```

and float target matrix:

```text
(batch, num_labels)
```

Example:

```text
[1, 0, 1, 0]
```

Do not make this section too long.

---

# 20. Loss Reduction

Teach:

```python
nn.MSELoss(reduction="mean")
nn.MSELoss(reduction="sum")
nn.MSELoss(reduction="none")
```

Explain:

```text
mean
→ average losses

sum
→ add losses

none
→ return individual losses
```

Demonstrate:

```python
pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([0.0, 2.0, 5.0])
```

and compare output shapes.

Explain why `"none"` can be useful for:

- custom weighting
- inspecting per-sample loss
- masking

---

# 21. Class Weights

Teach basic multiclass weighting:

```python
weights = torch.tensor([1.0, 2.0, 4.0])

loss_fn = nn.CrossEntropyLoss(
    weight=weights
)
```

Explain:

> Higher weight makes mistakes on that class contribute more strongly to the loss.

Mention this may help with class imbalance.

Do not present it as a universal solution to imbalance.

---

# 22. `pos_weight` in BCEWithLogitsLoss

Introduce:

```python
nn.BCEWithLogitsLoss(
    pos_weight=...
)
```

Explain at a practical level that it can increase the contribution of positive examples in imbalanced binary/multilabel problems.

Keep details moderate.

---

# 23. Loss Function Selection Exercises

Create at least 15 questions.

Examples:

### Scenario 1

Predict house price.

Expected:

```text
MSELoss or L1Loss
```

### Scenario 2

Cat vs dog.

Model output:

```text
(batch, 1)
```

Expected:

```text
BCEWithLogitsLoss
```

### Scenario 3

Classify image among 100 classes.

Expected:

```text
CrossEntropyLoss
```

### Scenario 4

Image can contain any combination of:

```text
dog
car
person
tree
```

Expected:

```text
BCEWithLogitsLoss
```

These should heavily test task → output → target → loss mapping.

---

# 24. What Is an Optimizer?

Explain:

> An optimizer uses parameter gradients to update model parameters in a direction intended to reduce the loss.

Show flow:

```text
loss.backward()
      ↓
parameter.grad
      ↓
optimizer.step()
      ↓
updated parameters
```

Make clear again:

```text
backward()
→ calculates gradients

step()
→ changes parameters
```

---

# 25. Creating an Optimizer

Teach:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)
```

and:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Explain:

```python
model.parameters()
```

tells the optimizer which trainable tensors to update.

---

# 26. Learning Rate

Mark:

```text
🔥 Interview Essential
```

Explain:

> Learning rate controls the size of each parameter update.

Conceptual diagram:

```text
too small
→ training very slow

reasonable
→ stable progress

too large
→ unstable / divergence
```

Do not imply one universally correct learning rate.

Common starting points can be mentioned cautiously:

```text
SGD: often around 1e-2 or 1e-1 depending on problem
Adam/AdamW: often around 1e-3
```

State clearly that these are heuristics, not rules.

---

# 27. SGD

Teach:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)
```

Explain simple idea:

```text
new parameter
=
old parameter
-
learning_rate × gradient
```

Connect back to manual gradient descent from the Autograd notebook.

---

# 28. SGD With Momentum

Teach:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

Explain intuitively:

> Momentum keeps a running direction from previous gradients, which can smooth updates and speed progress through consistent directions.

Do not derive equations unless brief.

---

# 29. Adam

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Explain intuitively:

> Adam maintains adaptive learning-rate information for each parameter using running estimates of gradient statistics.

Avoid overly mathematical explanations.

Mention advantages:

- convenient default in many deep-learning tasks
- often converges quickly
- less learning-rate tuning than plain SGD in many cases

Do not claim it is always superior.

---

# 30. AdamW

Mark:

```text
🔥 Interview Essential for modern AI
```

Teach:

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

Explain:

> AdamW uses decoupled weight decay, which handles weight decay more cleanly than adding ordinary L2 regularization into Adam's gradient update.

Mention:

```text
Transformers / modern deep learning
→ AdamW is very common
```

---

# 31. Adam vs AdamW

Create a concise comparison.

Explain:

```text
Adam
→ adaptive optimizer

AdamW
→ Adam-style optimization with decoupled weight decay
```

Interview answer:

> If weight decay is being used, AdamW is generally preferred because its weight decay is decoupled from Adam's adaptive gradient update.

Do not overstate that AdamW is always best.

---

# 32. Weight Decay

Mark:

```text
🔥 Interview Essential
```

Explain simply:

> Weight decay discourages parameters from becoming unnecessarily large and acts as a form of regularization.

Show:

```python
optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2
)
```

Explain difference between:

```text
learning rate
→ update size

weight decay
→ regularization on parameter magnitude
```

---

# 33. SGD vs Adam vs AdamW

Create an interview-oriented table:

| Optimizer | Main Idea | Common Use |
|---|---|---|
| SGD | direct gradient updates | classical vision / tuned training |
| SGD + momentum | smoother SGD | CNNs, large-scale training |
| Adam | adaptive updates | broad deep-learning use |
| AdamW | Adam + decoupled weight decay | transformers / modern AI |

Add caveat:

> Optimizer choice is problem-dependent; there is no universally best optimizer.

---

# 34. The Core Training Sequence

Make this one of the most important sections.

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
optimizer.zero_grad()

pred = model(X)

loss = loss_fn(pred, y)

loss.backward()

optimizer.step()
```

Explain each line:

```text
zero_grad()
→ clear old accumulated gradients

model(X)
→ forward pass

loss_fn(...)
→ calculate loss

backward()
→ compute gradients

step()
→ update parameters
```

Have learner write this sequence repeatedly from memory.

---

# 35. Why `zero_grad()` Comes Before `backward()`

Explain gradient accumulation again briefly.

Show:

```text
previous gradients
+
current gradients
```

if gradients are not cleared.

Mention intentional gradient accumulation exists, but standard independent minibatch training usually clears gradients each step.

---

# 36. Does `optimizer.step()` Clear Gradients?

Ask explicitly.

Answer:

```text
No.
```

Explain:

> `optimizer.step()` updates parameters but does not normally clear their `.grad` values.

This is a useful interview question.

---

# 37. Minimal Regression Training Example

Create:

```python
X = torch.randn(100, 1)
y = 3 * X + 2 + 0.1 * torch.randn(100, 1)
```

Model:

```python
model = nn.Linear(1, 1)
```

Loss:

```python
loss_fn = nn.MSELoss()
```

Optimizer:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.05
)
```

Train for a modest number of epochs.

Show that learned:

```text
weight ≈ 3
bias ≈ 2
```

Use tolerance instead of exact values.

---

# 38. Minimal Binary Classification Example

Create synthetic data.

Model:

```python
model = nn.Linear(2, 1)
```

Loss:

```python
nn.BCEWithLogitsLoss()
```

Optimizer:

```python
optim.Adam(...)
```

Show:

```text
training:
logits → BCEWithLogitsLoss

prediction:
sigmoid → threshold
```

Keep dataset simple and runtime short.

---

# 39. Minimal Multiclass Example

Create synthetic input:

```python
X = torch.randn(64, 10)
targets = torch.randint(0, 3, (64,))
```

Model:

```python
model = nn.Linear(10, 3)
```

Loss:

```python
nn.CrossEntropyLoss()
```

Show the expected shapes:

```text
model output:
(64, 3)

target:
(64,)
```

Run one or several training steps.

---

# 40. Optimizer Parameter Groups

Introduce at an intermediate level:

```python
optimizer = optim.AdamW([
    {
        "params": model.backbone.parameters(),
        "lr": 1e-4
    },
    {
        "params": model.classifier.parameters(),
        "lr": 1e-3
    }
])
```

Explain:

> Parameter groups let different sets of parameters use different optimization settings.

Common uses:

- smaller LR for pretrained backbone
- larger LR for new classifier head
- different weight decay settings

Do not make this a major section.

---

# 41. Inspecting Learning Rate

Teach:

```python
optimizer.param_groups
```

Example:

```python
for group in optimizer.param_groups:
    print(group["lr"])
```

Ask learner to modify LR:

```python
for group in optimizer.param_groups:
    group["lr"] = 1e-4
```

Mention schedulers will be covered later.

---

# 42. Optimizer State

Briefly teach:

```python
optimizer.state_dict()
```

Explain it contains optimizer configuration and internal state such as momentum/adaptive statistics.

Connect to checkpointing:

```text
model.state_dict()
optimizer.state_dict()
```

but leave full saving/loading to a later notebook.

---

# 43. Zeroing Gradients With `set_to_none=True`

Introduce as optional/intermediate:

```python
optimizer.zero_grad(
    set_to_none=True
)
```

Explain briefly that gradients can be reset to `None` rather than explicit zeros and that this can have performance/memory advantages.

Do not make this essential for beginners.

---

# 44. Loss Shape Exercises

Create at least 15 shape/dtype exercises.

Examples:

### Binary

```text
logits:  (32, 1)
targets: (32, 1), float

loss?
```

Expected:

```text
BCEWithLogitsLoss
```

### Multiclass

```text
logits:  (32, 10)
targets: (32,), long

loss?
```

Expected:

```text
CrossEntropyLoss
```

### Regression

```text
pred:   (64, 1)
target: (64, 1)

loss?
```

Expected:

```text
MSELoss
```

---

# 45. Common Mistakes

Include at least these.

## Mistake 1 — Softmax before CrossEntropyLoss

Wrong:

```python
loss = loss_fn(
    torch.softmax(logits, dim=1),
    targets
)
```

## Mistake 2 — Sigmoid before BCEWithLogitsLoss

## Mistake 3 — Float class targets with CrossEntropyLoss

## Mistake 4 — Long targets with binary BCE setup

## Mistake 5 — Wrong output shape

Binary model returning:

```text
(batch, 2)
```

while configured for one-logit BCE without intending two-output formulation.

## Mistake 6 — Target and prediction shape mismatch

## Mistake 7 — Forgetting `optimizer.zero_grad()`

## Mistake 8 — Forgetting `loss.backward()`

## Mistake 9 — Forgetting `optimizer.step()`

## Mistake 10 — Wrong training order

## Mistake 11 — Optimizer created without correct model parameters

## Mistake 12 — Learning rate far too large

## Mistake 13 — Assuming loss automatically changes parameters

## Mistake 14 — Confusing weight decay and learning rate

## Mistake 15 — Using multiclass CrossEntropyLoss for multilabel classification

---

# 46. Debugging Exercises

Create at least 20 debugging exercises.

Example:

```python
logits = torch.randn(32, 10)

targets = torch.randint(
    0,
    10,
    (32,)
).float()

loss = nn.CrossEntropyLoss()(
    logits,
    targets
)
```

Ask:

> What is wrong?

Expected:

```text
targets should normally be torch.long class indices.
```

---

Another:

```python
logits = model(X)

probs = torch.sigmoid(logits)

loss = nn.BCEWithLogitsLoss()(
    probs,
    targets
)
```

Ask what is wrong.

---

Another:

```python
optimizer.zero_grad()
loss = loss_fn(model(X), y)
optimizer.step()
```

Ask which important step is missing.

Expected:

```python
loss.backward()
```

---

Another:

```python
loss.backward()
optimizer.zero_grad()
optimizer.step()
```

Ask why this order is wrong.

Expected:

> Gradients are cleared after being computed and before the parameter update.

---

# 47. Exercise Format

Use cells such as:

```python
# Exercise:
# Create the correct loss function for
# multiclass classification.

loss_fn = None
```

Validation:

```python
assert isinstance(
    loss_fn,
    nn.CrossEntropyLoss
), "Use the standard multiclass classification loss."

print("✅ Correct!")
```

For optimizers:

```python
optimizer = None
```

Validation should check:

- optimizer type
- learning rate
- correct parameters where practical

---

# 48. Interview Questions

Create approximately 30–35 questions.

Include:

1. What does a loss function do?
2. What loss would you use for regression?
3. What is MSELoss?
4. MSE vs L1?
5. What is a logit?
6. What is BCEWithLogitsLoss?
7. Why is BCEWithLogitsLoss preferred to sigmoid + BCELoss?
8. What target dtype is usually used with BCEWithLogitsLoss?
9. How do you convert binary logits to predictions?
10. What is CrossEntropyLoss used for?
11. What shape should multiclass logits have?
12. What shape should multiclass class-index targets have?
13. What dtype should CrossEntropyLoss targets normally have?
14. Why shouldn't you apply softmax before CrossEntropyLoss?
15. What is the difference between multiclass and multilabel classification?
16. Which loss is commonly used for multilabel classification?
17. What does `reduction="none"` do?
18. How can class weighting help with imbalance?
19. What does an optimizer do?
20. What is learning rate?
21. What happens if learning rate is too high?
22. What happens if learning rate is too low?
23. What is SGD?
24. What does momentum do?
25. What is Adam?
26. What is AdamW?
27. Adam vs AdamW?
28. What is weight decay?
29. Weight decay vs learning rate?
30. Why do we call `optimizer.zero_grad()`?
31. What does `loss.backward()` do?
32. What does `optimizer.step()` do?
33. Does `optimizer.step()` clear gradients?
34. What are optimizer parameter groups?
35. Why might different model layers use different learning rates?

For each provide:

### Short Interview Answer

and:

### Detailed Explanation

Use collapsible sections.

---

# 49. Knowledge Check Quiz

Create approximately 25–30 multiple-choice questions.

Example:

```text
A multiclass classifier produces:

logits.shape = (64, 10)

targets.shape = (64,)

Which loss should normally be used?

A. MSELoss
B. BCELoss
C. BCEWithLogitsLoss
D. CrossEntropyLoss
```

Correct:

```text
D
```

Another:

```text
Which sequence is correct?

A.
step → backward → zero_grad

B.
zero_grad → backward → step

C.
backward → step → zero_grad

D.
step → zero_grad → backward
```

Expected:

```text
B
```

Clarify that the forward/loss calculation occurs between `zero_grad` and `backward`.

---

# 50. Write From Memory

Create approximately 20 prompts.

Examples:

> Create MSE loss.

> Create CrossEntropyLoss.

> Create BCEWithLogitsLoss.

> Convert binary logits to probabilities.

> Convert binary probabilities to 0/1 predictions.

> Convert multiclass logits to predicted classes.

> Create SGD with learning rate `0.01`.

> Create SGD with momentum `0.9`.

> Create Adam with learning rate `1e-3`.

> Create AdamW with learning rate `1e-4` and weight decay `0.01`.

> Write the standard five-line training step.

> Inspect optimizer learning rates.

> Change optimizer LR to `1e-4`.

Each should have executable validation where reasonable.

---

# 51. Task-to-Loss Challenge

Create a dedicated rapid challenge.

Provide 15 scenarios and ask for:

```text
1. output shape
2. output activation at training time
3. target shape
4. target dtype
5. loss function
6. inference conversion
```

Example:

```text
Task:
Classify one image into one of 100 classes.
```

Expected:

```text
output:
(batch, 100) raw logits

target:
(batch,) long

loss:
CrossEntropyLoss

inference:
argmax(dim=1)
```

This should be one of the most valuable sections in the notebook.

---

# 52. Optimizer Selection Challenge

Give scenarios such as:

> Training a transformer with weight decay.

Likely answer:

```text
AdamW
```

> Standard CNN training where carefully tuned SGD is desired.

Likely:

```text
SGD + momentum
```

> Need a convenient adaptive optimizer for a small MLP.

Likely:

```text
Adam or AdamW
```

Make clear that optimizer selection often has multiple reasonable answers.

Do not mark subjective choices as uniquely correct unless appropriate.

---

# 53. Final Challenge — Complete Training Components

Build a small 3-class classification setup.

Data:

```python
torch.manual_seed(42)

X = torch.randn(128, 10)
y = torch.randint(0, 3, (128,))
```

Model:

```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 3)
)
```

Ask learner to fill in:

### Step 1

Correct loss:

```python
loss_fn = ...
```

### Step 2

Optimizer:

```python
optimizer = ...
```

Use AdamW with:

```text
lr = 1e-3
weight_decay = 1e-2
```

### Step 3

Forward:

```python
logits = ...
```

### Step 4

Loss:

```python
loss = ...
```

### Step 5

Clear gradients.

### Step 6

Backward.

### Step 7

Optimizer step.

### Step 8

Prediction:

```python
pred = ...
```

### Step 9

Accuracy.

Run for enough epochs to prove the code functions, but do not require high accuracy because labels are random.

The goal is correctness of the training mechanics, not model performance.

---

# 54. Final Cheat Sheet

End with a compact cheat sheet.

```python
# Regression
loss_fn = nn.MSELoss()
loss_fn = nn.L1Loss()

# Binary classification
loss_fn = nn.BCEWithLogitsLoss()

logits = model(X)
loss = loss_fn(logits, targets)

probs = torch.sigmoid(logits)
pred = (probs >= 0.5).long()

# Multiclass classification
loss_fn = nn.CrossEntropyLoss()

logits = model(X)
loss = loss_fn(logits, targets)

pred = logits.argmax(dim=1)

# SGD
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)

# SGD + momentum
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)

# Adam
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)

# AdamW
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2
)

# Standard training step
optimizer.zero_grad()

pred = model(X)

loss = loss_fn(pred, y)

loss.backward()

optimizer.step()

# Inspect learning rate
for group in optimizer.param_groups:
    print(group["lr"])

# Change learning rate
for group in optimizer.param_groups:
    group["lr"] = 1e-4
```

---

# 55. Critical Mental Model

Include:

```text
Regression
model → continuous prediction → MSE/L1

Binary classification
model → one raw logit
      → BCEWithLogitsLoss

Multiclass classification
model → one raw logit per class
      → CrossEntropyLoss
```

And:

```text
zero_grad()
     ↓
forward
     ↓
loss
     ↓
backward()
     ↓
gradients
     ↓
step()
     ↓
updated parameters
```

---

# 56. Interview Quick Reference

Add:

```text
MSELoss
→ regression

BCEWithLogitsLoss
→ binary / multilabel classification

CrossEntropyLoss
→ multiclass classification

SGD
→ simple gradient-based optimizer

Adam
→ adaptive optimizer

AdamW
→ Adam with decoupled weight decay

learning rate
→ parameter update size

weight decay
→ regularization

backward()
→ calculate gradients

step()
→ update parameters

zero_grad()
→ clear old gradients
```

---

# 57. Completion Checklist

End with:

```text
## Before Moving to 06_training_loop.ipynb
```

Add:

- [ ] I can choose a loss for regression.
- [ ] I can choose a loss for binary classification.
- [ ] I can choose a loss for multiclass classification.
- [ ] I understand multiclass vs multilabel.
- [ ] I understand logits.
- [ ] I know why CrossEntropyLoss should receive raw logits.
- [ ] I know why BCEWithLogitsLoss should receive raw logits.
- [ ] I understand target shapes.
- [ ] I understand target dtypes.
- [ ] I understand loss reduction.
- [ ] I understand class weighting at a basic level.
- [ ] I understand what an optimizer does.
- [ ] I understand learning rate.
- [ ] I can use SGD.
- [ ] I understand momentum.
- [ ] I can use Adam.
- [ ] I can use AdamW.
- [ ] I understand weight decay.
- [ ] I can explain Adam vs AdamW.
- [ ] I can write `zero_grad → forward → loss → backward → step` from memory.
- [ ] I can debug common loss and optimizer mistakes.

---

# 58. Difficulty Labels

Use:

```text
🟢 Easy
🟡 Medium
🔴 Hard
```

Most content should be Easy to Medium.

Medium topics:

- class weighting
- multilabel classification
- `pos_weight`
- optimizer parameter groups
- optimizer state
- Adam vs AdamW details

Do not introduce advanced optimization algorithms such as:

- LAMB
- Lion
- second-order methods
- custom optimizers

unless only briefly mentioned as out-of-scope.

---

# 59. Quality Requirements

The notebook must:

- be a valid `.ipynb`
- execute successfully from top to bottom
- work entirely on CPU
- require only PyTorch
- require no internet access
- contain no placeholder sections
- include runnable examples
- include automatic validation
- contain approximately 70–90 exercises/questions
- emphasize choosing the correct loss
- emphasize target shape and dtype
- emphasize logits
- emphasize the optimizer training sequence
- emphasize interview-ready explanations
- avoid turning into a full training-loop notebook prematurely

The notebook should take approximately **2–3 hours** to study thoroughly.

The most important learning loop is:

```text
Look at the ML task
        ↓
determine output shape
        ↓
determine target format
        ↓
choose the correct loss
        ↓
choose optimizer
        ↓
write training-step order
        ↓
run it
        ↓
debug shape / dtype mistakes
```

Finally save the completed notebook as:

```text
05_losses_optimizers.ipynb
```