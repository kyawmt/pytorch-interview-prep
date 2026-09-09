Create a Jupyter Notebook named:

```text
15_two_day_pytorch_interview_sprint.ipynb
```

This notebook is a focused **two-day PyTorch interview sprint** for someone preparing for **AI Engineer / Machine Learning Engineer interviews**.

The learner has only two days. The notebook must therefore teach and test the smallest set of PyTorch skills with the highest interview value. It must not try to summarize every PyTorch feature.

Create an actual `.ipynb` notebook. Do not create a website, Markdown-only lesson, or collection of separate files.

---

# 1. Main Outcome

By the end of the notebook, the learner must be able to do the following without documentation:

1. reason about tensor shapes
2. build a small tabular MLP with `nn.Module`
3. choose the correct output shape, target dtype, and loss
4. write a training loop from memory
5. write a validation loop from memory
6. create a custom `Dataset` and a `DataLoader`
7. move a model and batches to the correct device
8. debug common shape, dtype, device, gradient, and mode errors
9. complete one small end-to-end multiclass classification problem
10. explain the most common PyTorch interview concepts clearly and briefly

The notebook should optimize for:

```text
interview readiness
+
syntax recall
+
shape reasoning
+
independent coding
+
debugging skill
```

Do not optimize for completeness.

---

# 2. Priority Order

Use this exact priority order when deciding how much time and notebook space to allocate:

## Tier 1 — Must Be Able to Write from Memory

1. training loop
2. validation loop
3. `nn.Module` and a small MLP
4. loss selection and target requirements
5. tensor shapes and common shape operations
6. `Dataset` and `DataLoader`

## Tier 2 — Must Understand and Debug

7. Autograd and gradient accumulation
8. `model.train()` vs `model.eval()`
9. `torch.inference_mode()` / `torch.no_grad()`
10. device handling
11. logits and prediction conversion
12. common training failures

## Tier 3 — Quick Review Only

13. tensor creation, dtype, indexing, `cat`, and `stack`
14. Adam vs AdamW at a high level
15. saving/loading a `state_dict`

Spend the most practice time on Tier 1.

---

# 3. Strict Scope: No Image Processing

Do **not** include image-processing material anywhere in the notebook.

Exclude all of the following:

- image classification
- image tensors such as `(B, C, H, W)`
- `torchvision`
- image datasets
- PIL or OpenCV
- image transforms
- normalization of image channels
- resizing, cropping, flipping, or augmentation
- convolutional neural networks
- `nn.Conv1d`, `nn.Conv2d`, or `nn.Conv3d`
- pooling layers
- CNN shape calculations
- computer-vision examples
- MNIST, Fashion-MNIST, CIFAR, or similar datasets

All model-building and project examples should use **small synthetic tabular data** with shape:

```text
(batch_size, num_features)
```

Do not mention image processing even as an optional extension.

---

# 4. Additional Topics to Exclude

Because the learner has only two days, also exclude:

- distributed training
- DDP, FSDP, and DeepSpeed
- multi-GPU training
- custom CUDA kernels
- deployment frameworks
- quantization
- pruning
- ONNX
- TorchScript
- `torch.compile` internals
- profiler deep dives
- optimizer derivations
- advanced learning-rate schedulers
- advanced mixed precision
- custom Autograd functions
- full Transformer implementation
- attention implementation
- language-model training
- reinforcement learning
- graph neural networks
- long mathematical derivations

The notebook may say that these topics are out of scope, but it must not teach them.

---

# 5. Expected Study Time and Two-Day Schedule

Design the notebook for approximately **10–12 focused hours total**, split across two days.

At the beginning, show this schedule in a compact table:

| Day | Block | Topic | Time |
|---|---:|---|---:|
| 1 | 1 | Pre-assessment and tensor shape review | 45 min |
| 1 | 2 | Autograd and `nn.Module` | 60 min |
| 1 | 3 | Loss selection and target formats | 60 min |
| 1 | 4 | Training and validation loops | 2 hr |
| 1 | 5 | Closed-book rewrite and review | 30 min |
| 2 | 1 | Dataset and DataLoader | 60 min |
| 2 | 2 | End-to-end tabular multiclass project | 2 hr |
| 2 | 3 | Debugging drills | 75 min |
| 2 | 4 | Mock coding interview | 60 min |
| 2 | 5 | Rapid-fire Q&A and final recall | 45 min |

Also include a compressed fallback plan called:

```text
If I Fall Behind
```

The fallback plan must preserve these five items:

```text
training loop
loss/output/target matching
shape reasoning
Dataset/DataLoader
one end-to-end multiclass classifier
```

---

# 6. Teaching and Exercise Style

Use the following learning loop repeatedly:

```text
Predict
→ write from memory
→ run
→ validate with assert
→ inspect the mistake
→ explain it aloud
→ repeat without notes
```

Use concise explanations. Most explanations should be one to four short paragraphs.

Use:

- runnable examples
- shape tables
- short active-recall prompts
- code-completion exercises
- bug-fixing exercises
- output-prediction exercises
- `assert`-based checks
- spoken interview questions
- time-box labels

Label content with:

```text
🔥 Must know
🟡 Useful if time remains
⏱ Suggested time
🧠 Say this aloud
🛠 Debug this
✍️ Write from memory
```

Do not use long lectures.

## Important execution rule

The completed notebook must run from top to bottom without the learner filling any blanks.

For every exercise:

1. present the blank or broken version in a Markdown code block
2. ask the learner to solve it in a scratch cell or on paper
3. place the runnable solution in the next code cell
4. validate the runnable solution with `assert`
5. explain the most likely incorrect answer

Do not leave `None`, `pass`, `...`, or `raise NotImplementedError` in executable cells if that would make a full run fail.

Use collapsible HTML `<details>` blocks for optional hints and explanations when helpful. The notebook must remain useful even if the Jupyter environment does not visually collapse them.

---

# 7. Global Technical Requirements

The notebook must:

- be valid notebook JSON
- execute successfully from top to bottom
- run fully on CPU
- optionally use CUDA or Apple Silicon MPS when available
- require no internet access
- download no datasets
- use only Python standard library and PyTorch
- not require NumPy, pandas, scikit-learn, matplotlib, seaborn, or torchvision
- use deterministic random seeds where practical
- use small synthetic datasets with learnable patterns
- finish all training quickly on CPU
- use modern supported PyTorch APIs
- avoid deprecated APIs
- contain no unfinished placeholder section
- avoid writing files except for one tiny temporary checkpoint demonstration
- clean up or safely reuse the checkpoint path

Start with:

```python
import copy
import random
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
```

Set reproducibility:

```python
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
```

Select a device safely:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"PyTorch: {torch.__version__}")
print(f"Device: {device}")
```

All required examples must work when `device` is CPU.

---

# 8. Notebook Structure

Use this main hierarchy:

```text
# Two-Day PyTorch Interview Sprint

## 0. How to Use This Notebook
## 1. Pre-Assessment
## 2. Tensor Shapes and Operations
## 3. Autograd Essentials
## 4. Build an MLP with nn.Module
## 5. Match Task, Output, Target, and Loss
## 6. Training Loop from Memory
## 7. Validation Loop and Metrics
## 8. Day 1 Closed-Book Checkpoint
## 9. Dataset and DataLoader
## 10. End-to-End Tabular Multiclass Project
## 11. Debugging Drills
## 12. Device Handling and Quick Save/Load
## 13. Mock Coding Interview
## 14. Rapid-Fire Interview Questions
## 15. Final Memory Sheet
## 16. Readiness Scorecard
```

Follow the detailed instructions below.

---

# 9. Section 0 — How to Use This Notebook

Explain that the learner should:

1. attempt each prompt before revealing or running the solution
2. type important patterns rather than copy/paste
3. say shape and dtype assumptions aloud
4. keep a short mistake log
5. rewrite the training loop at least three times
6. stop low-priority sections when the suggested time ends

Add a small mistake-log template:

| Mistake | Why it happened | Correct rule | Can I now write it? |
|---|---|---|---|
| Example: float targets for CE | confused binary and multiclass | CE targets are class indices with `torch.long` | ☐ |

Add this rule prominently:

```text
Do not merely read working code.
Predict it, type it, run it, and explain it.
```

---

# 10. Section 1 — Pre-Assessment

Create a 20–25 minute closed-book diagnostic with approximately 15 questions.

Cover only high-priority skills:

- predict shapes after `reshape`, `unsqueeze`, `squeeze`, `transpose`, and `permute`
- explain `cat` vs `stack`
- identify the output size of a small MLP
- select a loss for regression, binary classification, and multiclass classification
- identify correct target dtype and shape
- order `zero_grad`, forward, loss, backward, and step
- explain `train()` vs `eval()`
- explain `eval()` vs `inference_mode()`
- define Dataset vs DataLoader
- identify a device mismatch
- explain why gradients accumulate

Do not teach before this diagnostic.

After the questions, include:

- compact answers
- a score out of 15
- routing guidance

Use routing such as:

```text
0–6: do every section
7–10: do every Must Know exercise; skim quick reviews
11–13: focus on closed-book coding and debugging
14–15: skip directly to project and mock interview, then revisit mistakes
```

The score can be self-reported; it does not need an interactive widget.

---

# 11. Section 2 — Tensor Shapes and Operations

Mark this section:

```text
🔥 Must know
⏱ 45 minutes
```

Use only tabular, vector, matrix, and class-logit examples.

Cover:

- `.shape`, `.ndim`, `.dtype`, and `.device`
- indexing and slicing
- `reshape` and `view`
- contiguity as the practical reason `view` can fail after a transpose
- `flatten`
- `squeeze` and `unsqueeze`
- `transpose` and `permute`
- reductions with `dim` and `keepdim`
- broadcasting
- matrix multiplication
- `cat` vs `stack`
- boolean masks and `torch.where`
- `argmax(dim=...)`

Do not spend time cataloging all tensor creation functions.

## Required shape drills

Include at least 20 short shape-prediction questions. At least half should require no code before answering.

Use shapes such as:

```text
(8, 12)
(32, 10)
(32, 4)
(B, F)
(B, C)
```

Include common interview traps:

- reducing without `keepdim`
- adding `(B, F)` and `(F,)`
- multiplying `(B, F) @ (F, H)`
- using the wrong `argmax` dimension
- concatenating along feature dimension vs batch dimension
- stacking and accidentally creating a new dimension
- transposing a tensor and then calling `view`
- using `squeeze()` without a dimension when batch size can be 1

## Required reference table

Create a concise table:

| Operation | Example | Shape effect | Interview warning |
|---|---|---|---|

Include `reshape`, `view`, `flatten`, `unsqueeze`, `squeeze`, `transpose`, `permute`, `cat`, `stack`, `sum`, `mean`, `argmax`, and matrix multiplication.

## Required validation

Runnable examples must use `assert` statements to verify expected shapes and selected values.

End with a five-minute closed-book shape quiz.

---

# 12. Section 3 — Autograd Essentials

Mark:

```text
🔥 Must know
⏱ 35 minutes
```

Teach only what directly supports interview explanations and training code:

```python
requires_grad=True
loss.backward()
tensor.grad
optimizer.zero_grad()
torch.no_grad()
torch.inference_mode()
tensor.detach()
```

The learner must understand:

- PyTorch records a computation graph during the forward pass
- `backward()` computes gradients through that graph
- gradients are stored on leaf parameters
- gradients accumulate by default
- a normal independent training step clears old gradients
- evaluation should not construct an unnecessary gradient graph
- `detach()` returns a tensor disconnected from the current graph

## Required experiments

Include small runnable experiments proving:

1. a scalar derivative has the expected value
2. calling `backward()` twice with appropriate graph handling accumulates gradients
3. clearing `.grad` or using optimizer `zero_grad()` prevents unwanted accumulation
4. `torch.inference_mode()` produces tensors that do not track gradients
5. parameters receive gradients after a loss backward pass

Avoid long calculus derivations.

## Required questions

Include:

- Why did a gradient double?
- Why is `param.grad` `None` before backward?
- Why should validation disable gradient tracking?
- What is the difference between `detach()` and moving a tensor to CPU?
- Does `model.eval()` disable Autograd?

End with a one-minute spoken answer to:

```text
Explain Autograd and gradient accumulation to an interviewer.
```

Provide a model answer of 3–5 sentences.

---

# 13. Section 4 — Build an MLP with nn.Module

Mark:

```text
🔥 Must know
⏱ 35 minutes
```

Use a tabular input with:

```text
num_features = 10
num_classes = 3
```

Teach the learner to write this pattern from memory:

```python
class TabularMLP(nn.Module):
    def __init__(self, num_features: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)
```

Explain concisely:

- why the class inherits from `nn.Module`
- why `super().__init__()` is required
- what `forward()` describes
- why code should call `model(x)` instead of `model.forward(x)`
- why the last layer returns logits
- how input and output shapes are determined
- how parameters become registered
- `model.parameters()` and `model.named_parameters()`
- trainable parameter counting

Keep activation discussion to ReLU and GELU. Mention only that the last classification layer should not apply softmax when using `CrossEntropyLoss`.

## Required exercises

Include:

1. fill in the missing input and output sizes
2. predict every intermediate shape
3. count parameters manually, then verify in code
4. identify an unregistered layer stored incorrectly outside module initialization
5. fix a missing `super().__init__()`
6. rewrite the full MLP from memory

Use `assert logits.shape == (batch_size, num_classes)`.

---

# 14. Section 5 — Match Task, Output, Target, and Loss

Mark this section:

```text
🔥 Highest-value reference
⏱ 60 minutes
```

Create this table prominently:

| Task | Model output | Target | Target dtype | Loss | Prediction |
|---|---|---|---|---|---|
| Regression | `(B, 1)` or `(B,)` | continuous value | floating point | `nn.MSELoss()` | raw output |
| Binary classification | `(B, 1)` or consistently `(B,)` raw logits | 0/1 | floating point | `nn.BCEWithLogitsLoss()` | sigmoid then threshold |
| Multiclass classification | `(B, C)` raw logits | class index `(B,)` | `torch.long` | `nn.CrossEntropyLoss()` | `argmax(dim=1)` |
| Multilabel classification | `(B, C)` raw logits | independent 0/1 labels `(B, C)` | floating point | `nn.BCEWithLogitsLoss()` | sigmoid then per-class threshold |

Emphasize:

```text
CrossEntropyLoss:
logits shape = (B, C)
targets shape = (B,)
targets dtype = torch.long
```

Explain why:

- softmax should not be applied before `CrossEntropyLoss`
- sigmoid should not be applied before `BCEWithLogitsLoss`
- logits are raw scores, not probabilities
- binary and multiclass target dtypes differ
- the output layer size comes from the prediction task

## Required scenario drills

Include at least 15 short scenarios in which the learner chooses:

```text
output dimension
target shape
target dtype
loss
prediction conversion
```

Use non-image scenarios such as:

- house-price regression
- customer churn
- fraud detection
- credit-risk category
- product category from tabular attributes
- multilabel customer interests
- employee attrition
- three-class machine status from sensor features

Include broken-code examples for:

- float targets passed to cross entropy
- class-index targets passed to binary loss without conversion
- applying softmax before cross entropy
- squeezing away a batch dimension
- wrong final layer size

Add `assert` checks for finite losses and expected scalar loss shapes.

---

# 15. Section 6 — Training Loop from Memory

Mark:

```text
🔥 Deepest focus
⏱ 2 hours
```

This is the most important section in the notebook.

The learner must repeatedly write this order from memory:

```text
model.train()
→ get batch
→ move batch to device
→ optimizer.zero_grad()
→ forward pass
→ calculate loss
→ loss.backward()
→ optimizer.step()
→ record detached scalar metrics
```

Teach and use this core loop:

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

Explain every line briefly.

## Required progression

Build the learner up through these stages:

1. manually perform one optimization step
2. verify gradients exist
3. verify at least one parameter changed
4. train over one DataLoader
5. train for several epochs
6. aggregate loss correctly by number of examples
7. calculate multiclass accuracy
8. place the loop inside `train_one_epoch(...)`
9. rewrite the loop with missing lines
10. rewrite the complete loop closed-book

## Loss aggregation

Teach a correct method that remains correct for an incomplete final batch:

```python
total_loss += loss.item() * X.size(0)
total_examples += X.size(0)
epoch_loss = total_loss / total_examples
```

Explain why simply averaging batch means can be slightly wrong when batch sizes differ.

## Required assertions

Validate that:

- loss is scalar and finite
- gradients exist after backward
- parameters change after `optimizer.step()`
- returned epoch loss is finite and positive
- accuracy is between 0 and 1
- training loss decreases on the simple learnable synthetic data

Do not require perfectly monotonic loss.

## Required broken loops

Include at least 10 debugging exercises involving:

- missing `zero_grad()`
- missing `backward()`
- missing `step()`
- `step()` before `backward()`
- calling `zero_grad()` between backward and step
- detached logits used to compute the training loss
- storing loss tensors rather than `.item()` for logging
- model on one device and data on another
- wrong target dtype
- training while the model remains in evaluation mode

Each exercise must ask:

1. what is wrong?
2. what symptom would appear?
3. what is the smallest fix?

## Optimizer scope

Use:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

Explain only:

- learning rate controls update size
- `zero_grad()` clears accumulated gradients
- `step()` updates parameters
- AdamW is a strong common default
- SGD is simpler and remains important conceptually

Do not teach optimizer equations.

---

# 16. Section 7 — Validation Loop and Metrics

Mark:

```text
🔥 Must know
⏱ 45 minutes
```

Teach this pattern:

```python
model.eval()

with torch.inference_mode():
    for X, y in val_loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)
        predictions = logits.argmax(dim=1)
```

Emphasize:

```text
model.eval() is not the same as torch.inference_mode()
```

Explain:

- `eval()` changes the behavior of layers such as Dropout and BatchNorm
- `inference_mode()` disables gradient tracking and reduces overhead
- validation has no backward pass
- validation has no optimizer step
- validation data should not update parameters
- metrics must be aggregated across all examples

Although the sprint MLP does not need BatchNorm or Dropout, use a tiny separate non-image demonstration to prove that `train()` and `eval()` can change outputs. A Dropout-only example is sufficient.

Create a reusable:

```python
def evaluate(model, data_loader, loss_fn, device):
    ...
```

It should return average loss and accuracy.

## Required validation tests

Use a copy of the model state before validation and assert that every parameter is unchanged afterward.

Also assert:

- no parameter gradient is created by a clean standalone inference pass
- predictions have shape `(B,)`
- validation accuracy is within `[0.0, 1.0]`

Include debugging questions about:

- using `eval()` but forgetting `inference_mode()`
- using `inference_mode()` but forgetting `eval()`
- accidentally calling `optimizer.step()` during validation
- using `argmax(dim=0)` instead of `dim=1`
- averaging accuracy incorrectly

---

# 17. Section 8 — Day 1 Closed-Book Checkpoint

Create a 30-minute checkpoint with no teaching before the answer section.

Require the learner to write or explain:

1. a tabular three-class MLP
2. correct cross-entropy target shape and dtype
3. a complete one-batch training step
4. a full training loop
5. a validation loop
6. correct loss aggregation
7. `train()` vs `eval()`
8. `eval()` vs `inference_mode()`
9. why gradients accumulate
10. five tensor shape predictions

Provide a 20-point rubric.

Use readiness guidance:

```text
17–20: proceed to Day 2
13–16: rewrite training and validation once, then proceed
0–12: repeat Sections 5–7 before starting the project
```

---

# 18. Section 9 — Dataset and DataLoader

Mark:

```text
🔥 Must know
⏱ 60 minutes
```

Teach:

```python
Dataset
TensorDataset
DataLoader
```

The learner must understand:

```text
Dataset
→ defines dataset size and how to retrieve one sample

DataLoader
→ batches, shuffles, and iterates over samples
```

Teach this custom Dataset pattern from memory:

```python
class TabularDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]
```

Cover:

- `__len__`
- `__getitem__`
- one sample vs one batch
- default collation
- `batch_size`
- `shuffle`
- incomplete final batches
- `drop_last`
- why training is normally shuffled
- why validation normally need not be shuffled
- when `TensorDataset` is enough
- `num_workers=0` as the safest notebook default
- `pin_memory` only as a one-paragraph optional note

Do not spend significant time on multiprocessing or custom `collate_fn`.

## Required exercises

Include:

1. implement and validate a custom Dataset
2. compare one Dataset sample with one DataLoader batch
3. predict DataLoader length using ceiling division
4. inspect the smaller final batch
5. compare `drop_last=False` and `drop_last=True`
6. explain why the validation loader uses `shuffle=False`
7. rewrite the custom Dataset from memory
8. convert the same tensors to a `TensorDataset`

Assert sample shapes, batch shapes, dtypes, and loader length.

---

# 19. Section 10 — End-to-End Tabular Multiclass Project

Mark:

```text
🔥 Best final practical test
⏱ 2 hours
```

Create one complete **synthetic tabular multiclass classification** project.

Use:

```text
12 input features
4 output classes
approximately 1,200–2,000 total examples
80/20 train-validation split
```

Create labels from a learnable relationship, for example by applying a fixed linear rule plus modest noise and taking `argmax`. Do not use purely random labels.

Use PyTorch only.

## Project workflow

Require this sequence:

```text
understand one sample
→ inspect shapes and dtypes
→ split tensors
→ create Dataset objects
→ create DataLoaders
→ define MLP
→ choose output dimension
→ choose CrossEntropyLoss
→ choose AdamW
→ train
→ validate
→ inspect predictions
→ debug
→ explain decisions
```

## Required model

Use a small MLP such as:

```text
12 → 32 → 16 → 4
```

with ReLU or GELU. Keep runtime short.

Do not add convolutional layers or image-shaped data.

## Independent-decision prompts

Before showing code for each major step, ask the learner to decide:

- input shape
- target shape
- target dtype
- final layer size
- loss function
- prediction rule
- metric
- whether to shuffle each loader
- where device transfers belong

## Required evaluation

Report:

- final training loss
- final validation loss
- final validation accuracy
- five example target/prediction pairs

Do not require plotting.

Use a small number of epochs that normally demonstrates learning on CPU. Include a reasonable accuracy assertion that is robust to platform differences and seed behavior. Prefer verifying that performance is clearly above random chance rather than requiring an unrealistically high exact number.

## Required ablation/debug questions

Ask the learner what happens if:

- the output layer has 3 units instead of 4
- targets are converted to float
- softmax is added before the loss
- `zero_grad()` is removed
- ReLU is removed from every hidden layer
- training loader shuffling is disabled
- validation accidentally updates parameters
- inputs remain on CPU while the model is on CUDA/MPS

## Required project explanation

End with a two-minute interview walkthrough template:

```text
Problem
Data representation
Model and output shape
Loss and target format
Training/validation design
Metric
One bug I checked for
One next improvement
```

Provide a concise example answer.

---

# 20. Section 11 — Debugging Drills

Mark:

```text
🔥 Must know
⏱ 75 minutes
```

Teach a consistent debugging order:

```text
1. inspect shape
2. inspect dtype
3. inspect device
4. inspect requires_grad and gradients
5. inspect train/eval mode
6. inspect loss/output/target compatibility
7. verify parameters change
8. test one batch
9. try to overfit a tiny subset
```

Add a compact diagnostic helper that prints or returns:

```text
shape
dtype
device
requires_grad
finite status
```

## Required bugs

Include at least 18 short debugging tasks across these categories:

### Shape

- incompatible matrix multiplication
- wrong final output size
- wrong target shape for cross entropy
- wrong `argmax` dimension
- unsafe `squeeze()` removing the batch dimension

### Dtype

- cross-entropy targets are float
- features accidentally use integer dtype
- binary targets use an incompatible shape/dtype combination

### Device

- model and features on different devices
- targets left on CPU
- `.to(device)` called without assigning the returned tensor

### Gradient and optimizer

- missing `zero_grad()`
- forward pass inside `inference_mode()` during training
- detached loss path
- learning rate much too high
- optimizer constructed from the wrong model parameters

### Train/eval behavior

- validation executed in training mode
- training accidentally executed in evaluation mode

For each bug include:

1. the broken snippet in Markdown
2. the expected error or behavioral symptom
3. a prompt to diagnose before revealing the answer
4. corrected runnable code
5. a one-sentence prevention rule

At least six bugs should be behavioral bugs that do not necessarily raise an exception.

## Loss-not-decreasing checklist

Create a concise ordered checklist:

```text
Can the model overfit one tiny batch?
Are labels learnable and aligned with features?
Is the loss correct for output/target?
Are shapes and dtypes correct?
Do parameters have gradients?
Do parameters change after step()?
Is the learning rate reasonable?
Is the model in training mode?
Are activations/output layers appropriate?
```

---

# 21. Section 12 — Device Handling and Quick Save/Load

Mark:

```text
🟡 Quick review
⏱ 25 minutes
```

## Device handling

Teach the learner to remember:

```python
model = model.to(device)
X = X.to(device)
y = y.to(device)
```

Explain:

- parameters and input tensors must be on compatible devices
- `.to(...)` returns the converted/moved tensor
- outputs can be moved to CPU when needed
- device-safe code must still work on CPU-only machines

Do not teach GPU performance tuning.

## Save/load

Teach only:

```python
torch.save(model.state_dict(), checkpoint_path)

state = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True,
)

restored_model.load_state_dict(state)
restored_model.to(device)
restored_model.eval()
```

Use `tempfile.TemporaryDirectory()` so the demonstration does not leave a permanent artifact.

Validate that the original and restored models produce matching outputs for the same input while both are in evaluation mode.

Explain in one paragraph why the architecture must be instantiated before loading a `state_dict`.

Do not cover full checkpoint resume logic unless it is a two- or three-sentence optional note.

---

# 22. Section 13 — Mock Coding Interview

Create a 60-minute mock interview with no solutions until the end of the section.

Use a fresh synthetic tabular binary or multiclass classification scenario. It must not involve images.

The mock interview should have four rounds:

## Round A — Shape Reasoning, 10 minutes

Include five shape/output questions.

## Round B — Model and Loss, 10 minutes

Ask the learner to define a small MLP and justify:

- input size
- output size
- loss
- target dtype
- prediction rule

## Round C — Training and Evaluation, 25 minutes

Ask the learner to write:

- one complete training epoch
- one evaluation function
- correct loss and accuracy aggregation
- device transfers

## Round D — Debugging and Explanation, 15 minutes

Provide two bugs and ask the learner to:

- diagnose them
- repair them
- explain the symptom
- describe how they would test the fix

Provide:

- complete reference solutions after all rounds
- a 30-point rubric
- explicit scoring for correctness, shape reasoning, clean code, debugging, and communication

Use readiness bands:

```text
26–30: interview ready for core PyTorch tasks
21–25: nearly ready; repeat weakest round
15–20: repeat project and training-loop drills
0–14: return to Day 1 essentials
```

---

# 23. Section 14 — Rapid-Fire Interview Questions

Create approximately 30 concise questions and answers.

Each short answer should normally be 1–4 sentences and suitable for speaking in an interview.

Required questions:

1. What is a tensor?
2. What is a logit?
3. What does `requires_grad` do?
4. What happens when `loss.backward()` runs?
5. Why do PyTorch gradients accumulate?
6. Why call `optimizer.zero_grad()`?
7. What does `optimizer.step()` do?
8. What is the difference between an epoch and a batch?
9. Why use mini-batches?
10. Why subclass `nn.Module`?
11. Why call `super().__init__()`?
12. What is `forward()`?
13. Why call `model(x)` rather than `model.forward(x)`?
14. Why are nonlinear activations needed?
15. What is the required input for `CrossEntropyLoss`?
16. Why not apply softmax before `CrossEntropyLoss`?
17. Why use `BCEWithLogitsLoss`?
18. What is Dataset vs DataLoader?
19. Why shuffle training data?
20. What does `model.train()` do?
21. What does `model.eval()` do?
22. Does `model.eval()` disable gradients?
23. `no_grad()` vs `inference_mode()`?
24. Why must model and data share a device?
25. `reshape` vs `view`?
26. `cat` vs `stack`?
27. Why might loss not decrease?
28. How do you verify that parameters update?
29. How do you debug a shape mismatch?
30. What is a `state_dict`?

Add five scenario follow-ups that test whether the learner can apply the answers rather than recite definitions.

---

# 24. Section 15 — Final Memory Sheet

Create a compact last-minute review section designed to take 10–15 minutes.

It must include only the highest-value syntax.

## Device

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

## Model

Include the complete tabular MLP pattern.

## Dataset

Include the complete custom Dataset pattern.

## DataLoader

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)
```

## Loss choice

Repeat the regression/binary/multiclass/multilabel table in an abbreviated form.

## Training

Include the complete core training loop.

## Validation

Include the complete core validation loop.

## Prediction

```python
# Multiclass
predictions = logits.argmax(dim=1)

# Binary
probabilities = torch.sigmoid(logits)
predictions = (probabilities >= 0.5).long()
```

## Save/load

Include the minimal `state_dict` pattern.

## Debug order

```text
shape → dtype → device → gradients → mode → loss compatibility → parameter update
```

## Ten rules not to forget

Include:

1. Cross entropy takes raw `(B, C)` logits.
2. Cross-entropy targets are `(B,)` and `torch.long`.
3. BCE-with-logits takes raw logits and floating-point targets.
4. PyTorch gradients accumulate by default.
5. Clear gradients before a normal independent training step.
6. Backward computes gradients; step updates parameters.
7. `eval()` does not disable gradients.
8. Validation uses both evaluation mode and disabled gradient tracking.
9. Model, features, and targets must be on compatible devices.
10. Dataset gives samples; DataLoader gives batches.

Keep this section extremely scannable.

---

# 25. Section 16 — Readiness Scorecard

End with a checklist:

```text
I can do this without notes:
```

- [ ] predict common tensor-operation shapes
- [ ] explain broadcasting
- [ ] explain `cat` vs `stack`
- [ ] explain gradient accumulation
- [ ] build a tabular MLP with `nn.Module`
- [ ] determine the correct output layer size
- [ ] select regression, binary, and multiclass losses
- [ ] state the correct target shape and dtype
- [ ] write a training loop
- [ ] write a validation loop
- [ ] aggregate loss correctly
- [ ] calculate multiclass accuracy
- [ ] explain `train()` vs `eval()`
- [ ] explain `eval()` vs `inference_mode()`
- [ ] write a custom Dataset
- [ ] create train and validation DataLoaders
- [ ] write device-agnostic code
- [ ] debug shape, dtype, and device errors
- [ ] verify gradients and parameter updates
- [ ] save and restore a `state_dict`
- [ ] explain an end-to-end project in two minutes

Add this rule:

```text
Any unchecked Tier 1 skill should be practiced again before spending time on lower-priority topics.
```

Finish with a final 15-minute closed-book task:

```text
From an empty cell, write:

1. a three-class tabular MLP
2. CrossEntropyLoss and AdamW setup
3. a training loop
4. a validation loop
5. a custom Dataset

Then explain every expected shape and dtype aloud.
```

---

# 26. Exercise and Question Counts

Across the notebook include approximately:

- 20–25 tensor shape/value predictions
- 10 Autograd/model questions
- 15 task/loss/output/target scenarios
- 10 broken training-loop exercises
- 8 Dataset/DataLoader exercises
- 18 debugging drills
- 1 complete tabular multiclass project
- 1 timed mock coding interview
- 30 rapid-fire interview questions
- 3 closed-book rewrites of the training loop
- 2 closed-book rewrites of the validation loop

Do not inflate counts with trivial duplicates. Repetition should focus on recall of high-value patterns.

---

# 27. Quality Bar

The final notebook must feel like a carefully edited interview workbook, not a dump of API documentation.

It must:

- prioritize writing and debugging over reading
- repeatedly test shapes, dtypes, and device placement
- make the learner choose losses instead of merely showing them
- make the learner write loops from memory several times
- use one coherent tabular project to connect all core concepts
- provide concise reference solutions
- include meaningful `assert` checks
- keep code cells small and readable
- avoid needless abstraction
- use descriptive variable names such as `features`, `targets`, and `logits`
- use type hints only where they improve clarity
- include expected output descriptions where platform output may vary
- avoid assertions based on exact floating-point values unless stable
- avoid platform-specific CUDA assumptions
- avoid excessive emoji beyond the defined priority labels
- contain no image-processing examples or terminology

The most important principle is:

```text
If a topic does not help the learner write, debug, or explain a core PyTorch interview solution within two days, leave it out.
```

Finally save the completed notebook as:

```text
15_two_day_pytorch_interview_sprint.ipynb
```
