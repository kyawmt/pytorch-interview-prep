Create a Jupyter Notebook named:

```text
12_pytorch_mini_projects.ipynb
```

This notebook is the next practical stage of a **PyTorch Interview Preparation** course for someone preparing for **AI Engineer / Machine Learning Engineer / Generative AI Engineer interviews**.

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
```

Do **not** teach PyTorch fundamentals again.

The purpose of this notebook is to make the learner apply PyTorch in several **small end-to-end ML projects** similar to what may appear in:

- AI Engineer coding interviews
- ML Engineer take-home exercises
- pair-programming interviews
- practical screening tests
- project walkthrough discussions

This notebook should focus on:

```text
problem
→ data
→ Dataset/DataLoader
→ model
→ loss
→ optimizer
→ training
→ validation
→ debugging
→ evaluation
→ explanation
```

The learner should build significant parts themselves rather than simply running completed code.

---

# Main Goal

By the end of the notebook, the learner should be able to take a simple ML problem description and independently decide:

- input representation
- output shape
- model architecture
- target format
- loss function
- optimizer
- Dataset/DataLoader setup
- training loop
- evaluation metric
- device handling
- basic debugging strategy

The notebook should test whether the learner can combine all previous PyTorch knowledge into complete solutions.

---

# Projects to Include

Create **5 mini-projects**:

```text
Project 1 — Linear Regression
Project 2 — Binary Classification
Project 3 — Multiclass Classification
Project 4 — Image Classification with a CNN
Project 5 — Text Classification with Embeddings
```

Then include an optional:

```text
Project 6 — Tiny Transformer Language Model
```

The first five projects are required.

---

# Notebook Structure

Use approximately:

```text
# PyTorch Mini Projects

## 1. How to Approach a PyTorch Project
## 2. Project 1 — Linear Regression
## 3. Project 2 — Binary Classification
## 4. Project 3 — Multiclass Classification
## 5. Project 4 — CNN Image Classification
## 6. Project 5 — Text Classification
## 7. Optional Project 6 — Tiny Language Model
## 8. Compare the Projects
## 9. Project Debugging Scenarios
## 10. Interview Walkthrough Practice
## 11. Architecture Decision Questions
## 12. Final From-Scratch Challenge
## 13. Project Cheat Sheet
```

---

# 1. General Project Framework

Start with a concise reusable framework:

```text
Step 1: Understand the task
Step 2: Inspect input and target shapes
Step 3: Choose output representation
Step 4: Choose loss function
Step 5: Build Dataset/DataLoader
Step 6: Build model
Step 7: Select optimizer
Step 8: Train
Step 9: Validate
Step 10: Debug
Step 11: Evaluate
Step 12: Explain design decisions
```

Mark:

```text
🔥 Interview Essential
```

Explain that interviewers often care more about whether the learner can reason through these decisions than whether they remember every PyTorch API perfectly.

---

# 2. Reusable Setup

Start with:

```python
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    Dataset,
    TensorDataset,
    DataLoader,
    random_split,
)

torch.manual_seed(42)
random.seed(42)
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

print("Using:", device)
```

The notebook must run fully on CPU.

Use **synthetic data only** so it requires no internet.

---

# 3. Reusable Training Utilities

Do not immediately provide all utilities as finished code.

First ask the learner to implement reusable functions such as:

```python
def train_one_epoch(...):
    ...
```

and:

```python
def evaluate_classifier(...):
    ...
```

However, because different projects have different output/loss structures, avoid forcing one overly generic training framework.

Prefer simple readable functions.

The learner should understand the code rather than hide everything behind abstractions.

---

# PROJECT 1 — LINEAR REGRESSION

# 4. Problem

Generate synthetic data representing:

```text
y = 3x1 - 2x2 + 1.5 + noise
```

Use approximately:

```python
num_samples = 1000
```

Input shape:

```text
(1000, 2)
```

Target shape:

```text
(1000, 1)
```

Generate with:

```python
X = torch.randn(num_samples, 2)

noise = 0.2 * torch.randn(
    num_samples,
    1
)

y = (
    3 * X[:, 0:1]
    - 2 * X[:, 1:2]
    + 1.5
    + noise
)
```

---

# 5. Before Coding

Ask learner:

```text
1. Is this regression or classification?
2. What should the model output shape be?
3. What loss should we use?
4. Should the final layer have an activation?
5. What metric could we use?
```

Expected reasoning:

```text
Regression
output → (B,1)
MSELoss
no required classification activation
MAE/MSE/RMSE are reasonable metrics
```

Keep answers hidden.

---

# 6. Build Dataset and DataLoaders

Learner should:

1. Create `TensorDataset`.
2. Split 80/20.
3. Create training DataLoader.
4. Create validation DataLoader.

Requirements:

```text
train batch size = 32
shuffle=True

val batch size = 64
shuffle=False
```

Use validation cells.

---

# 7. Build Regression Model

Start with blank:

```python
model = None
```

Require architecture approximately:

```text
2
↓
16
↓ ReLU
↓
8
↓ ReLU
↓
1
```

Accept reasonable equivalent architectures.

Do not put an unnecessary sigmoid/softmax at the output.

---

# 8. Select Loss and Optimizer

Ask learner to fill:

```python
loss_fn = ...
optimizer = ...
```

Preferred:

```text
MSELoss
Adam or AdamW
```

Do not require exactly one optimizer if another reasonable choice works.

---

# 9. Train Regression Model

Require:

- device handling
- `model.train()`
- forward
- loss
- backward
- optimizer step
- validation
- loss history

Train only long enough to show meaningful convergence.

Expected learned relation should approximately recover:

```text
3
-2
1.5
```

Do not require exact values because the MLP does not expose direct linear coefficients.

---

# 10. Regression Evaluation

Compute:

```text
MSE
MAE
```

Optionally RMSE:

```python
rmse = mse ** 0.5
```

Ask learner:

> Why is accuracy not appropriate here?

---

# 11. Regression Interview Questions

Ask:

- Why MSE?
- When might L1Loss be preferable?
- Why no softmax?
- What output shape should the model use?
- What if prediction is `(B,1)` but target is `(B,)`?
- Why could unintended broadcasting be dangerous?

---

# PROJECT 2 — BINARY CLASSIFICATION

# 12. Generate Dataset

Create a learnable binary dataset.

For example:

```python
X = torch.randn(
    1500,
    10
)

true_w = torch.randn(
    10,
    1
)

scores = X @ true_w

y = (
    scores > 0
).float()
```

Shapes:

```text
X:
(1500,10)

y:
(1500,1)
```

---

# 13. Binary Task Planning

Ask learner:

```text
What should model output?

What dtype should target use?

Which loss?

Should sigmoid be inside model?

How do we convert logits into predictions?
```

Expected:

```text
output:
(B,1) raw logits

target:
float

loss:
BCEWithLogitsLoss

training:
NO sigmoid before loss

inference:
sigmoid → threshold
```

Mark:

```text
🔥 Interview Essential
```

---

# 14. Build Binary Classifier

Architecture:

```text
10
↓
32
↓ ReLU
↓
16
↓ ReLU
↓
1 logit
```

Ask learner to implement custom `nn.Module` or `nn.Sequential`.

Do not apply sigmoid in `forward()`.

Validation:

```python
test_x = torch.randn(8, 10)

assert model(test_x).shape == (8, 1)
```

---

# 15. Binary Loss and Training

Use:

```python
nn.BCEWithLogitsLoss()
```

Require learner to write complete training logic.

Prediction:

```python
probs = torch.sigmoid(logits)

pred = (
    probs >= 0.5
).float()
```

Compute accuracy.

---

# 16. Binary Classification Metrics

Calculate:

```text
accuracy
precision
recall
```

Implement them manually using PyTorch tensors.

Do not require sklearn.

Teach:

```text
TP
FP
TN
FN
```

Keep formulas concise.

This is useful for interview practice.

---

# 17. Threshold Exercise

Ask learner to compare:

```text
threshold = 0.5
threshold = 0.7
```

Explain conceptually:

```text
higher threshold
→ usually fewer positive predictions
→ precision may increase
→ recall may decrease
```

Do not imply this always happens monotonically in every finite dataset, but explain the general trade-off.

---

# PROJECT 3 — MULTICLASS CLASSIFICATION

# 18. Dataset

Generate:

```python
num_samples = 2000
num_features = 20
num_classes = 5

X = torch.randn(
    num_samples,
    num_features
)

true_w = torch.randn(
    num_features,
    num_classes
)

scores = X @ true_w

y = scores.argmax(
    dim=1
)
```

Shapes:

```text
X:
(2000,20)

y:
(2000,)
```

Target dtype:

```text
torch.long
```

---

# 19. Planning Questions

Ask:

```text
Output shape?

Loss?

Target shape?

Target dtype?

Do we softmax before loss?

How do we predict?
```

Expected:

```text
(B,5)

CrossEntropyLoss

(B,)

long

No softmax before loss

argmax(dim=1)
```

---

# 20. Build Multiclass Model

Architecture:

```text
20
↓
64
↓ ReLU
↓ Dropout(0.2)
↓
32
↓ ReLU
↓
5 logits
```

Ask learner to create it.

---

# 21. Train / Validate

Require full:

```text
train_one_epoch
evaluate
```

Compute:

- loss
- accuracy

Store history.

Ask learner to identify whether overfitting occurs.

---

# 22. Confusion Matrix From Scratch

Create:

```python
confusion = torch.zeros(
    num_classes,
    num_classes,
    dtype=torch.int64
)
```

Fill manually from predictions/targets.

Rows:

```text
actual
```

Columns:

```text
predicted
```

or clearly document the chosen convention.

Do not require sklearn.

This is a good medium difficulty exercise.

---

# 23. Top-K Accuracy

Teach:

```python
top_values, top_indices = torch.topk(
    logits,
    k=3,
    dim=1
)
```

Ask learner to compute:

```text
top-3 accuracy
```

Explain why top-k metrics are useful when there are many classes.

---

# PROJECT 4 — IMAGE CLASSIFICATION WITH CNN

# 24. Synthetic Image Dataset

Do not download MNIST/CIFAR.

Generate synthetic image classification data with learnable patterns.

For example create:

```text
3 classes
images shape = (1, 16, 16)
```

Classes can represent:

```text
class 0 → bright vertical stripe
class 1 → bright horizontal stripe
class 2 → bright diagonal pattern
```

Add random noise.

Create approximately:

```text
1500 images
```

This lets the CNN learn meaningful patterns without internet access.

---

# 25. Image Tensor Format

Ask:

```text
What shape should one batch use for Conv2d?
```

Expected:

```text
(B,C,H,W)
```

For this project:

```text
(B,1,16,16)
```

Mark:

```text
🔥 Interview Essential
```

---

# 26. Build CNN

Require architecture approximately:

```text
Conv2d 1→8, kernel=3, padding=1
ReLU
MaxPool2d(2)

Conv2d 8→16, kernel=3, padding=1
ReLU
MaxPool2d(2)

Flatten

Linear → 32
ReLU

Linear → 3 logits
```

Ask learner to calculate shape after every stage **before running the model**.

Expected:

```text
(B,1,16,16)

→
(B,8,16,16)

→ pool
(B,8,8,8)

→
(B,16,8,8)

→ pool
(B,16,4,4)

→ flatten
(B,256)

→
(B,32)

→
(B,3)
```

---

# 27. CNN Parameter Counting

Ask learner to manually calculate parameter count for:

```python
nn.Conv2d(
    1,
    8,
    kernel_size=3,
    padding=1
)
```

Expected:

```text
8 × 1 × 3 × 3 + 8
```

Do the same for second conv and first Linear layer.

Validate programmatically.

---

# 28. Train CNN

Use:

```text
CrossEntropyLoss
AdamW
```

Require standard device-aware loop.

Run enough epochs for high validation accuracy because the synthetic pattern task should be relatively easy.

---

# 29. CNN Debugging Exercise

Give the learner a broken input:

```text
(B,H,W,C)
```

Ask them to diagnose Conv2d error.

Expected:

```python
x = x.permute(
    0,
    3,
    1,
    2
)
```

when appropriate.

---

# PROJECT 5 — TEXT CLASSIFICATION

# 30. Synthetic Text-Style Dataset

Create integer token sequences.

No tokenizer dependency.

Define:

```text
vocab_size = 500
num_classes = 3
```

Create variable sequence lengths approximately:

```text
5–30 tokens
```

Make class labels learnable.

For example:

```text
class 0:
contains many tokens from range 1–50

class 1:
contains many tokens from range 51–100

class 2:
contains many tokens from range 101–150
```

Add random noise tokens from rest of vocabulary.

Generate approximately:

```text
1500 sequences
```

---

# 31. Custom Dataset

Require learner to implement:

```python
class TextDataset(Dataset):
    ...
```

Each sample should return:

```python
{
    "tokens": ...,
    "label": ...
}
```

---

# 32. Dynamic Padding `collate_fn`

Require learner to implement:

```python
def collate_batch(batch):
    ...
```

Return:

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...
}
```

Use:

```python
torch.nn.utils.rnn.pad_sequence
```

if desired.

Expected shapes:

```text
input_ids:
(B,T)

attention_mask:
(B,T)

labels:
(B,)
```

---

# 33. Build Text Classifier

Architecture:

```text
token IDs
↓
Embedding(vocab_size, 64)
↓
masked mean pooling
↓
Linear(64,32)
↓ ReLU
↓
Linear(32,3)
```

Require custom `nn.Module`.

---

# 34. Masked Mean Pooling

Learner should implement:

```python
embeddings = self.embedding(
    input_ids
)
```

Shape:

```text
(B,T,D)
```

Mask:

```python
mask = attention_mask.unsqueeze(
    -1
)
```

Shape:

```text
(B,T,1)
```

Then:

```python
masked = embeddings * mask
```

Sum:

```python
summed = masked.sum(
    dim=1
)
```

Lengths:

```python
lengths = mask.sum(
    dim=1
).clamp(
    min=1
)
```

Mean:

```python
pooled = summed / lengths
```

Ask learner to reason about all shapes.

Mark:

```text
🔥 Interview Useful for NLP
```

---

# 35. Train Text Classifier

Use:

```text
CrossEntropyLoss
AdamW
```

Train and evaluate.

Explain that this is intentionally simpler than a Transformer and is useful as a baseline.

---

# 36. Text Project Interview Discussion

Ask:

- Why embedding instead of one-hot vectors?
- Why mask padding?
- Why mean pooling?
- What happens if padding is included in mean?
- How would a Transformer improve this model?
- Where would a tokenizer normally fit?
- Why does Embedding input need integer IDs?

---

# OPTIONAL PROJECT 6 — TINY TRANSFORMER LANGUAGE MODEL

# 37. Optional Project

Mark:

```text
🔴 Hard / Optional
```

Build on Notebook 10.

Use:

```text
vocab_size = 128
hidden_size = 64
num_heads = 4
num_layers = 2
```

Generate synthetic token sequences with a simple learnable pattern rather than pure random next tokens.

For example:

```text
next token = (current token + 1) % vocabulary
```

with variation/noise.

This lets the tiny model demonstrate learning.

---

# 38. Tiny Language Model

Architecture:

```text
Token Embedding
+
Position Embedding
↓
Transformer Blocks
↓
LayerNorm
↓
Linear to vocab
```

Output:

```text
(B,T,V)
```

Loss:

```python
nn.CrossEntropyLoss()
```

using flattened token positions.

---

# 39. Greedy Generation

After training:

```python
context = torch.tensor(
    [[1, 2, 3]]
)
```

generate several tokens using:

```python
next_logits = logits[:, -1, :]

next_token = next_logits.argmax(
    dim=-1,
    keepdim=True
)
```

Append and repeat.

Explain that this is for learning mechanics, not production LLM generation.

---

# 40. Compare All Projects

Create a summary table:

| Project | Output | Target | Loss | Prediction |
|---|---|---|---|---|
| Regression | `(B,1)` | float `(B,1)` | MSE | raw value |
| Binary | `(B,1)` logits | float `(B,1)` | BCEWithLogits | sigmoid + threshold |
| Multiclass | `(B,C)` logits | long `(B,)` | CrossEntropy | argmax |
| CNN multiclass | `(B,C)` logits | long `(B,)` | CrossEntropy | argmax |
| Text multiclass | `(B,C)` logits | long `(B,)` | CrossEntropy | argmax |
| Language model | `(B,T,V)` logits | long `(B,T)` | CrossEntropy | next-token selection |

Mark:

```text
🔥 Memorize This Table
```

---

# 41. Architecture Decision Questions

Create approximately 20 questions.

Examples:

> You have 40 numeric features and 5 classes. What model could you start with?

Expected:

```text
MLP
```

> Input is token IDs `(B,T)`. What layer converts IDs to dense vectors?

```text
nn.Embedding
```

> Input is `(B,3,224,224)`. What family of layer is natural for basic image processing?

```text
Conv2d / CNN
```

> You need order-sensitive contextual processing of long text.

Possible:

```text
Transformer
```

Do not imply only one architecture is ever valid.

---

# 42. Project Debugging Scenarios

Create at least 20.

Examples:

### Scenario

Regression predictions:

```text
(B,1)
```

targets:

```text
(B,)
```

MSE produces strange warning/results.

Ask what is happening.

---

### Scenario

Binary model uses:

```python
nn.Sigmoid()
```

and:

```python
BCEWithLogitsLoss()
```

Ask fix.

---

### Scenario

Multiclass model output:

```text
(B,5)
```

labels are floats.

Ask fix.

---

### Scenario

CNN first Linear expects 1024 features, but flatten produces 256.

Ask how to debug.

---

### Scenario

Text classifier accuracy is low because padded tokens are included in average pooling.

Ask how to fix.

---

# 43. Interview Project Walkthrough Practice

For every project, ask learner to explain it in this format:

```text
1. What problem am I solving?
2. What does one input look like?
3. What does one target look like?
4. Why did I choose this model?
5. Why this loss?
6. Why this optimizer?
7. How do I measure performance?
8. What bugs did I check?
9. What would I improve?
```

Provide concise model answers after hidden sections.

This section is important for project-based interviews.

---

# 44. 60-Second Project Explanation

For each required project, provide a prompt:

```text
Explain this project to an interviewer in 60 seconds.
```

Provide a high-quality short example answer after the learner attempts it.

Do not make answers overly formal.

---

# 45. Training History Interpretation

Create synthetic example histories such as:

### Case A

```text
train loss ↓
val loss ↓
```

Ask interpretation.

### Case B

```text
train loss ↓
val loss ↑ after epoch 8
```

Expected:

```text
likely overfitting
```

### Case C

```text
both losses remain high
```

Possible:

```text
underfitting or optimization problem
```

Ask what to investigate before changing architecture.

---

# 46. Improving Each Project

After each project ask:

```text
What would you try next?
```

Possible topics:

- more data
- feature normalization
- architecture changes
- regularization
- learning-rate tuning
- class weighting
- data augmentation
- better evaluation metrics
- early stopping

Do not encourage blind hyperparameter tuning.

---

# 47. Final From-Scratch Challenge

Create a completely new synthetic task without telling the learner which project pattern to copy.

Example:

> Each sample has 30 features and belongs to one of 6 categories.

Generate learnable synthetic data.

Only provide:

```python
X = ...
y = ...
```

The learner must independently decide:

1. train/validation split
2. Dataset/DataLoader
3. model
4. output dimension
5. loss
6. optimizer
7. training loop
8. validation loop
9. accuracy metric
10. device handling

Do not give architecture hints initially.

---

# 48. Final Challenge Requirements

The learner's code must demonstrate:

```text
Dataset/DataLoader
nn.Module or Sequential
correct loss
correct target dtype
AdamW or reasonable optimizer
model.train()
model.eval()
inference_mode/no_grad
device handling
correct metric calculation
```

Automatically validate important properties.

---

# 49. Final Challenge Follow-Up

Ask:

> How would this change if it were binary instead of six classes?

Expected differences:

```text
output:
1 logit

loss:
BCEWithLogitsLoss

target:
float

prediction:
sigmoid + threshold
```

Ask:

> How would this change if the target were a continuous number?

Expected:

```text
regression output
MSE/L1
```

---

# 50. Coding Constraints

For all projects:

- Do not rely on external datasets.
- Do not use scikit-learn for training.
- Do not use pandas.
- Do not use torchvision.
- Do not use Hugging Face.
- Do not require internet.
- Use PyTorch directly.
- Keep runtime reasonable on CPU.
- Use deterministic seeds where appropriate.

The focus is **PyTorch skill**, not external libraries.

---

# 51. Exercise Style

Do not simply show completed projects.

For every major step:

```text
Ask learner
→ blank code cell
→ validation
→ hidden solution
```

Example:

```python
# Choose the correct loss function.

loss_fn = None
```

Validation:

```python
assert isinstance(
    loss_fn,
    nn.CrossEntropyLoss
)

print("✅ Correct!")
```

---

# 52. Debugging Validation

Where possible, automatically verify:

- correct output shape
- correct dtype
- finite loss
- gradients exist
- parameters change
- accuracy between 0 and 1
- validation does not change parameters
- tensors/models share device

---

# 53. Project Completion Checklist

For each project include:

```text
Before moving on:
```

- [ ] I understand input shape.
- [ ] I understand target shape.
- [ ] I chose the loss myself.
- [ ] I built the model.
- [ ] I wrote the training loop.
- [ ] I wrote the validation loop.
- [ ] I calculated the metric.
- [ ] I can explain the architecture.
- [ ] I can explain why the loss is correct.
- [ ] I can describe at least two likely bugs.
- [ ] I can explain one possible improvement.

---

# 54. Interview Questions

Create approximately 30 project-oriented questions.

Include:

1. How do you decide a model's output dimension?
2. How do you choose a loss function?
3. Why should a binary classifier often return one raw logit?
4. Why should a multiclass classifier return one logit per class?
5. How do you choose target dtype?
6. What is a good first architecture for tabular data?
7. When would you use a CNN?
8. When would you use an embedding layer?
9. Why dynamically pad text?
10. Why mask padding during pooling?
11. Why separate train and validation datasets?
12. Why shuffle training data?
13. Why not shuffle validation data necessarily?
14. How do you determine whether a model is learning?
15. How do you detect overfitting?
16. How would you debug flat training loss?
17. How do you verify gradients exist?
18. How do you verify parameters update?
19. Why is validation run in eval mode?
20. Why disable Autograd during validation?
21. How would you handle class imbalance?
22. How would you handle GPU OOM?
23. What would you change to speed training?
24. What would you log during training?
25. How would you explain your model architecture?
26. How would you decide whether a more complex model is justified?
27. Why start with a baseline?
28. What changes between regression and classification?
29. What changes between binary and multiclass classification?
30. How would you productionize one of these projects at a high level?

Provide:

### Short Interview Answer

and:

### Detailed Explanation

in collapsible sections.

---

# 55. Final Project Comparison Challenge

Give scenarios and ask learner to choose:

```text
Model family
Output shape
Loss
Target dtype
Metric
```

Include at least 15 scenarios covering:

- house-price regression
- fraud binary classification
- 100-class image classification
- multilabel tagging
- sentiment classification
- token-level language modeling
- tabular multiclass prediction

---

# 56. Final Cheat Sheet

End with a compact project-selection cheat sheet.

```text
REGRESSION

Model output:
(B,1) or task-specific continuous shape

Target:
float

Loss:
MSELoss / L1Loss

Prediction:
raw output
```

```text
BINARY CLASSIFICATION

Model output:
(B,1) raw logits

Target:
float

Loss:
BCEWithLogitsLoss

Prediction:
sigmoid → threshold
```

```text
MULTICLASS CLASSIFICATION

Model output:
(B,C) raw logits

Target:
(B,) long

Loss:
CrossEntropyLoss

Prediction:
argmax(dim=1)
```

```text
MULTILABEL CLASSIFICATION

Model output:
(B,C) raw logits

Target:
(B,C) float

Loss:
BCEWithLogitsLoss

Prediction:
sigmoid → threshold per class
```

```text
LANGUAGE MODELING

Model output:
(B,T,V)

Target:
(B,T) long

Loss:
CrossEntropyLoss on flattened token positions
```

---

# 57. Critical Mental Model

End with:

```text
New ML problem
      ↓
What is one input?
      ↓
What is one target?
      ↓
What should model output?
      ↓
What loss matches that output/target?
      ↓
What architecture is a reasonable baseline?
      ↓
Build data pipeline
      ↓
Train
      ↓
Validate
      ↓
Debug
      ↓
Improve
```

Mark:

```text
🔥 This is more important than memorizing architectures.
```

---

# 58. Final Completion Checklist

Add:

```text
## PyTorch Project Readiness
```

- [ ] I can build a regression model from scratch.
- [ ] I can build a binary classifier from scratch.
- [ ] I can build a multiclass classifier from scratch.
- [ ] I can build a basic CNN.
- [ ] I can build an embedding-based text classifier.
- [ ] I can choose losses without hints.
- [ ] I can choose target shapes and dtypes correctly.
- [ ] I can write Dataset/DataLoader code.
- [ ] I can write training and validation loops.
- [ ] I can make training device-aware.
- [ ] I can calculate appropriate metrics.
- [ ] I can debug common training failures.
- [ ] I can explain my architecture choices.
- [ ] I can explain a project clearly in an interview.
- [ ] I can take a new ML problem and build a baseline without copying a template.

---

# 59. Quality Requirements

The notebook must:

- be a valid `.ipynb`
- execute successfully from top to bottom
- work on CPU
- optionally use CUDA/MPS automatically
- require no internet
- use only PyTorch and Python standard library
- use meaningful learnable synthetic datasets
- avoid pure-random labels for projects where model learning needs to be demonstrated
- contain automatic validation
- contain hidden solutions
- heavily emphasize independent coding
- contain at least 5 complete mini-projects
- include at least 20 debugging/scenario questions
- include project walkthrough interview practice
- avoid adding major new PyTorch concepts unnecessarily

The notebook should take approximately **5–8 hours** to complete thoroughly and should also work as several independent practice sessions.

The core principle is:

```text
Do not give the learner a finished model and ask them to run it.

Give them a problem.

Make them decide:

data
→ shape
→ model
→ output
→ loss
→ optimizer
→ training
→ evaluation
→ debugging
```

By the end, the learner should be able to receive an interview prompt like:

> "Build a PyTorch classifier for this dataset."

and know how to start without needing a tutorial.

Finally save the completed notebook as:

```text
12_pytorch_mini_projects.ipynb
```