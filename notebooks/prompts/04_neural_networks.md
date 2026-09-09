Create a Jupyter Notebook named:

```text
04_neural_networks.ipynb
```

This notebook is the fourth part of a **PyTorch Interview Preparation** course for someone preparing for **AI Engineer / Machine Learning Engineer interviews**.

Assume the learner has already completed:

```text
01_tensor_basics.ipynb
02_tensor_operations.ipynb
03_autograd.ipynb
```

and already understands:

- tensors and tensor shapes
- broadcasting and matrix multiplication
- vectorized operations
- Autograd
- `requires_grad`
- `.backward()`
- `.grad`
- gradient accumulation
- `torch.no_grad()`
- `.detach()`
- freezing parameters

Do not spend significant time reteaching those topics.

The purpose of this notebook is to teach how neural networks are constructed in PyTorch using `torch.nn`, with special emphasis on **`nn.Module`, layers, activations, model structure, forward passes, parameters, tensor shapes, and common interview questions**.

Do NOT make loss functions, optimizers, or full training loops the main focus yet. Those will be covered in later notebooks.

---

# Main Learning Goals

By the end of the notebook, the learner should be comfortable using and explaining:

```python
import torch.nn as nn

nn.Module
nn.Linear
nn.Sequential

nn.ReLU
nn.LeakyReLU
nn.GELU
nn.Sigmoid
nn.Tanh
nn.Softmax

nn.Dropout

nn.BatchNorm1d
nn.BatchNorm2d
nn.LayerNorm

nn.Flatten

nn.Embedding

nn.Conv1d
nn.Conv2d
nn.MaxPool2d
nn.AdaptiveAvgPool2d
```

The learner should also understand:

- how to subclass `nn.Module`
- why `super().__init__()` is needed
- what `forward()` does
- what happens when calling `model(x)`
- model parameters
- trainable vs non-trainable parameters
- `model.parameters()`
- `model.named_parameters()`
- parameter shapes
- input and output shapes
- hidden layers
- activation functions
- logits
- why neural networks need nonlinear activations
- `nn.Sequential`
- custom modules
- dropout
- normalization
- embedding layers
- basic convolutional layers
- how tensor shapes flow through a model

Mark these as especially important:

```text
🔥 Interview Essential
```

- `nn.Module`
- `__init__`
- `super().__init__()`
- `forward()`
- `nn.Linear`
- activation functions
- logits
- `model.parameters()`
- `nn.Sequential`
- Dropout
- BatchNorm vs LayerNorm
- parameter counting
- shape reasoning through a network

---

# Teaching Style

Use:

- concise explanations
- runnable examples
- shape diagrams
- "predict the output shape" questions
- incomplete code exercises
- automatic validation using `assert`
- debugging exercises
- interview questions
- write-from-memory exercises
- a final model-building challenge

The learner should repeatedly **construct models themselves**, not just read completed code.

Use simple models at first, then gradually increase complexity.

---

# Notebook Structure

Use approximately this structure:

```text
# PyTorch Neural Networks

## 1. Setup
## 2. What Is torch.nn?
## 3. nn.Module
## 4. Building Your First Model
## 5. Understanding __init__ and forward
## 6. Calling model(x)
## 7. nn.Linear
## 8. Parameter Shapes
## 9. Activation Functions
## 10. Why Nonlinearity Matters
## 11. Building Multi-Layer Networks
## 12. nn.Sequential
## 13. Custom Modules
## 14. Model Parameters
## 15. Counting Parameters
## 16. Logits and Output Layers
## 17. Dropout
## 18. Batch Normalization
## 19. Layer Normalization
## 20. BatchNorm vs LayerNorm
## 21. Flatten
## 22. Embedding Layers
## 23. Introduction to Conv2d
## 24. Pooling Layers
## 25. Shape Reasoning Through Networks
## 26. Common Model Patterns
## 27. Common Mistakes
## 28. Debugging Exercises
## 29. Interview Questions
## 30. Knowledge Check Quiz
## 31. Write From Memory
## 32. Final Model-Building Challenge
## 33. Cheat Sheet
```

---

# 1. Setup

Start with:

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
```

Optionally print:

```python
print(torch.__version__)
```

---

# 2. What Is `torch.nn`?

Explain simply:

> `torch.nn` contains PyTorch's building blocks for neural networks, such as layers, activation functions, normalization layers, and base classes for creating models.

Show examples:

```python
nn.Linear
nn.ReLU
nn.Conv2d
nn.Dropout
```

Explain that most models are subclasses of:

```python
nn.Module
```

---

# 3. `nn.Module`

Mark:

```text
🔥 Interview Essential
```

Explain:

> `nn.Module` is the base class for neural-network models and layers in PyTorch.

Show:

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
```

Explain each part:

```text
class SimpleModel(nn.Module)
→ model inherits PyTorch model behavior

__init__()
→ define layers and model components

super().__init__()
→ initialize nn.Module properly

forward()
→ define how input flows through the model
```

Do not make this overly object-oriented.

---

# 4. Why `super().__init__()`?

Create a short dedicated explanation.

Explain that:

```python
super().__init__()
```

initializes the parent `nn.Module` machinery so PyTorch can correctly register:

- parameters
- child modules
- buffers
- hooks

For interview purposes, concise answer:

> It initializes the `nn.Module` base class so PyTorch can track the model's layers and parameters properly.

Mark:

```text
🔥 Interview Essential
```

Add a short quiz question later.

---

# 5. First Model

Use:

```python
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)
```

Instantiate:

```python
model = LinearModel()
```

Input:

```python
x = torch.randn(5, 3)
```

Run:

```python
output = model(x)
print(output.shape)
```

Expected:

```text
(5, 1)
```

Explain:

```text
input:
(5, 3)

Linear(3, 1)

output:
(5, 1)
```

Interpret:

```text
5 → batch size
3 → input features
1 → output feature
```

---

# 6. First Exercise

Ask learner to create:

```text
Input features: 4
Output features: 2
```

Starter:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        pass
```

Validation:

```python
model = MyModel()

x = torch.randn(8, 4)
output = model(x)

assert isinstance(model, nn.Module)
assert output.shape == (8, 2), "Expected output shape (8, 2)"

print("✅ Correct!")
```

---

# 7. What Happens When You Call `model(x)`?

Mark:

```text
🔥 Interview Essential
```

Explain that users normally write:

```python
output = model(x)
```

rather than:

```python
output = model.forward(x)
```

Explain simply:

> Calling `model(x)` invokes PyTorch's `nn.Module.__call__()` machinery, which then calls `forward()` and also handles functionality such as hooks.

Interview answer:

> `forward()` defines the computation, but we normally call `model(x)` so PyTorch's module machinery can run correctly.

Do not go too deeply into hooks.

---

# 8. `nn.Linear`

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
nn.Linear(in_features, out_features)
```

Example:

```python
layer = nn.Linear(10, 4)

x = torch.randn(32, 10)

y = layer(x)

print(y.shape)
```

Expected:

```text
(32, 4)
```

Explain:

```text
(batch, in_features)
        ↓
Linear(10, 4)
        ↓
(batch, out_features)
```

Show mathematical intuition:

```text
y = xWᵀ + b
```

Do not overemphasize matrix orientation details.

---

# 9. Linear Layer Parameter Shapes

Teach:

```python
layer = nn.Linear(10, 4)
```

Inspect:

```python
print(layer.weight.shape)
print(layer.bias.shape)
```

Expected:

```text
weight → (4, 10)
bias   → (4,)
```

Mark:

```text
🔥 Interview Useful
```

Explain:

> `nn.Linear(in_features, out_features)` stores weight with shape `(out_features, in_features)`.

Create several exercises.

Example:

```python
layer = nn.Linear(128, 64)
```

Ask:

```text
What is weight.shape?
What is bias.shape?
```

Expected:

```text
(64, 128)
(64,)
```

---

# 10. Parameter Counting

Teach how many trainable parameters are in:

```python
nn.Linear(10, 4)
```

Calculate manually:

```text
weights:
10 × 4 = 40

biases:
4

total:
44
```

Show programmatically:

```python
sum(p.numel() for p in layer.parameters())
```

Create at least 8 parameter-counting exercises.

Examples:

```text
Linear(100, 50)
Linear(50, 10)
```

Ask for individual and total counts.

Mark this as interview useful because model-size reasoning is common.

---

# 11. Activation Functions

Introduce:

```python
nn.ReLU()
nn.LeakyReLU()
nn.GELU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax()
```

For each:

- short explanation
- simple example
- typical use
- important interview note

Do not spend equal time on every activation.

Prioritize:

```text
ReLU
GELU
Sigmoid
Softmax
```

---

# 12. ReLU

Teach:

```python
relu = nn.ReLU()

x = torch.tensor([-2.0, 0.0, 3.0])

relu(x)
```

Expected:

```text
tensor([0., 0., 3.])
```

Explain:

```text
ReLU(x) = max(0, x)
```

Mention:

- widely used in hidden layers
- simple and efficient
- can suffer from dead ReLUs

Keep dead ReLU explanation short.

---

# 13. LeakyReLU

Teach:

```python
nn.LeakyReLU(negative_slope=0.01)
```

Explain that negative values retain a small slope rather than becoming exactly zero.

Mention it can reduce the dead-ReLU problem.

Keep brief.

---

# 14. GELU

Teach:

```python
nn.GELU()
```

Explain:

> GELU is a smooth nonlinear activation widely used in transformer architectures.

Mention:

```text
Transformers → GELU is common
```

Do not derive the GELU equation.

---

# 15. Sigmoid

Teach:

```python
nn.Sigmoid()
```

Explain that outputs are between:

```text
0 and 1
```

Common use:

```text
binary probability-like outputs
```

But add important interview note:

> When using `BCEWithLogitsLoss`, do not manually apply sigmoid before the loss because the loss already includes the sigmoid operation internally.

The loss function will be covered later, so keep this note concise.

---

# 16. Tanh

Teach:

```python
nn.Tanh()
```

Range:

```text
-1 to 1
```

Mention historical/common use in recurrent neural networks.

Keep concise.

---

# 17. Softmax

Teach:

```python
nn.Softmax(dim=1)
```

Example:

```python
softmax = nn.Softmax(dim=1)

logits = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 0.0]
])

probs = softmax(logits)
```

Show:

```python
probs.sum(dim=1)
```

Expected approximately:

```text
tensor([1., 1.])
```

Explain:

> For classification output shaped `(batch, classes)`, `dim=1` usually means normalize across classes.

Add important note:

> For training with `CrossEntropyLoss`, pass raw logits to the loss. Do not manually apply softmax first.

Mark:

```text
🔥 Interview Essential
```

---

# 18. What Are Logits?

Create a dedicated section.

Mark:

```text
🔥 Interview Essential
```

Explain:

> Logits are the raw, unnormalized scores produced by a model before probability transformations such as softmax or sigmoid.

Example:

```python
model = nn.Linear(5, 3)

x = torch.randn(4, 5)

logits = model(x)

print(logits.shape)
```

Shape:

```text
(4, 3)
```

Interpret:

```text
4 samples
3 class scores per sample
```

Explain:

```text
logits
→ softmax
→ probabilities
```

But during common classification training:

```text
logits
→ CrossEntropyLoss
```

---

# 19. Why Do Neural Networks Need Activations?

Mark:

```text
🔥 Interview Essential
```

Explain:

> Without nonlinear activations, stacking multiple linear layers still behaves like one linear transformation.

Show conceptual example:

```text
Linear
→ Linear
→ Linear
```

without activation is still effectively linear.

Compare:

```text
Linear
→ ReLU
→ Linear
```

which can learn nonlinear relationships.

Do not provide a formal proof unless very short.

---

# 20. Multi-Layer Network

Build:

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

Use:

```python
x = torch.randn(64, 10)

output = model(x)
```

Expected shape:

```text
(64, 2)
```

Show shape flow:

```text
(64, 10)
     ↓ Linear(10, 32)
(64, 32)
     ↓ ReLU
(64, 32)
     ↓ Linear(32, 2)
(64, 2)
```

Create shape-prediction exercises.

---

# 21. Exercise: Build an MLP

Ask learner to build:

```text
Input: 20 features

Hidden layer 1: 64
ReLU

Hidden layer 2: 32
ReLU

Output: 5 classes
```

Validation:

```python
model = MyMLP()

x = torch.randn(16, 20)
output = model(x)

assert output.shape == (16, 5)

print("✅ Correct!")
```

Also validate that the model contains at least three `nn.Linear` layers.

---

# 22. `nn.Sequential`

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)
```

Explain:

> `nn.Sequential` passes the output of each layer directly into the next layer.

Visualize:

```text
Input
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
Output
```

Compare custom `nn.Module` vs `nn.Sequential`.

Use a concise table:

```text
Sequential
→ convenient for simple straight-line models

Custom nn.Module
→ better for complex logic, branching, multiple inputs, skip connections
```

---

# 23. Exercise: Convert to Sequential

Provide:

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
```

Ask learner to create an equivalent:

```python
model = nn.Sequential(...)
```

Validation should check output shape and module types.

---

# 24. Custom Modules

Show a reusable custom block:

```python
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))
```

Then:

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = LinearBlock(10, 32)
        self.block2 = LinearBlock(32, 16)
        self.output = nn.Linear(16, 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.output(x)
```

Explain that PyTorch encourages composing modules from smaller modules.

---

# 25. Model Parameters

Teach:

```python
model.parameters()
```

Example:

```python
for param in model.parameters():
    print(param.shape)
```

Then:

```python
model.named_parameters()
```

Example:

```python
for name, param in model.named_parameters():
    print(name, param.shape)
```

Explain that these parameters are what optimizers later update.

Mark:

```text
🔥 Interview Essential
```

---

# 26. Trainable Parameter Count

Teach:

```python
total_params = sum(
    p.numel()
    for p in model.parameters()
)
```

and trainable only:

```python
trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)
```

Connect to the previous Autograd notebook.

Ask learner to freeze one layer and compare:

```text
total parameters
vs
trainable parameters
```

---

# 27. `children()` and `modules()`

Briefly demonstrate:

```python
model.children()
model.modules()
```

Explain:

```text
children()
→ immediate child modules

modules()
→ recursively includes nested modules
```

Keep this interview-useful but not a major section.

---

# 28. Dropout

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
nn.Dropout(p=0.5)
```

Explain:

> During training, dropout randomly zeroes some activations to reduce overfitting.

Show:

```python
dropout = nn.Dropout(p=0.5)

x = torch.ones(10)

dropout.train()
print(dropout(x))
```

Then:

```python
dropout.eval()
print(dropout(x))
```

Explain the key distinction:

```text
model.train()
→ dropout active

model.eval()
→ dropout disabled
```

Do not deeply teach `train()` and `eval()` yet, but introduce this relationship because it is important.

---

# 29. Why Dropout Output Is Scaled

Optional short explanation:

> PyTorch scales the remaining activations during training so expected activation magnitude remains approximately consistent between training and evaluation.

Keep this short.

---

# 30. Batch Normalization

Teach:

```python
nn.BatchNorm1d()
nn.BatchNorm2d()
```

Start with:

```python
bn = nn.BatchNorm1d(10)

x = torch.randn(32, 10)

y = bn(x)
```

Explain:

> Batch normalization normalizes activations using statistics calculated across a batch and learns scale and shift parameters.

Mention common use:

```text
MLP → BatchNorm1d
CNN → BatchNorm2d
```

Do not dive deeply into the exact formulas.

Mark:

```text
🔥 Interview Essential
```

---

# 31. BatchNorm and Train/Eval Modes

Explain that BatchNorm behaves differently during training and evaluation.

Training:

```text
uses batch statistics
updates running statistics
```

Evaluation:

```text
uses stored running statistics
```

This is why:

```python
model.eval()
```

matters during evaluation.

This should be a strong interview note.

---

# 32. Layer Normalization

Teach:

```python
nn.LayerNorm()
```

Example:

```python
ln = nn.LayerNorm(128)

x = torch.randn(32, 20, 128)

y = ln(x)
```

Explain:

> `LayerNorm(128)` normalizes across the final feature dimension of size 128 for each sample/token independently.

Mention:

```text
Transformers → LayerNorm is extremely common
```

Mark:

```text
🔥 Interview Essential for modern AI
```

---

# 33. BatchNorm vs LayerNorm

Create a clear interview comparison.

Use a concise table:

| Feature | BatchNorm | LayerNorm |
|---|---|---|
| Depends on batch statistics | Yes | No |
| Behavior changes in train/eval | Yes | Usually no meaningful running-stat change |
| Common in CNNs | Very common | Less common |
| Common in Transformers | Rare | Very common |
| Small-batch sensitivity | Higher | Lower |

Explain simply:

> BatchNorm normalizes using statistics across the batch, while LayerNorm normalizes features within each individual sample/token.

Create several interview questions about this.

---

# 34. Flatten

Teach:

```python
nn.Flatten()
```

Example:

```python
flatten = nn.Flatten()

x = torch.randn(32, 3, 28, 28)

y = flatten(x)
```

Expected:

```text
(32, 2352)
```

Explain that this is commonly used when connecting convolutional outputs to fully connected layers.

---

# 35. Embedding Layers

Mark:

```text
🔥 Interview Essential for NLP / LLM roles
```

Teach:

```python
nn.Embedding(num_embeddings, embedding_dim)
```

Example:

```python
embedding = nn.Embedding(
    num_embeddings=10000,
    embedding_dim=128
)

tokens = torch.tensor([
    [5, 20, 8],
    [9, 4, 2]
])

output = embedding(tokens)

print(output.shape)
```

Expected:

```text
(2, 3, 128)
```

Explain shape transformation:

```text
token IDs:
(batch, sequence)

(2, 3)

        ↓ Embedding(10000, 128)

embeddings:
(batch, sequence, embedding_dim)

(2, 3, 128)
```

Explain parameters:

```text
embedding weight shape:
(10000, 128)
```

---

# 36. What Does an Embedding Layer Do?

Explain:

> An embedding layer maps integer IDs to learned dense vectors.

Examples:

- word IDs
- token IDs
- user IDs
- item IDs
- categorical IDs

Make clear that input to `nn.Embedding` should generally contain integer indices.

Add dtype note:

```text
commonly torch.long / int64
```

---

# 37. Embedding Parameter Count

Example:

```python
nn.Embedding(50000, 768)
```

Ask:

```text
How many parameters?
```

Expected:

```text
50,000 × 768
```

Calculate programmatically.

Create 3–5 embedding parameter-count questions.

---

# 38. Introduction to Convolution

Do not make this a full CNN notebook.

Teach enough `Conv2d` syntax and shape reasoning for interviews.

Use:

```python
nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3
)
```

Input:

```python
x = torch.randn(32, 3, 28, 28)
```

Output:

```python
conv = nn.Conv2d(3, 16, kernel_size=3)

y = conv(x)

print(y.shape)
```

Expected:

```text
(32, 16, 26, 26)
```

because no padding is used.

Explain:

```text
batch stays 32

channels:
3 → 16

spatial:
28 → 26
```

---

# 39. Conv2d Shape Formula

Teach a simplified form:

```text
output_size =
floor(
    (input + 2×padding - kernel_size) / stride
) + 1
```

For normal dilation = 1.

Use examples.

Example:

```text
input  = 28
kernel = 3
padding = 0
stride = 1

output = 26
```

Another:

```text
input = 28
kernel = 3
padding = 1
stride = 1

output = 28
```

Mark:

```text
🔥 Interview Useful
```

Create at least 10 convolution shape questions.

---

# 40. Padding

Teach:

```python
nn.Conv2d(
    3,
    16,
    kernel_size=3,
    padding=1
)
```

Explain that for:

```text
kernel=3
stride=1
padding=1
```

height and width remain unchanged.

This is a common CNN pattern.

---

# 41. Stride

Teach:

```python
nn.Conv2d(
    3,
    16,
    kernel_size=3,
    stride=2,
    padding=1
)
```

Explain that stride > 1 reduces spatial resolution.

Ask learners to predict output sizes.

---

# 42. Conv2d Parameter Count

Teach parameter count for:

```python
nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3
)
```

Weights:

```text
16 × 3 × 3 × 3
```

Biases:

```text
16
```

Total:

```text
16 × 3 × 3 × 3 + 16
```

Create several exercises.

---

# 43. Max Pooling

Teach:

```python
nn.MaxPool2d(kernel_size=2)
```

Example:

```python
pool = nn.MaxPool2d(2)

x = torch.randn(32, 16, 28, 28)

y = pool(x)
```

Expected:

```text
(32, 16, 14, 14)
```

Explain:

- batch unchanged
- channels unchanged
- spatial size reduced

---

# 44. Adaptive Average Pooling

Teach:

```python
nn.AdaptiveAvgPool2d((1, 1))
```

Example:

```python
pool = nn.AdaptiveAvgPool2d((1, 1))

x = torch.randn(32, 128, 7, 7)

y = pool(x)
```

Expected:

```text
(32, 128, 1, 1)
```

Explain why this is useful:

> It can reduce any spatial resolution to a fixed output shape.

Mention common CNN classifier pattern:

```text
Conv features
↓
AdaptiveAvgPool2d((1,1))
↓
Flatten
↓
Linear
```

---

# 45. Simple CNN Example

Build:

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

Use input:

```python
x = torch.randn(8, 3, 32, 32)
```

Show shape flow:

```text
(8, 3, 32, 32)
↓ Conv
(8, 16, 32, 32)
↓ Pool
(8, 16, 16, 16)
↓ Conv
(8, 32, 16, 16)
↓ Pool
(8, 32, 8, 8)
↓ Flatten
(8, 2048)
↓ Linear
(8, 10)
```

This should be an important shape reasoning example.

---

# 46. Shape Reasoning Through Networks

Create a dedicated section with at least 20 questions.

Examples:

### Question 1

```python
nn.Linear(20, 64)
```

Input:

```text
(32, 20)
```

Output:

```text
?
```

Expected:

```text
(32, 64)
```

### Question 2

```python
nn.Embedding(10000, 256)
```

Input:

```text
(16, 50)
```

Output:

```text
(16, 50, 256)
```

### Question 3

```python
nn.Conv2d(3, 32, kernel_size=3, padding=1)
```

Input:

```text
(8, 3, 64, 64)
```

Output:

```text
(8, 32, 64, 64)
```

### Question 4

```python
nn.MaxPool2d(2)
```

Input:

```text
(8, 32, 64, 64)
```

Output:

```text
(8, 32, 32, 32)
```

### Question 5

```python
nn.Flatten()
```

Input:

```text
(16, 64, 7, 7)
```

Output:

```text
(16, 3136)
```

Use assert validation where practical.

---

# 47. Common Model Patterns

Create a concise section showing several reusable patterns.

## Tabular Classification

```python
nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, num_classes)
)
```

## Binary Classification

```python
nn.Sequential(
    nn.Linear(num_features, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
```

Explain output is usually one logit per sample.

## Multiclass Classification

```python
nn.Linear(hidden_size, num_classes)
```

Output:

```text
(batch, num_classes)
```

## Token Embedding

```python
nn.Embedding(vocab_size, hidden_size)
```

## CNN Classifier

```text
Conv
→ ReLU
→ Pool
→ Conv
→ ReLU
→ Pool
→ Flatten
→ Linear
```

---

# 48. Binary vs Multiclass Output Shapes

Mark:

```text
🔥 Interview Essential
```

Explain:

### Binary classification

Common:

```text
output shape:
(batch, 1)
```

one raw logit per sample.

### Multiclass classification

```text
output shape:
(batch, num_classes)
```

one raw logit per class.

Example:

```text
32 samples
10 classes

output:
(32, 10)
```

This prepares for the loss-functions notebook.

---

# 49. Common Mistake — Applying Activation in Wrong Place

Show:

```python
class Model(nn.Module):
    ...
    def forward(self, x):
        x = self.fc(x)
        return torch.softmax(x, dim=1)
```

Explain:

> This is not always wrong, but during training with `CrossEntropyLoss`, the model should normally return raw logits instead of applying softmax.

Similarly:

> For `BCEWithLogitsLoss`, the model should return raw logits rather than applying sigmoid.

Mark:

```text
🔥 Interview Essential
```

---

# 50. Common Mistakes

Include at least these:

## Mistake 1 — Forgetting `super().__init__()`

Explain model registration issues.

## Mistake 2 — Wrong `in_features`

Example:

```python
nn.Linear(20, 10)
```

but input shape:

```text
(32, 30)
```

## Mistake 3 — Wrong flatten size

Common in CNNs.

## Mistake 4 — Confusing `forward()` with calling `model.forward()`

Explain why `model(x)` is preferred.

## Mistake 5 — Adding softmax before `CrossEntropyLoss`

## Mistake 6 — Adding sigmoid before `BCEWithLogitsLoss`

## Mistake 7 — Wrong embedding input dtype

## Mistake 8 — Wrong image tensor layout

PyTorch normally expects:

```text
NCHW
(batch, channels, height, width)
```

not:

```text
NHWC
```

## Mistake 9 — Forgetting ReLU / nonlinear activation

Explain linear-stack problem.

## Mistake 10 — Mixing BatchNorm dimensions

Example:

```python
nn.BatchNorm1d(64)
```

should match relevant feature dimension.

## Mistake 11 — Assuming Dropout behaves the same in evaluation mode.

## Mistake 12 — Wrong Conv2d channel count.

---

# 51. Debugging Exercises

Create at least 15 debugging exercises.

Example:

```python
model = nn.Linear(10, 3)

x = torch.randn(32, 8)

output = model(x)
```

Ask:

> Why does this fail?

Expected:

```text
The input has 8 features but the layer expects 10.
```

Another:

```python
embedding = nn.Embedding(1000, 64)

tokens = torch.tensor([
    [1.0, 2.0, 3.0]
])

embedding(tokens)
```

Ask:

> What is wrong?

Expected:

```text
Embedding indices should use an integer dtype such as torch.long.
```

Another:

```python
conv = nn.Conv2d(3, 16, 3)

x = torch.randn(8, 32, 32, 3)

conv(x)
```

Ask:

> What is wrong?

Expected:

```text
The tensor is NHWC, while Conv2d expects NCHW by default.
```

Another:

```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.Linear(32, 64),
    nn.Linear(64, 2)
)
```

Ask:

> What conceptual problem might this model have?

Expected:

```text
There are no nonlinear activations between the linear layers.
```

---

# 52. Automatic Validation Format

For model-building exercises use cells like:

```python
# Exercise:
# Build a model:
# 10 -> 32 -> ReLU -> 2

model = None
```

Then:

```python
assert isinstance(model, nn.Module), "model must be an nn.Module"

x = torch.randn(4, 10)
output = model(x)

assert output.shape == (4, 2), "Expected output shape (4, 2)"

linear_layers = [
    module
    for module in model.modules()
    if isinstance(module, nn.Linear)
]

assert len(linear_layers) >= 2, "Expected at least two Linear layers"

print("✅ Correct!")
```

Avoid requiring one exact implementation when several correct model designs are acceptable.

---

# 53. Interview Questions

Create approximately 30 interview questions.

Include:

1. What is `nn.Module`?
2. Why do PyTorch models inherit from `nn.Module`?
3. Why do we call `super().__init__()`?
4. What is the purpose of `forward()`?
5. Why should we use `model(x)` instead of `model.forward(x)`?
6. What does `nn.Linear` do?
7. What are the weight and bias shapes of `nn.Linear(10, 4)`?
8. How do you count model parameters?
9. What is a hidden layer?
10. Why are activation functions needed?
11. Why is ReLU commonly used?
12. What is the dead ReLU problem?
13. What is GELU and where is it commonly used?
14. What is a logit?
15. What is the difference between logits and probabilities?
16. Why should a model often return logits during training?
17. What is `nn.Sequential`?
18. When would you prefer a custom `nn.Module` over `nn.Sequential`?
19. What does `model.parameters()` return?
20. What does `named_parameters()` return?
21. What does dropout do?
22. How does dropout behave during training vs evaluation?
23. What is BatchNorm?
24. What is LayerNorm?
25. What is the difference between BatchNorm and LayerNorm?
26. Why is LayerNorm common in transformers?
27. What does `nn.Embedding` do?
28. What input dtype does an embedding layer usually expect?
29. What does `nn.Conv2d` do?
30. What is the common image tensor layout in PyTorch?
31. How do you calculate Conv2d output size?
32. What does MaxPool do?
33. What does AdaptiveAvgPool do?
34. What is the difference between binary and multiclass model output shapes?
35. Why can stacking linear layers without activations be redundant?

For each, provide:

### Short Interview Answer

A concise answer suitable for speaking.

### Detailed Explanation

A slightly deeper explanation.

Use collapsible `<details>` sections.

---

# 54. Knowledge Check Quiz

Create approximately 25 multiple-choice questions.

Focus heavily on:

- output shapes
- parameter shapes
- parameter count
- activations
- logits
- `nn.Module`
- `Sequential`
- Dropout
- BatchNorm
- LayerNorm
- Embedding
- Conv2d

Example:

```text
For:

nn.Linear(20, 5)

what is the weight shape?

A. (20, 5)
B. (5, 20)
C. (20,)
D. (5,)
```

Correct:

```text
B
```

Another:

```text
Input shape:

(32, 100)

Layer:

nn.Linear(100, 10)

Output shape?

A. (100, 10)
B. (32, 100)
C. (32, 10)
D. (10, 32)
```

Correct:

```text
C
```

Put answers in a separate answer section.

---

# 55. Write From Memory

Create approximately 20 prompts.

Examples:

> Define a PyTorch model class inheriting from `nn.Module`.

> Create a linear layer from 10 input features to 3 outputs.

> Write a model with `10 → 32 → ReLU → 2`.

> Create the same model using `nn.Sequential`.

> Iterate through model parameters.

> Print all named parameters.

> Count total model parameters.

> Create a dropout layer with probability 0.3.

> Create a BatchNorm1d layer for 64 features.

> Create a LayerNorm layer for hidden size 768.

> Create an embedding layer for a vocabulary of 50,000 tokens with embedding size 512.

> Create a Conv2d layer from 3 channels to 32 channels with kernel size 3 and padding 1.

> Create a 2×2 max pooling layer.

> Flatten a CNN output before a Linear layer.

Each should include executable validation.

---

# 56. Parameter Count Challenge

Create a section with 10 quick model-size questions.

Examples:

```python
nn.Linear(128, 64)
```

Ask total parameters.

Another:

```python
nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)
```

Ask total trainable parameters.

Another:

```python
nn.Embedding(30000, 256)
```

Ask parameter count.

Another:

```python
nn.Conv2d(3, 16, 3)
```

Ask parameter count including bias.

Provide programmatic checks after manual calculation.

---

# 57. Shape Challenge

Create:

```text
## 10-Minute Neural Network Shape Challenge
```

Include 15 rapid questions involving:

- Linear
- Embedding
- Flatten
- Conv2d
- MaxPool
- Sequential networks

Example:

```text
Input:
(32, 784)

Linear(784, 128)
ReLU
Linear(128, 10)

Final shape?
```

Expected:

```text
(32, 10)
```

Another:

```text
Input:
(16, 3, 32, 32)

Conv2d(3, 16, kernel=3, padding=1)
MaxPool2d(2)
Conv2d(16, 32, kernel=3, padding=1)
MaxPool2d(2)

Final feature-map shape?
```

Expected:

```text
(16, 32, 8, 8)
```

---

# 58. Final Model-Building Challenge

Create a final project where the learner builds several models.

## Part A — Tabular Classifier

Requirements:

```text
Input features: 20

20
↓
64
↓ ReLU
Dropout(0.2)
↓
32
↓ ReLU
↓
5 logits
```

Input:

```python
x = torch.randn(16, 20)
```

Expected output:

```text
(16, 5)
```

Validate architecture and output shape.

---

## Part B — NLP Embedding Model

Build a model that accepts:

```text
(batch, sequence)
```

token IDs.

Architecture:

```text
Embedding(vocab_size=10000, embedding_dim=128)
↓
mean pooling across sequence dimension
↓
Linear(128, 3)
```

Input:

```python
tokens = torch.randint(0, 10000, (8, 20))
```

Expected:

```text
(8, 3)
```

Ask learner to implement the pooling in `forward()`:

```python
x = x.mean(dim=1)
```

Validate output.

---

## Part C — Small CNN

Build:

```text
Input:
(batch, 3, 32, 32)

Conv2d 3 → 16, kernel=3, padding=1
ReLU
MaxPool2d(2)

Conv2d 16 → 32, kernel=3, padding=1
ReLU
MaxPool2d(2)

Flatten

Linear → 64
ReLU

Linear → 10 logits
```

Expected output:

```text
(batch, 10)
```

Validation should test multiple batch sizes.

---

# 59. Architecture Reading Exercise

Create several exercises where the learner is shown code and must describe it.

Example:

```python
model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)
)
```

Ask:

1. Input feature count?
2. Hidden size?
3. Number of output logits?
4. Where is nonlinearity applied?
5. What does dropout do?
6. What shape comes out for `(32, 100)` input?

This is useful for reading real AI repositories.

---

# 60. Model Summary Utility

Optionally create a small helper function without external dependencies:

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable
```

Use it throughout the notebook.

Do NOT require packages such as `torchinfo`.

---

# 61. Final Cheat Sheet

End with a compact cheat sheet.

Include:

```python
import torch.nn as nn

# Basic model
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

model = Model()

# Linear
nn.Linear(in_features, out_features)

# Activations
nn.ReLU()
nn.LeakyReLU()
nn.GELU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=1)

# Sequential
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

# Dropout
nn.Dropout(p=0.5)

# Normalization
nn.BatchNorm1d(64)
nn.BatchNorm2d(32)
nn.LayerNorm(768)

# Flatten
nn.Flatten()

# Embedding
nn.Embedding(
    num_embeddings=50000,
    embedding_dim=768
)

# Conv
nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    padding=1
)

# Pool
nn.MaxPool2d(2)
nn.AdaptiveAvgPool2d((1, 1))

# Parameters
model.parameters()
model.named_parameters()

# Count
sum(p.numel() for p in model.parameters())

# Trainable count
sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)
```

Add a concise mental model:

```text
Model
=
Layers
+
Parameters
+
forward()
```

And:

```text
Input
↓
Linear / Conv / Embedding
↓
Activation
↓
more layers
↓
raw logits
```

---

# 62. Interview Quick Reference

End with a short table:

```text
nn.Module
→ base class for PyTorch models

forward()
→ defines computation

nn.Linear
→ fully connected transformation

ReLU
→ common hidden activation

GELU
→ common transformer activation

Dropout
→ regularization

BatchNorm
→ batch-based normalization

LayerNorm
→ feature-based normalization, common in transformers

Embedding
→ integer IDs → learned vectors

Conv2d
→ image feature extraction

Logits
→ raw model output scores
```

---

# 63. Completion Checklist

End with:

```text
## Before Moving to 05_losses_optimizers.ipynb
```

Add:

- [ ] I can create a model using `nn.Module`.
- [ ] I know why `super().__init__()` is required.
- [ ] I understand `forward()`.
- [ ] I know why `model(x)` is preferred over `model.forward(x)`.
- [ ] I can use `nn.Linear`.
- [ ] I can predict Linear output shapes.
- [ ] I can calculate Linear parameter counts.
- [ ] I understand why activation functions are needed.
- [ ] I know the main uses of ReLU, GELU, Sigmoid, and Softmax.
- [ ] I understand what logits are.
- [ ] I can create models using `nn.Sequential`.
- [ ] I can inspect and count model parameters.
- [ ] I understand Dropout.
- [ ] I understand BatchNorm at a practical level.
- [ ] I understand LayerNorm at a practical level.
- [ ] I can explain BatchNorm vs LayerNorm.
- [ ] I can use `nn.Embedding`.
- [ ] I can predict embedding output shapes.
- [ ] I understand basic `Conv2d` syntax.
- [ ] I can predict basic convolution and pooling shapes.
- [ ] I can build a simple MLP, embedding classifier, and CNN.

---

# 64. Difficulty Labels

Use:

```text
🟢 Easy
🟡 Medium
🔴 Hard
```

Most content should be Easy to Medium.

Treat these as Medium:

- Conv2d output-size calculations
- convolution parameter counting
- normalization differences
- custom modules
- embedding parameter counts

Avoid advanced topics such as:

- custom CUDA layers
- complex hooks
- distributed modules
- custom autograd Functions
- detailed transformer architecture

Those belong later.

---

# 65. Quality Requirements

The notebook must:

- be a valid `.ipynb`
- execute successfully from top to bottom
- work on CPU-only systems
- require only PyTorch
- not require internet access
- contain no empty placeholder sections
- include runnable examples
- include automatic `assert` validation
- contain approximately 60–80 exercises/questions
- heavily emphasize model construction
- heavily emphasize shape reasoning
- heavily emphasize parameter shapes and counts
- include interview-focused explanations
- avoid prematurely turning into a full training tutorial

The notebook should take approximately **2–3 hours** to study thoroughly.

The core learning principle should be:

```text
Do not just read neural-network architectures.

Repeatedly:

see a requirement
      ↓
build the model
      ↓
predict tensor shapes
      ↓
run the model
      ↓
inspect parameters
      ↓
fix mistakes
      ↓
explain the architecture
```

Finally save the completed notebook as:

```text
04_neural_networks.ipynb
```