Create a polished, production-quality **interactive PyTorch learning website** designed specifically for someone preparing for **AI Engineer / Machine Learning Engineer interviews**.

The website should teach PyTorch through four main methods:

1. **PyTorch syntax cheat sheets**
2. **Short coding exercises**
3. **Quizzes and interview questions**
4. **Mini projects**

The goal is NOT to execute real PyTorch code in the browser. The site should instead teach syntax and concepts and validate whether the learner has entered the expected PyTorch code or an equivalent acceptable answer.

---

# 1. Product Goal

Build a website that helps a learner repeatedly practice the PyTorch syntax and concepts most likely to appear in AI Engineer interviews.

The experience should feel like a combination of:

* PyTorch documentation
* LeetCode-style exercises
* Anki-style repetition
* Interactive cheat sheets
* Interview preparation
* Small guided ML projects

The site should emphasize **active recall** rather than passive reading.

A user should be able to:

* Browse PyTorch syntax by topic.
* Search for syntax quickly.
* See concise explanations and examples.
* Hide examples and try to recall the syntax.
* Type one or several lines of PyTorch code as answers.
* Submit an answer and immediately see whether it is correct.
* See hints when stuck.
* Reveal the expected solution.
* Practice questions repeatedly.
* Track which topics they have completed.
* Take interview-style quizzes.
* Complete guided mini projects.
* Review weak areas.

---

# 2. Target Audience

Target users are developers preparing for roles such as:

* AI Engineer
* Machine Learning Engineer
* Applied AI Engineer
* Deep Learning Engineer
* Generative AI Engineer

Assume the learner already understands:

* Python basics
* NumPy basics
* Basic machine learning
* Basic neural-network concepts

Do NOT spend too much time teaching basic Python.

Focus on **practical PyTorch knowledge that an AI Engineer is expected to know**.

---

# 3. Technology Stack

Use a modern frontend stack.

Preferred:

* Next.js
* React
* TypeScript
* Tailwind CSS

Use a clean component architecture.

The website should work locally with:

```bash
npm install
npm run dev
```

Do not require a database for the initial version.

Store:

* exercises
* quiz questions
* cheat-sheet content
* project content

as structured TypeScript/JSON data.

Use browser `localStorage` for:

* exercise completion
* quiz scores
* bookmarked topics
* weak questions
* progress
* user preferences

Structure the project so a real database can be added later.

---

# 4. Important Constraint: Do NOT Execute PyTorch

The website must NOT execute Python or PyTorch code.

There should be:

* no Python backend
* no Jupyter kernel
* no Pyodide
* no remote code execution
* no Docker execution service

Instead, coding exercises should validate the learner's answer using safe text-based validation.

Possible techniques:

* whitespace normalization
* ignoring unnecessary spaces
* ignoring trailing semicolons
* accepted-answer lists
* token-based comparison
* regular expressions where appropriate
* checking for required syntax elements
* flexible ordering when order is irrelevant

For example, these should ideally be considered equivalent:

```python
x = torch.tensor([1,2,3])
```

and

```python
x = torch.tensor([1, 2, 3])
```

Do NOT require an exact character-for-character match when equivalent formatting should be accepted.

Each exercise object can contain fields such as:

```ts
{
  id: "tensor-001",
  topic: "Tensor Creation",
  difficulty: "Easy",
  question: "Create a tensor containing 1, 2 and 3.",
  starterCode: "",
  acceptedAnswers: [
    "x = torch.tensor([1, 2, 3])",
    "x=torch.tensor([1,2,3])"
  ],
  requiredPatterns: [
    "torch.tensor"
  ],
  hint: "Use torch.tensor().",
  explanation: "...",
  solution: "x = torch.tensor([1, 2, 3])"
}
```

Build the validation architecture so additional accepted solutions can easily be added.

---

# 5. Main Navigation

Create navigation similar to:

```text
PyTorch Prep
├── Dashboard
├── Cheat Sheet
├── Practice
├── Interview Questions
├── Mini Projects
├── Review
└── Progress
```

Desktop:

* left sidebar navigation

Mobile:

* collapsible menu

The current section should be clearly highlighted.

---

# 6. Dashboard

Create a dashboard showing:

### Progress overview

Example:

```text
Overall Progress       42%

Tensor Basics          90%
Autograd               70%
Neural Networks        55%
Training Loops         40%
DataLoaders            35%
GPU / Devices          70%
Model Saving           20%
```

Also show cards:

* Questions completed
* Current streak
* Accuracy
* Topics mastered
* Questions needing review
* Mini projects completed

Add:

### Continue Learning

Example:

```text
Continue: Autograd
12 / 18 exercises completed
```

### Weak Areas

Automatically identify sections where the learner has low accuracy.

Example:

```text
Weak Areas

1. tensor dimensions
2. DataLoader
3. model.train() vs model.eval()
4. optimizer.zero_grad()
```

### Daily Practice

Create a short practice session containing approximately:

* 5 syntax questions
* 3 concept questions
* 2 interview questions

---

# 7. PyTorch Cheat Sheet

This is one of the most important sections.

Create a searchable, categorized PyTorch cheat sheet.

Each item should contain:

* concept name
* short explanation
* syntax
* small example
* expected result where useful
* interview note where useful

Use syntax-highlighted code blocks.

Add buttons:

* Copy
* Practice
* Bookmark
* Hide / Show Code

Allow the learner to collapse and expand sections.

---

# 8. Cheat Sheet Topics

Include comprehensive coverage of the following.

## A. Imports

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
```

---

## B. Tensor Creation

Include:

```python
torch.tensor()
torch.zeros()
torch.ones()
torch.empty()
torch.full()
torch.arange()
torch.linspace()
torch.rand()
torch.randn()
torch.randint()
torch.eye()
torch.zeros_like()
torch.ones_like()
torch.rand_like()
```

Explain when each is commonly used.

---

## C. Tensor Properties

Include:

```python
x.shape
x.size()
x.ndim
x.dtype
x.device
x.numel()
```

---

## D. Data Types

Include important PyTorch dtypes:

```python
torch.float32
torch.float64
torch.float16
torch.bfloat16
torch.int64
torch.int32
torch.bool
```

Explain why `torch.long` / `torch.int64` is commonly needed for classification targets and embeddings.

---

## E. Type Conversion

Include:

```python
x.float()
x.long()
x.int()
x.bool()
x.to(torch.float32)
```

---

## F. Tensor Reshaping

This should be an especially important section.

Include:

```python
x.reshape()
x.view()
x.flatten()
x.squeeze()
x.unsqueeze()
x.permute()
x.transpose()
```

Clearly explain the difference between:

```text
reshape
view
permute
transpose
squeeze
unsqueeze
```

Include shape examples.

Example:

```python
x = torch.randn(32, 3, 224, 224)
```

Ask learners to identify dimensions.

---

## G. Indexing and Slicing

Include:

```python
x[0]
x[:, 0]
x[:, :, 0]
x[1:4]
x[..., 0]
```

Include boolean indexing:

```python
x[x > 0]
```

---

## H. Combining Tensors

Include:

```python
torch.cat()
torch.stack()
```

Explain the difference very clearly.

Include exercises asking:

> What will the resulting shape be?

---

## I. Mathematical Operations

Include:

```python
x + y
x * y
torch.sum()
torch.mean()
torch.max()
torch.min()
torch.argmax()
torch.argmin()
torch.abs()
torch.sqrt()
torch.exp()
torch.log()
```

---

## J. Matrix Operations

Include:

```python
torch.matmul()
x @ y
torch.mm()
torch.bmm()
torch.dot()
```

Explain their different use cases.

---

# 9. Broadcasting

Create a dedicated broadcasting section.

Explain using simple shape examples:

```text
(3, 4)
+
(4,)
```

and:

```text
(32, 10)
+
(10,)
```

Include shape prediction exercises.

---

# 10. Autograd

This is an important interview topic.

Include:

```python
requires_grad=True
loss.backward()
x.grad
torch.no_grad()
tensor.detach()
```

Explain computational graphs.

Example:

```python
x = torch.tensor(2.0, requires_grad=True)

y = x ** 2

y.backward()

print(x.grad)
```

Explain why the result is 4.

Include questions such as:

* What does `requires_grad=True` do?
* What does `.backward()` do?
* Where are gradients stored?
* Why do gradients accumulate?
* What is `torch.no_grad()`?
* What does `.detach()` do?

---

# 11. Neural Network Modules

Teach:

```python
nn.Module
```

Show the standard model structure:

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)
```

Explain:

* `__init__`
* `super().__init__()`
* `forward`
* parameters
* calling the model

---

# 12. Important Layers

Create syntax cards for:

```python
nn.Linear
nn.ReLU
nn.LeakyReLU
nn.GELU
nn.Sigmoid
nn.Tanh
nn.Dropout
nn.BatchNorm1d
nn.BatchNorm2d
nn.LayerNorm
nn.Embedding
nn.Conv1d
nn.Conv2d
nn.MaxPool2d
nn.AdaptiveAvgPool2d
nn.LSTM
nn.GRU
nn.MultiheadAttention
nn.Sequential
```

Focus more heavily on commonly used interview topics.

---

# 13. Activation Functions

Compare:

* ReLU
* LeakyReLU
* GELU
* Sigmoid
* Tanh
* Softmax

Explain where each is commonly used.

Include:

```python
torch.softmax(x, dim=1)
```

and explain what `dim` means.

---

# 14. Loss Functions

Include:

```python
nn.MSELoss()
nn.L1Loss()
nn.CrossEntropyLoss()
nn.BCELoss()
nn.BCEWithLogitsLoss()
```

Explain when each is used.

Important interview concept:

Explain that:

```python
nn.CrossEntropyLoss()
```

expects raw logits, so you normally should NOT manually apply softmax before it.

Similarly explain why:

```python
BCEWithLogitsLoss
```

combines sigmoid and BCE.

---

# 15. Optimizers

Include:

```python
optim.SGD()
optim.Adam()
optim.AdamW()
```

Example:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Explain:

* learning rate
* momentum
* weight decay
* model parameters

---

# 16. Training Loop

Make this one of the most important sections.

Teach the standard pattern:

```python
model.train()

for X, y in dataloader:
    optimizer.zero_grad()

    pred = model(X)

    loss = loss_fn(pred, y)

    loss.backward()

    optimizer.step()
```

The learner should eventually be able to write this from memory.

Break it into individual exercises:

> Put model in training mode.

Expected:

```python
model.train()
```

> Clear previous gradients.

Expected:

```python
optimizer.zero_grad()
```

> Run forward pass.

Expected:

```python
pred = model(X)
```

> Calculate loss.

Expected:

```python
loss = loss_fn(pred, y)
```

> Calculate gradients.

Expected:

```python
loss.backward()
```

> Update parameters.

Expected:

```python
optimizer.step()
```

Then have a final exercise asking the learner to write the complete training loop.

---

# 17. Validation / Evaluation Loop

Teach:

```python
model.eval()

with torch.no_grad():
    for X, y in dataloader:
        pred = model(X)
```

Explain the differences between:

```python
model.train()
model.eval()
torch.no_grad()
```

This should have dedicated interview questions because these concepts are frequently confused.

---

# 18. Dataset and DataLoader

Teach custom Dataset:

```python
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

Teach:

```python
DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

Explain:

* dataset
* batch
* batch size
* shuffling
* epochs
* workers

Include:

```python
num_workers
pin_memory
```

at an intermediate level.

---

# 19. Device / GPU

Teach:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

And:

```python
model.to(device)
X = X.to(device)
y = y.to(device)
```

Also include modern patterns such as:

```python
torch.cuda.is_available()
```

Explain the common interview mistake:

> Moving the model to GPU but forgetting to move the input tensors.

Also discuss at a conceptual level:

* CPU
* CUDA
* MPS
* GPU memory

---

# 20. Saving and Loading Models

Teach:

```python
torch.save(model.state_dict(), "model.pth")
```

and:

```python
model.load_state_dict(
    torch.load("model.pth")
)
```

Explain why saving `state_dict()` is generally preferred.

Also cover:

```python
model.state_dict()
optimizer.state_dict()
```

---

# 21. Reproducibility

Teach:

```python
torch.manual_seed(42)
```

Mention:

```python
torch.cuda.manual_seed_all(42)
```

Explain why experiments can still have some nondeterminism.

---

# 22. Gradient Management

Include:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Explain why gradients accumulate in PyTorch.

Also introduce:

```python
torch.nn.utils.clip_grad_norm_
```

---

# 23. Learning Rate Schedulers

Include examples such as:

```python
torch.optim.lr_scheduler.StepLR
torch.optim.lr_scheduler.ReduceLROnPlateau
torch.optim.lr_scheduler.CosineAnnealingLR
```

Keep this section intermediate rather than overwhelming beginners.

---

# 24. Mixed Precision

Create an advanced/intermediate section explaining mixed precision.

Discuss:

* float32
* float16
* bfloat16
* automatic mixed precision

Show modern PyTorch AMP usage where appropriate.

Explain why mixed precision can:

* reduce GPU memory
* increase training speed
* sometimes cause numerical issues

---

# 25. Common Tensor Shape Patterns

Create a special interview-oriented section showing common shapes.

Examples:

### Tabular

```text
(batch_size, features)
```

### Images

```text
(batch_size, channels, height, width)
```

### NLP / Transformers

```text
(batch_size, sequence_length, hidden_size)
```

### Classification output

```text
(batch_size, num_classes)
```

Create many shape reasoning exercises.

---

# 26. Practice Section

Create a dedicated exercise browser.

Allow filtering by:

* topic
* difficulty
* completed / incomplete
* correct / incorrect
* bookmarked
* interview importance

Difficulty levels:

```text
Easy
Medium
Hard
```

Interview importance:

```text
High
Medium
Low
```

---

# 27. Exercise Interface

Each exercise page/card should display:

### Question

Example:

> Create a 3 × 4 tensor filled with zeros.

### Answer editor

Use a code-editor style input.

Example:

```python
____________________________
```

### Actions

Buttons:

```text
Check Answer
Hint
Show Answer
Next
Bookmark
```

If correct:

```text
✓ Correct

torch.zeros(3, 4) creates a tensor containing
3 rows and 4 columns.
```

If wrong:

```text
✗ Not quite.

Hint:
Look at torch.zeros().
```

Do NOT immediately show the full answer after the first incorrect attempt.

---

# 28. Exercise Types

Support multiple exercise formats.

## Type 1: Write the syntax

Question:

> Convert tensor `x` to float32.

Expected:

```python
x = x.float()
```

or another valid equivalent.

---

## Type 2: Fill in the blank

```python
loss.________()
```

Expected:

```text
backward
```

---

## Type 3: Predict the shape

```python
x = torch.randn(32, 3, 224, 224)
y = x.flatten(1)
```

Ask:

```text
What is y.shape?
```

---

## Type 4: Multiple choice

Example:

> Which function clears accumulated gradients?

* optimizer.step()
* optimizer.zero_grad()
* loss.backward()
* model.eval()

---

## Type 5: Spot the bug

Example:

```python
model.to(device)

for X, y in loader:
    prediction = model(X)
```

Ask:

> What is likely wrong when `model` is on CUDA?

Expected concept:

```text
X must also be moved to the same device.
```

---

## Type 6: Order the steps

Ask learner to arrange:

```text
optimizer.step()
loss.backward()
optimizer.zero_grad()
prediction = model(X)
loss = loss_fn(prediction, y)
```

into the correct training order.

---

## Type 7: Complete the code

Provide:

```python
for X, y in loader:
    optimizer.zero_grad()

    ________

    loss = loss_fn(pred, y)

    ________

    optimizer.step()
```

Expected:

```python
pred = model(X)
loss.backward()
```

---

# 29. Progressive Exercise System

Organize exercises into levels.

### Level 1 — Tensor Basics

Approximately 30 exercises.

### Level 2 — Tensor Manipulation

Approximately 30 exercises.

### Level 3 — Autograd

Approximately 20 exercises.

### Level 4 — Neural Networks

Approximately 30 exercises.

### Level 5 — Training

Approximately 30 exercises.

### Level 6 — Data Pipeline

Approximately 20 exercises.

### Level 7 — GPU and Performance

Approximately 20 exercises.

### Level 8 — Interview Challenge

Approximately 30 exercises.

The initial site should therefore support approximately **150–200 exercises**.

It is okay to initially populate fewer exercises if the architecture makes adding more easy, but provide a meaningful seed dataset.

---

# 30. Interview Questions Section

Create a separate section specifically for AI Engineer interviews.

Categories:

* PyTorch fundamentals
* tensors
* autograd
* neural networks
* optimization
* losses
* training loops
* GPU
* performance
* debugging
* transformers
* practical ML

Create questions such as:

> Why does PyTorch accumulate gradients by default?

> What is the difference between `model.eval()` and `torch.no_grad()`?

> What happens if you forget `optimizer.zero_grad()`?

> What is the difference between `reshape()` and `view()`?

> What is the difference between `torch.stack()` and `torch.cat()`?

> Why shouldn't you apply softmax before `CrossEntropyLoss`?

> What is a computational graph?

> What does `.detach()` do?

> What is the difference between `model.parameters()` and `state_dict()`?

> Why do we use mini-batches?

> What happens when model parameters and inputs are on different devices?

> What does `requires_grad` mean?

> How would you debug NaN loss?

> What causes exploding gradients?

> What causes vanishing gradients?

> What is gradient clipping?

> Why use AdamW instead of Adam?

> What does weight decay do?

> What is batch normalization?

> What is layer normalization?

> Why is LayerNorm common in transformers?

---

# 31. Interview Answer Mode

Allow each question to have:

```text
Think
Reveal Answer
I Knew This
Need Review
```

After revealing the answer, show:

### Short interview answer

A concise answer that could realistically be spoken during an interview.

### Detailed explanation

More detail for studying.

### Related PyTorch syntax

Relevant code.

This is important because the learner needs to practice both coding and verbally explaining concepts.

---

# 32. Mini Projects

Include several guided mini projects.

These projects should NOT actually train models in the website.

Instead, users fill in important pieces of PyTorch code.

---

# Mini Project 1 — Linear Regression

Goal:

Build a PyTorch linear regression model.

Steps:

1. Create tensors.
2. Create `nn.Linear`.
3. Define MSE loss.
4. Define optimizer.
5. Write training loop.
6. Perform inference.

Learner should fill in important lines.

---

# Mini Project 2 — Binary Classification

Create a small neural network using:

```python
nn.Linear
nn.ReLU
nn.BCEWithLogitsLoss
Adam
```

Ask questions about:

* logits
* sigmoid
* thresholding
* loss

---

# Mini Project 3 — MNIST Classifier

Guide the learner through creating:

```text
Input
↓
Linear
↓
ReLU
↓
Linear
↓
10 logits
```

Include exercises for:

* model
* DataLoader
* CrossEntropyLoss
* optimizer
* training
* evaluation
* argmax

---

# Mini Project 4 — CNN Image Classifier

Include:

```python
nn.Conv2d
nn.ReLU
nn.MaxPool2d
nn.Flatten
nn.Linear
```

Focus heavily on tensor shapes.

Ask learners to calculate output dimensions.

---

# Mini Project 5 — Text Classification

Include:

```python
nn.Embedding
pooling
nn.Linear
```

Teach:

```text
token IDs
→ embeddings
→ representation
→ classifier
```

---

# Mini Project 6 — Transformer Building Blocks

Keep this interview-focused rather than implementing an entire transformer.

Teach:

* embedding
* positional information
* Q, K, V
* attention
* multi-head attention
* LayerNorm
* residual connection
* feed-forward network

Include syntax involving:

```python
nn.Embedding
nn.MultiheadAttention
nn.LayerNorm
nn.Linear
```

---

# Mini Project 7 — Complete Training Pipeline

Make the learner construct an entire training pipeline from memory:

```text
Dataset
↓
DataLoader
↓
Model
↓
Loss
↓
Optimizer
↓
Training Loop
↓
Validation
↓
Save Model
```

This should be the final project.

---

# 33. Interview Challenge Mode

Create a special timed mode.

Example:

```text
PyTorch Interview Challenge

20 Questions
30 Minutes
```

Mix:

* syntax
* debugging
* concepts
* code completion
* tensor shapes

At the end show:

```text
Score: 16 / 20

Tensor Basics       100%
Autograd             75%
Training              80%
GPU                   50%

Recommended Review:
- torch.no_grad()
- device placement
- detach()
```

---

# 34. Review System

Create a "Review" page.

Automatically add questions where the learner:

* answered incorrectly
* used hints
* revealed the solution
* manually selected "Need Review"

Support filters:

```text
All
Wrong Answers
Used Hint
Revealed Answer
Bookmarked
```

Add a:

```text
Practice Weak Questions
```

button.

---

# 35. Spaced Repetition

Implement a simple local spaced-repetition mechanism.

Questions can have states:

```text
New
Learning
Review
Mastered
```

After answering, allow:

```text
Again
Hard
Good
Easy
```

Use this information to determine when the question should appear again.

It does not have to implement a sophisticated algorithm initially.

---

# 36. Search

Add global search.

The user should be able to search:

```text
CrossEntropyLoss
reshape
zero_grad
DataLoader
CUDA
detach
```

and immediately find:

* cheat-sheet entries
* exercises
* interview questions

---

# 37. Keyboard Shortcuts

Add useful keyboard shortcuts.

For example:

```text
Cmd/Ctrl + K       Search

Cmd/Ctrl + Enter   Check answer

H                  Hint

N                  Next question
```

Do not trigger shortcuts while the user is typing when inappropriate.

---

# 38. Code Display

Use proper Python syntax highlighting.

Code blocks should resemble a modern developer IDE.

Include:

* line numbers where useful
* Copy button
* clear monospace font
* good contrast

For answer input, use either:

* Monaco Editor

or a lightweight syntax-highlighted editor if Monaco adds too much complexity.

---

# 39. Design Style

Use a clean developer-oriented design.

Style inspiration:

* modern documentation
* Linear
* Vercel
* GitHub
* LeetCode
* PyTorch documentation

Avoid excessive gradients and decorative elements.

Prioritize:

* readability
* speed
* code
* learning

Use rounded cards but keep the UI professional.

---

# 40. Dark Mode

Support:

```text
Light
Dark
System
```

Dark mode should look especially good because developers may spend long sessions on the site.

---

# 41. Responsive Design

The website must work well on:

* desktop
* laptop
* tablet
* mobile

On mobile:

* sidebar becomes a menu
* code should remain readable
* cards should stack vertically

---

# 42. Content Architecture

Do NOT hardcode hundreds of exercises directly inside React components.

Create structured data files.

For example:

```text
data/
├── cheatsheet/
│   ├── tensors.ts
│   ├── autograd.ts
│   ├── neural-networks.ts
│   ├── training.ts
│   └── gpu.ts
│
├── exercises/
│   ├── tensors.ts
│   ├── autograd.ts
│   ├── training.ts
│   └── gpu.ts
│
├── interview/
│   ├── fundamentals.ts
│   ├── training.ts
│   └── debugging.ts
│
└── projects/
    ├── linear-regression.ts
    ├── classifier.ts
    └── cnn.ts
```

Make the system easy to extend.

---

# 43. Exercise Data Model

Design a reusable TypeScript interface.

For example:

```ts
interface Exercise {
  id: string;
  title: string;
  topic: string;
  difficulty: "easy" | "medium" | "hard";
  importance: "high" | "medium" | "low";

  type:
    | "code"
    | "fill-blank"
    | "multiple-choice"
    | "shape"
    | "debug"
    | "ordering";

  question: string;

  starterCode?: string;

  acceptedAnswers?: string[];

  requiredPatterns?: string[];

  options?: string[];

  correctOption?: number;

  hint?: string;

  solution: string;

  explanation: string;

  interviewNote?: string;
}
```

Improve the interface if necessary.

---

# 44. Validation Engine

Create a reusable answer-validation system rather than validation logic inside individual components.

For simple code answers:

1. Trim leading/trailing whitespace.
2. Normalize spaces.
3. Normalize line endings.
4. Ignore optional semicolons.
5. Compare against accepted solutions.
6. Allow required-pattern checks.
7. Support custom validators.

For example:

```ts
validateAnswer(userAnswer, exercise)
```

should return something like:

```ts
{
  correct: true,
  feedback: "Correct!",
  matchedSolution: 0
}
```

Do not use `eval()`.

Do not execute arbitrary user code.

---

# 45. Progress Data

Create TypeScript models such as:

```ts
interface ExerciseProgress {
  exerciseId: string;
  attempts: number;
  correct: boolean;
  usedHint: boolean;
  revealedAnswer: boolean;
  lastAttempt: string;
  reviewState: "new" | "learning" | "review" | "mastered";
}
```

Persist this in `localStorage`.

---

# 46. Important Interview Topics

Visually mark particularly important PyTorch topics with:

```text
🔥 Interview Essential
```

Examples:

* tensor shapes
* broadcasting
* autograd
* training loop
* optimizer.zero_grad()
* loss.backward()
* optimizer.step()
* model.train()
* model.eval()
* torch.no_grad()
* Dataset
* DataLoader
* device handling
* CrossEntropyLoss
* BCEWithLogitsLoss
* saving/loading models
* detach()
* gradient accumulation
* batch size
* learning rate
* Adam vs SGD
* normalization
* dropout

---

# 47. "Explain This Code" Cards

Include cards showing short PyTorch snippets.

Example:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Ask:

> Explain what these three lines do.

After revealing, show the explanation.

---

# 48. Common Mistakes Section

Create a section containing common PyTorch mistakes.

Examples:

### Forgetting zero_grad

```python
loss.backward()
optimizer.step()
```

Explain gradient accumulation.

### Softmax before CrossEntropyLoss

Show why this is usually incorrect.

### Model on GPU but tensor on CPU

Show the device mismatch issue.

### Forgetting model.eval()

Explain dropout / batch normalization behavior.

### Forgetting torch.no_grad()

Explain unnecessary computation graphs during inference.

### Incorrect tensor shape

Explain typical:

```text
(batch, features)
```

vs:

```text
(features, batch)
```

problems.

### Wrong target dtype

Explain why `CrossEntropyLoss` normally expects class labels as integer/long tensors.

---

# 49. Tensor Shape Playground

Add a page where learners can practice tensor-shape reasoning without executing PyTorch.

For example:

```text
Starting shape:

(32, 3, 64, 64)

Operation:

x.flatten(1)

Your answer:

[____________]
```

Expected:

```text
(32, 12288)
```

Generate questions from predefined transformations.

Focus on:

* reshape
* flatten
* unsqueeze
* squeeze
* transpose
* permute
* cat
* stack
* Linear
* Conv2d
* pooling

---

# 50. Training Loop Builder

Create another interactive learning activity.

Show blocks such as:

```text
optimizer.zero_grad()

pred = model(X)

loss = loss_fn(pred, y)

loss.backward()

optimizer.step()
```

Allow the learner to arrange them into the correct order.

Also include optional blocks containing incorrect operations.

---

# 51. "Write From Memory" Mode

Create a mode specifically for memorization.

Example:

```text
Write a basic PyTorch training loop from memory.
```

Large code editor.

The validator should look for key elements:

```text
model.train()
optimizer.zero_grad()
model(...)
loss_fn(...)
loss.backward()
optimizer.step()
```

Do NOT require the exact variable names.

For example:

```python
outputs = network(inputs)
```

and:

```python
pred = model(X)
```

should both potentially be acceptable if the structure is correct.

For these complex exercises, use required-pattern or semantic-rule validation rather than exact string comparison.

---

# 52. Topic Mastery

Calculate mastery based on factors such as:

```text
accuracy
number of attempts
hint usage
solution reveals
review performance
```

Display:

```text
Tensor Basics        ██████████ 100%
Autograd             ███████░░░ 70%
Training Loop        █████░░░░░ 50%
GPU                   ███░░░░░░░ 30%
```

---

# 53. Learning Path

Create a recommended learning path:

```text
1. Tensor Basics
        ↓
2. Tensor Manipulation
        ↓
3. Autograd
        ↓
4. nn.Module
        ↓
5. Loss Functions
        ↓
6. Optimizers
        ↓
7. Training Loop
        ↓
8. Dataset + DataLoader
        ↓
9. GPU
        ↓
10. CNN
        ↓
11. Transformer Basics
        ↓
12. Interview Challenge
```

Show completion status.

---

# 54. PyTorch vs NumPy Interview Comparisons

Include a concise section covering:

```python
numpy.array
torch.tensor
```

and concepts such as:

* GPU support
* automatic differentiation
* neural-network modules
* converting between NumPy and PyTorch

Examples:

```python
torch.from_numpy(arr)
tensor.numpy()
```

---

# 55. Transformer-Relevant PyTorch

Because the site is for modern AI Engineer interviews, include basic PyTorch knowledge relevant to LLMs.

Cover:

```python
nn.Embedding
nn.Linear
nn.LayerNorm
nn.MultiheadAttention
torch.softmax
torch.matmul
```

Also basic tensor concepts involving:

```text
batch
sequence
embedding dimension
attention heads
```

Example shape:

```text
(batch_size, sequence_length, embedding_dim)
```

Do NOT turn the website into a full transformer course.

Keep it focused on the PyTorch knowledge expected during interviews.

---

# 56. Performance Concepts

Create a concise intermediate section covering:

* vectorization
* batching
* avoiding unnecessary CPU ↔ GPU transfers
* `torch.no_grad()`
* mixed precision
* gradient accumulation
* DataLoader workers
* pinned memory

Do not go too deeply into CUDA kernel programming.

---

# 57. Debugging Questions

Include practical debugging scenarios.

Example:

```text
RuntimeError:
Expected all tensors to be on the same device
```

Ask what caused it.

Other scenarios:

```text
shape mismatch
NaN loss
loss not decreasing
CUDA out of memory
wrong target dtype
wrong output dimensions
```

These should resemble realistic AI Engineer interview questions.

---

# 58. Question Quality

Do NOT generate trivia simply to increase question count.

Questions should test understanding that actually matters when:

* writing PyTorch
* debugging models
* training models
* reading ML repositories
* discussing ML systems during interviews

Prefer questions such as:

> Why do we clear gradients?

over obscure API memorization.

---

# 59. Explanation Style

All explanations should be concise and beginner-friendly.

Prefer:

```text
optimizer.zero_grad()

Clears gradients from the previous training step.
PyTorch accumulates gradients by default.
```

Avoid unnecessarily academic language.

Where possible explain:

1. What it does.
2. Why it is needed.
3. When it is used.
4. Common interview mistake.

---

# 60. Interview Priority

Assign every topic an interview priority.

Example:

```text
CrossEntropyLoss
Priority: HIGH

torch.empty()
Priority: LOW
```

Use this to create a:

```text
High Priority Only
```

filter.

This allows someone with limited preparation time to focus on the most important concepts.

---

# 61. Rapid Revision Mode

Create a section called:

```text
15-Minute PyTorch Review
```

It should quickly review roughly 20–30 critical concepts.

Examples:

```text
Tensor creation
Tensor shapes
reshape
permute
matmul
autograd
nn.Module
Linear
CrossEntropyLoss
Adam
zero_grad
backward
step
train
eval
no_grad
Dataset
DataLoader
CUDA
state_dict
```

This should be suitable immediately before an interview.

---

# 62. UI Feedback

Use subtle animations for:

* correct answer
* wrong answer
* progress updates

For correct answer:

```text
✓ Correct
```

For incorrect:

```text
✗ Try again
```

Do not make the UI childish or overly game-like.

---

# 63. Accessibility

Include:

* keyboard navigation
* proper button labels
* semantic HTML
* adequate contrast
* readable font sizes
* accessible form inputs

---

# 64. Initial Content

Do not build an empty shell.

Populate the first version with meaningful content.

At minimum include approximately:

* 80+ cheat-sheet entries
* 75+ exercises
* 50+ interview questions
* 5+ mini projects
* 20+ common mistakes/debugging scenarios

Prioritize quality over artificial quantity.

The architecture must make expanding to several hundred questions straightforward.

---

# 65. Code Quality

Use:

* clean reusable React components
* TypeScript types
* sensible directory organization
* no unnecessary global state library
* no duplicated validation logic
* reusable exercise components
* reusable code display components
* reusable progress utilities

Comment complicated logic, but do not over-comment obvious JSX.

---

# 66. Suggested Component Structure

Consider components similar to:

```text
components/
├── layout/
│   ├── Sidebar.tsx
│   ├── Header.tsx
│   └── MobileNav.tsx
│
├── cheatsheet/
│   ├── CheatSheetCard.tsx
│   ├── CheatSheetSection.tsx
│   └── CodeExample.tsx
│
├── exercises/
│   ├── ExerciseCard.tsx
│   ├── CodeExercise.tsx
│   ├── MultipleChoiceExercise.tsx
│   ├── ShapeExercise.tsx
│   ├── DebugExercise.tsx
│   └── ExerciseFeedback.tsx
│
├── interview/
│   ├── InterviewQuestion.tsx
│   └── InterviewAnswer.tsx
│
├── progress/
│   ├── ProgressBar.tsx
│   ├── TopicProgress.tsx
│   └── StatsCard.tsx
│
└── common/
    ├── CodeBlock.tsx
    ├── Search.tsx
    └── Badge.tsx
```

You may improve this architecture where appropriate.

---

# 67. Pages

Create pages/routes approximately like:

```text
/
 /cheatsheet
 /cheatsheet/tensors
 /cheatsheet/autograd
 /cheatsheet/training

 /practice
 /practice/tensors
 /practice/autograd
 /practice/training

 /interview
 /projects
 /projects/[project]

 /review
 /progress
 /rapid-review
```

---

# 68. MVP Priorities

If implementing everything at once becomes too large, prioritize development in this order:

### Phase 1

Build:

* application shell
* sidebar
* dashboard
* cheat sheet
* exercise system
* validation engine
* local progress tracking

### Phase 2

Add:

* interview questions
* review system
* search
* topic mastery
* rapid revision

### Phase 3

Add:

* mini projects
* spaced repetition
* interview challenge
* training-loop builder
* tensor-shape playground

However, design the architecture from the beginning so these later features fit naturally.

---

# 69. Most Important Design Principle

The website should not become another passive tutorial.

Its core loop should be:

```text
See concept
      ↓
Hide syntax
      ↓
Recall it
      ↓
Type code
      ↓
Check answer
      ↓
Understand mistake
      ↓
Try again later
```

Optimize the entire experience around **active PyTorch recall for AI Engineer interviews**.

---

# 70. Example User Experience

A learner opens:

```text
Practice → Training Loop
```

Question:

> Write the PyTorch command that clears gradients before backpropagation.

They type:

```python
optimizer.zero_grad()
```

Website responds:

```text
✓ Correct

PyTorch accumulates gradients by default.

Typical training sequence:

optimizer.zero_grad()
pred = model(X)
loss = loss_fn(pred, y)
loss.backward()
optimizer.step()

🔥 Interview Essential
```

Then:

```text
Next Question →
```

Next question:

> Calculate gradients from `loss`.

Learner types:

```python
loss.backward()
```

Continue in this style.

---

# 71. Final Deliverable

Create the actual functioning website, not merely mockups.

I expect:

1. Complete Next.js/React/TypeScript project.
2. Responsive UI.
3. Dark/light mode.
4. Reusable content architecture.
5. PyTorch cheat-sheet content.
6. Interactive exercises.
7. Safe answer validation without executing Python.
8. Interview question system.
9. Local progress tracking.
10. Review system.
11. Mini-project framework.
12. Meaningful initial PyTorch content.
13. Clean, maintainable source code.
14. README explaining how to run and extend the site.

Do not leave important sections as empty placeholder cards.

Implement the core functionality and populate enough real PyTorch content that I can immediately use the website for **AI Engineer interview preparation**.

When making technical/content decisions, prioritize:

```text
Interview relevance
        >
Active practice
        >
Clarity
        >
Completeness
        >
Visual decoration
```
