Create a Jupyter Notebook named:

```text
03_autograd.ipynb
```

This notebook is the third part of a **PyTorch Interview Preparation** course for someone preparing for **AI Engineer / Machine Learning Engineer interviews**.

Assume the learner has already completed:

```text
01_tensor_basics.ipynb
02_tensor_operations.ipynb
```

and already understands:

- tensor creation
- shapes and dimensions
- reshaping
- broadcasting
- reductions
- matrix multiplication
- vectorized tensor operations
- classification-style tensor shapes

Do not spend significant time reteaching those topics.

The purpose of this notebook is to teach **PyTorch automatic differentiation (Autograd)** thoroughly enough that the learner can confidently use it in training code and explain it during AI Engineer interviews.

---

# Main Learning Goals

By the end of the notebook, the learner should understand and be able to use:

```python
requires_grad=True
tensor.requires_grad
tensor.grad
loss.backward()

torch.no_grad()
tensor.detach()

tensor.requires_grad_()
```

The learner should also understand:

- computational graphs
- forward pass
- backward pass
- gradients
- chain rule at an intuitive level
- leaf tensors
- gradient accumulation
- why gradients must be cleared
- scalar vs non-scalar backward calls
- stopping gradient tracking
- inference without gradients
- in-place operation problems
- detached tensors
- training-loop gradient flow
- frozen parameters
- common Autograd bugs

Mark the following as especially important:

```text
🔥 Interview Essential
```

- `requires_grad`
- `.backward()`
- `.grad`
- computational graph
- gradient accumulation
- `optimizer.zero_grad()`
- `torch.no_grad()`
- `.detach()`
- leaf tensors
- freezing parameters

---

# Teaching Style

Use:

- concise explanations
- runnable code examples
- manual gradient calculations
- prediction-before-running exercises
- automatic validation with `assert`
- debugging questions
- interview questions
- write-from-memory exercises
- a final mini training example

Do not make the notebook overly mathematical.

The learner should understand enough calculus to follow gradients, but this is a **PyTorch interview notebook**, not a calculus course.

Use simple equations such as:

```text
y = x²

dy/dx = 2x
```

to connect mathematical gradients to PyTorch.

---

# Notebook Structure

Use approximately:

```text
# PyTorch Autograd

## 1. Setup
## 2. What Is Autograd?
## 3. requires_grad
## 4. Computational Graphs
## 5. backward()
## 6. Accessing Gradients with .grad
## 7. Chain Rule Intuition
## 8. Gradients with Multiple Variables
## 9. Gradient Accumulation
## 10. Clearing Gradients
## 11. Leaf and Non-Leaf Tensors
## 12. torch.no_grad()
## 13. detach()
## 14. requires_grad_()
## 15. Freezing Parameters
## 16. Non-Scalar Outputs
## 17. Autograd and Neural Network Parameters
## 18. Manual Gradient Descent
## 19. Common Autograd Mistakes
## 20. Debugging Exercises
## 21. Interview Questions
## 22. Knowledge Check Quiz
## 23. Write From Memory
## 24. Final Autograd Challenge
## 25. Cheat Sheet
```

---

# 1. Setup

Start with:

```python
import torch

torch.manual_seed(42)
```

Print the PyTorch version if useful.

---

# 2. What Is Autograd?

Explain simply:

> Autograd is PyTorch's automatic differentiation system. It records tensor operations and automatically calculates gradients during the backward pass.

Explain the basic training flow:

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Loss
  ↓
backward()
  ↓
Gradients
  ↓
Optimizer updates parameters
```

Make clear:

```text
Forward pass
→ calculate predictions and loss

Backward pass
→ calculate gradients
```

Use this simple example:

```python
x = torch.tensor(2.0, requires_grad=True)

y = x ** 2

y.backward()

print(x.grad)
```

Expected:

```text
tensor(4.)
```

Explain:

```text
y = x²
dy/dx = 2x

x = 2
gradient = 4
```

Mark this section:

```text
🔥 Interview Essential
```

---

# 3. `requires_grad=True`

Teach:

```python
x = torch.tensor(3.0, requires_grad=True)
```

Explain:

> `requires_grad=True` tells PyTorch to track operations involving this tensor so gradients can later be computed.

Show:

```python
print(x.requires_grad)
```

Expected:

```text
True
```

Compare:

```python
a = torch.tensor(3.0)
b = torch.tensor(3.0, requires_grad=True)
```

Show:

```python
print(a.requires_grad)
print(b.requires_grad)
```

Explain that normal tensors do not track gradients unless needed.

---

# 4. First Exercise

Give:

```python
# Exercise:
# Create a scalar tensor containing 5.0 that tracks gradients.

x = None
```

Validation:

```python
assert isinstance(x, torch.Tensor), "x must be a PyTorch tensor"
assert x.shape == torch.Size([]), "x should be a scalar tensor"
assert x.item() == 5.0, "x should contain 5.0"
assert x.requires_grad, "Gradient tracking should be enabled"

print("✅ Correct!")
```

---

# 5. Computational Graphs

Mark:

```text
🔥 Interview Essential
```

Explain computational graphs intuitively.

Example:

```python
x = torch.tensor(2.0, requires_grad=True)

a = x * 3
b = a + 4
y = b ** 2
```

Visualize:

```text
x
│
× 3
│
a
│
+ 4
│
b
│
square
│
y
```

Explain that PyTorch remembers enough information about these operations to calculate gradients during the backward pass.

Do NOT claim that the graph is always permanently stored.

Mention that PyTorch normally uses a **dynamic computational graph**, which is built during execution.

Add interview note:

> PyTorch's graph is created dynamically during the forward pass, which makes normal Python control flow easy to use.

---

# 6. Inspecting `grad_fn`

Show:

```python
x = torch.tensor(2.0, requires_grad=True)

y = x * 3
z = y ** 2

print(x.grad_fn)
print(y.grad_fn)
print(z.grad_fn)
```

Explain:

- `x` is a leaf tensor, so its `grad_fn` is normally `None`.
- tensors produced by tracked operations have a `grad_fn`.

Keep this explanation simple.

---

# 7. `backward()`

Teach:

```python
loss.backward()
```

Explain:

> `.backward()` starts reverse-mode automatic differentiation from the output and calculates gradients for tensors that require them.

Show:

```python
x = torch.tensor(3.0, requires_grad=True)

y = x ** 2

y.backward()

print(x.grad)
```

Expected:

```text
6
```

Ask the learner to predict the gradient before running the cell.

---

# 8. `.grad`

Teach:

```python
x.grad
```

Explain that gradients are stored in the `.grad` attribute of appropriate leaf tensors.

Example:

```python
x = torch.tensor(4.0, requires_grad=True)

y = 3 * x ** 2

y.backward()

print(x.grad)
```

Manually calculate:

```text
y = 3x²

dy/dx = 6x

x = 4

gradient = 24
```

---

# 9. Gradient Prediction Exercises

Create at least 8 small exercises where the learner predicts gradients before running PyTorch.

Examples:

### Exercise

```python
x = torch.tensor(5.0, requires_grad=True)

y = 2 * x
```

Ask:

```text
What should dy/dx be?
```

Expected:

```text
2
```

### Exercise

```python
x = torch.tensor(3.0, requires_grad=True)

y = x ** 3
```

Expected gradient:

```text
27
```

because:

```text
dy/dx = 3x²
```

### Exercise

```python
x = torch.tensor(2.0, requires_grad=True)

y = 4 * x ** 2 + 3 * x + 1
```

Expected:

```text
19
```

because:

```text
dy/dx = 8x + 3
```

Use automatic validation after each.

---

# 10. Chain Rule Intuition

Explain the chain rule using a simple example.

```python
x = torch.tensor(2.0, requires_grad=True)

a = x * 3
y = a ** 2

y.backward()
```

Manual reasoning:

```text
a = 3x

y = a²

dy/da = 2a
da/dx = 3

dy/dx = dy/da × da/dx
```

At:

```text
x = 2
a = 6

dy/dx = 12 × 3 = 36
```

Show:

```python
print(x.grad)
```

Expected:

```text
36
```

Explain the chain rule visually:

```text
x ──×3──> a ──square──> y

gradient flows backward:

dy/dx
=
dy/da × da/dx
```

Do not go deeper into calculus than necessary.

---

# 11. Multiple Variables

Teach gradients with more than one variable.

Example:

```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = x * y

z.backward()

print(x.grad)
print(y.grad)
```

Expected:

```text
dz/dx = y = 3
dz/dy = x = 2
```

Add exercises such as:

```python
z = x**2 + y**2
```

Ask learner to calculate both gradients.

---

# 12. Gradient Accumulation

This should be one of the most important sections.

Mark:

```text
🔥 Interview Essential
```

Demonstrate:

```python
x = torch.tensor(2.0, requires_grad=True)

y = x ** 2
y.backward()

print(x.grad)
```

Expected:

```text
4
```

Then run another backward pass:

```python
y = x ** 2
y.backward()

print(x.grad)
```

Expected:

```text
8
```

Explain:

> PyTorch accumulates gradients instead of automatically replacing them.

This is why training loops normally contain:

```python
optimizer.zero_grad()
```

before:

```python
loss.backward()
```

Explain the usual sequence:

```text
optimizer.zero_grad()
      ↓
forward pass
      ↓
loss
      ↓
loss.backward()
      ↓
optimizer.step()
```

---

# 13. Why Does PyTorch Accumulate Gradients?

Explain practical reasons.

Mention that gradient accumulation allows workflows such as accumulating gradients across multiple mini-batches before taking an optimizer step.

Keep this concise.

Interview answer:

> PyTorch accumulates gradients into `.grad` by default, so gradients must normally be cleared between independent training steps. The behavior also allows intentional gradient accumulation across batches.

---

# 14. Clearing Gradients Manually

Before introducing optimizers deeply, demonstrate:

```python
x.grad.zero_()
```

Example:

```python
x = torch.tensor(2.0, requires_grad=True)

y = x ** 2
y.backward()

print(x.grad)

x.grad.zero_()

print(x.grad)
```

Then explain that neural network training generally uses:

```python
optimizer.zero_grad()
```

rather than manually clearing every parameter.

---

# 15. Exercise: Gradient Accumulation

Create an exercise where:

```python
x = torch.tensor(3.0, requires_grad=True)
```

and the learner performs two backward passes.

Ask them to predict:

```text
gradient after first backward
gradient after second backward
```

Then ask them to reset the gradient and verify it returns to zero.

---

# 16. Leaf Tensors

Mark:

```text
🔥 Interview Useful
```

Explain leaf tensors simply.

Example:

```python
x = torch.tensor(2.0, requires_grad=True)

y = x * 3
```

Here:

```text
x → leaf tensor
y → non-leaf tensor
```

Show:

```python
print(x.is_leaf)
print(y.is_leaf)
```

Explain:

> A leaf tensor is generally a tensor created directly by the user rather than produced as the result of another tracked operation.

Mention that model parameters are leaf tensors that require gradients.

Explain that `.grad` is normally populated for leaf tensors requiring gradients.

Do not overcomplicate edge cases.

---

# 17. Non-Leaf Tensor Gradients

Show:

```python
x = torch.tensor(2.0, requires_grad=True)

y = x * 3
z = y ** 2

z.backward()

print(x.grad)
print(y.grad)
```

Explain why `y.grad` is normally not retained automatically.

Mention:

```python
y.retain_grad()
```

can be used when gradients for intermediate tensors are explicitly needed.

Keep this as an intermediate concept.

---

# 18. `torch.no_grad()`

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
with torch.no_grad():
    prediction = model(x)
```

Before using a real model, demonstrate simply:

```python
x = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = x ** 2

print(y.requires_grad)
```

Expected:

```text
False
```

Explain:

> `torch.no_grad()` temporarily disables gradient tracking.

Common uses:

- validation
- inference
- evaluation
- operations where gradients are unnecessary

Explain benefits:

- less memory
- less computation
- no computational graph construction for those operations

---

# 19. `detach()`

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
detached = tensor.detach()
```

Example:

```python
x = torch.tensor(2.0, requires_grad=True)

y = x * 3

z = y.detach()

print(y.requires_grad)
print(z.requires_grad)
```

Explain:

> `.detach()` returns a tensor disconnected from the current computational graph.

Show that operations using the detached tensor will not backpropagate through the original history.

---

# 20. `no_grad()` vs `detach()`

Create a comparison table.

Explain:

```text
torch.no_grad()
→ disables gradient tracking for operations performed inside a block

detach()
→ creates a tensor disconnected from the existing graph
```

Example use cases:

```text
Validation/inference:
torch.no_grad()

Convert a particular tensor into a value that should no longer carry gradient history:
detach()
```

Create an interview question specifically asking the difference.

---

# 21. Exercise: `detach()`

Give:

```python
x = torch.tensor(5.0, requires_grad=True)
y = x * 2

answer = None
```

Ask:

> Create a tensor from `y` that does not track the computational graph.

Validation:

```python
assert isinstance(answer, torch.Tensor)
assert not answer.requires_grad
assert answer.item() == y.item()

print("✅ Correct!")
```

---

# 22. `requires_grad_()`

Teach:

```python
x.requires_grad_(True)
```

Example:

```python
x = torch.tensor([1.0, 2.0, 3.0])

print(x.requires_grad)

x.requires_grad_(True)

print(x.requires_grad)
```

Explain that methods ending in `_` typically modify tensors in place.

Briefly introduce the PyTorch convention:

```text
operation_
```

usually means an in-place operation.

Examples:

```python
zero_()
add_()
requires_grad_()
```

---

# 23. In-Place Operations and Autograd

Explain carefully that in-place modifications can cause problems when PyTorch needs old tensor values for gradient computation.

Show a safe demonstration without intentionally destroying the notebook state.

For example, explain that:

```python
x += 1
```

can be problematic in certain Autograd situations.

Do not encourage complicated in-place manipulation.

Interview advice:

> When debugging Autograd errors, check for in-place operations that modify tensors needed by the backward pass.

---

# 24. Freezing Parameters

Mark:

```text
🔥 Interview Essential
```

Explain transfer-learning style freezing.

Example:

```python
for param in model.parameters():
    param.requires_grad = False
```

Then selectively enable:

```python
for param in model.classifier.parameters():
    param.requires_grad = True
```

If a simple model is needed, create:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2)
)
```

Show:

```python
for param in model.parameters():
    print(param.requires_grad)
```

Then freeze the first layer.

Explain:

> Parameters with `requires_grad=False` will not receive gradients during backpropagation.

Relate this to:

- transfer learning
- fine-tuning
- frozen backbones
- adapters

---

# 25. Exercise: Freeze a Layer

Create:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2)
)
```

Ask the learner to freeze:

```text
model[0]
```

Validation should confirm:

```python
for p in model[0].parameters():
    assert not p.requires_grad
```

and verify the final layer is still trainable.

---

# 26. Non-Scalar Outputs

Explain that:

```python
backward()
```

works directly when the output is a scalar.

Example:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

y = x ** 2
```

Here:

```text
y.shape = (3,)
```

Calling:

```python
y.backward()
```

without supplying gradient information will fail.

Show a common solution:

```python
loss = y.sum()

loss.backward()
```

Then:

```python
print(x.grad)
```

Expected:

```text
tensor([2., 4., 6.])
```

Explain that ML losses are normally reduced to scalars, which is why:

```python
loss.backward()
```

usually works directly.

---

# 27. Optional Advanced Example: Explicit Gradient Argument

Briefly show:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

y = x ** 2

y.backward(torch.ones_like(y))
```

Explain conceptually that for non-scalar outputs we are effectively specifying how the output contributes to the final scalar quantity.

Keep this short.

Do not dive deeply into Jacobians.

---

# 28. Autograd With Neural Network Parameters

Introduce a tiny model:

```python
import torch.nn as nn

model = nn.Linear(2, 1)
```

Inspect parameters:

```python
for name, param in model.named_parameters():
    print(name)
    print(param.shape)
    print(param.requires_grad)
```

Use sample input:

```python
X = torch.tensor([[1.0, 2.0]])
target = torch.tensor([[1.0]])
```

Calculate:

```python
prediction = model(X)

loss = ((prediction - target) ** 2).mean()

loss.backward()
```

Then inspect:

```python
for name, param in model.named_parameters():
    print(name)
    print(param.grad)
```

Explain:

> `loss.backward()` automatically computes gradients for every trainable model parameter that contributed to the loss.

Mark:

```text
🔥 Interview Essential
```

---

# 29. Visualize Gradient Flow

Use a simple flow:

```text
X
│
▼
Linear layer
│
▼
Prediction
│
▼
Loss
│
▼
loss.backward()
│
├── gradient for weight
└── gradient for bias
```

Explain that the gradients do not update the parameters by themselves.

Make this distinction explicit:

```text
loss.backward()
→ calculates gradients

optimizer.step()
→ updates parameters
```

This is a very important interview distinction.

---

# 30. Manual Gradient Descent

Before fully introducing optimizers in the next notebook, show parameter updates manually.

Example:

```python
w = torch.tensor(0.0, requires_grad=True)

x = torch.tensor(2.0)
target = torch.tensor(4.0)

prediction = w * x

loss = (prediction - target) ** 2

loss.backward()

print(w.grad)
```

Then update manually:

```python
learning_rate = 0.1

with torch.no_grad():
    w -= learning_rate * w.grad

w.grad.zero_()
```

Explain why the update uses:

```python
torch.no_grad()
```

We do not want the optimizer-style parameter update itself to become part of the computational graph.

---

# 31. Build a Tiny Training Loop Manually

Create a minimal regression problem:

```python
X = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([2.0, 4.0, 6.0, 8.0])
```

Model:

```text
prediction = w * X
```

Initialize:

```python
w = torch.tensor(0.0, requires_grad=True)
```

Ask the learner to build a training loop manually using:

```text
forward prediction
loss
backward
parameter update
clear gradient
```

Use:

```python
for epoch in range(...):
```

without an optimizer.

At the end, `w` should approach approximately:

```text
2
```

This section should connect Autograd to actual training.

---

# 32. Explain the Training Cycle

Show:

```text
1. Forward pass
       ↓
2. Calculate loss
       ↓
3. Clear old gradients
       ↓
4. Backward pass
       ↓
5. Update parameters
       ↓
6. Repeat
```

Also show the common PyTorch implementation:

```python
optimizer.zero_grad()
prediction = model(X)
loss = loss_fn(prediction, y)
loss.backward()
optimizer.step()
```

Do not deeply teach optimizers yet.

State:

> Loss functions and optimizers will be covered in detail later.

---

# 33. Common Mistake — Calling `backward()` Repeatedly on Same Graph

Demonstrate conceptually:

```python
x = torch.tensor(2.0, requires_grad=True)

y = x ** 2

y.backward()
```

Then explain that calling:

```python
y.backward()
```

again on the exact same graph normally causes an error because intermediate graph information was released after the first backward pass.

Mention:

```python
retain_graph=True
```

exists:

```python
y.backward(retain_graph=True)
```

but should not normally be added just to silence an error.

Explain:

> If you unexpectedly need `retain_graph=True`, first check whether the code is reusing a computational graph incorrectly.

This is a useful debugging concept.

---

# 34. Common Mistakes

Create a dedicated section containing at least:

## Mistake 1 — Forgetting `requires_grad=True`

```python
x = torch.tensor(2.0)

y = x ** 2
```

Then expecting:

```python
x.grad
```

to contain a gradient.

---

## Mistake 2 — Forgetting `backward()`

Calculating loss does not automatically calculate gradients.

---

## Mistake 3 — Forgetting to clear gradients

Explain gradient accumulation.

---

## Mistake 4 — Confusing `backward()` and `step()`

```text
backward()
→ calculate gradients

step()
→ update parameters
```

---

## Mistake 5 — Using gradients during inference

Explain why:

```python
torch.no_grad()
```

is normally used.

---

## Mistake 6 — Detaching accidentally

If a required tensor is detached, gradients cannot flow back through it.

---

## Mistake 7 — In-place operations

Mention potential graph modification problems.

---

## Mistake 8 — Calling `.numpy()` directly on a gradient-tracking tensor

Teach this common pattern where appropriate:

```python
x.detach().cpu().numpy()
```

Explain each part:

```text
detach()
→ disconnect from Autograd

cpu()
→ NumPy expects CPU memory

numpy()
→ convert to ndarray
```

Do not require CUDA.

---

## Mistake 9 — Wrong assumption about `.grad`

Explain that `.grad` is typically populated for leaf tensors.

---

## Mistake 10 — Reusing a freed graph

Explain repeated backward calls.

---

# 35. Debugging Exercises

Create at least 15 debugging exercises.

Example:

```python
x = torch.tensor(3.0)

y = x ** 2

y.backward()
```

Ask:

> Why does this not work as expected?

Expected:

```text
x does not have requires_grad=True.
```

---

Another:

```python
x = torch.tensor(2.0, requires_grad=True)

for _ in range(3):
    y = x ** 2
    y.backward()

print(x.grad)
```

Ask:

> Why is the gradient 12 instead of 4?

Expected:

```text
Gradients accumulated across three backward passes.
```

---

Another:

```python
prediction = model(X)

with torch.no_grad():
    loss = loss_fn(prediction, y)

loss.backward()
```

Ask:

> What is wrong?

Explain that the loss was created without gradient tracking.

---

Another:

```python
x = torch.tensor(2.0, requires_grad=True)

y = x * 3
y = y.detach()
z = y ** 2
```

Ask:

> Will `z.backward()` propagate a gradient into `x`?

Expected:

```text
No.
```

---

# 36. Exercise Format

Use incomplete cells followed by validation cells.

Example:

```python
# Exercise:
# Calculate the gradient of y = x^3 at x = 2.

x = torch.tensor(2.0, requires_grad=True)

y = ...

# Perform backward pass
...
```

Validation:

```python
assert x.grad is not None, "No gradient was calculated"
assert torch.allclose(x.grad, torch.tensor(12.0)), "Check your function or backward pass"

print("✅ Correct!")
```

Use helpful assertions without revealing the exact line the learner must type.

---

# 37. Prediction Before Execution

Frequently use this pattern:

Markdown:

```text
Before running the next cell, predict:

1. What is x.grad?
2. Why?
```

Then provide code.

This should be used especially for:

- chain rule
- multiple variables
- gradient accumulation
- detach
- no_grad
- non-scalar tensors

---

# 38. Interview Questions

Create approximately 25 interview questions.

Include:

1. What is PyTorch Autograd?
2. What does `requires_grad=True` do?
3. What is a computational graph?
4. What is a dynamic computational graph?
5. What does `.backward()` do?
6. Where are gradients stored?
7. What is the difference between forward and backward pass?
8. Why does PyTorch accumulate gradients?
9. Why do we call `optimizer.zero_grad()`?
10. What does `.grad` contain?
11. What is a leaf tensor?
12. Why might a non-leaf tensor have `.grad=None`?
13. What does `torch.no_grad()` do?
14. When should `torch.no_grad()` be used?
15. What does `.detach()` do?
16. What is the difference between `detach()` and `no_grad()`?
17. What happens if you detach part of a computational graph?
18. How do you freeze model parameters?
19. Why would you freeze parameters?
20. What is the difference between `loss.backward()` and `optimizer.step()`?
21. Why is loss usually a scalar?
22. What happens if you call backward twice?
23. What does `retain_graph=True` do?
24. Why can in-place operations cause Autograd errors?
25. How would you convert a gradient-tracking tensor to NumPy?

For each, provide:

### Short Interview Answer

A concise answer suitable for speaking.

### Detailed Explanation

A slightly more detailed answer.

Use collapsible `<details>` answer sections.

---

# 39. Example Interview Answer Quality

For:

```text
Why do we call optimizer.zero_grad()?
```

A good short answer should be approximately:

> PyTorch accumulates gradients in each parameter's `.grad` field by default. `optimizer.zero_grad()` clears gradients from the previous training step so the next backward pass starts clean.

Avoid unnecessarily long textbook answers.

---

# 40. Knowledge Check Quiz

Create approximately 20 multiple-choice questions.

Focus on conceptual understanding.

Example:

```text
What does loss.backward() do?

A. Updates model parameters
B. Calculates gradients
C. Clears old gradients
D. Disables gradient tracking
```

Correct:

```text
B
```

Another:

```text
Which command updates model parameters?

A. loss.backward()
B. optimizer.zero_grad()
C. optimizer.step()
D. model.train()
```

Correct:

```text
C
```

Another:

```text
What happens if optimizer.zero_grad() is omitted?

A. Gradients are deleted
B. Gradients accumulate
C. Backpropagation stops
D. The model switches to evaluation mode
```

Correct:

```text
B
```

Do not show answers immediately. Put them in a final answer section.

---

# 41. Write From Memory

Create approximately 15 prompts.

Examples:

> Create scalar tensor `x = 3.0` with gradient tracking enabled.

> Compute `y = x²` and calculate its gradient.

> Access the gradient stored for `x`.

> Clear `x.grad` manually.

> Disable gradient tracking for a block of code.

> Detach tensor `y` from the computational graph.

> Enable gradient tracking on an existing float tensor.

> Freeze all parameters in `model`.

> Re-enable gradients for `model.classifier`.

> Perform the three standard optimizer gradient operations in correct order.

For each coding prompt, include validation.

---

# 42. Autograd Flow Challenge

Create a section where learners determine whether gradients can flow.

For each example, ask:

```text
Will x receive a gradient?

YES / NO
```

Examples:

### Case 1

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 2
z = y ** 2
```

Expected:

```text
YES
```

### Case 2

```python
x = torch.tensor(2.0, requires_grad=True)
y = x.detach()
z = y ** 2
```

Expected:

```text
NO
```

### Case 3

```python
x = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = x * 2

z = y ** 2
```

Expected:

```text
NO
```

Create at least 8 cases.

---

# 43. Gradient Calculation Challenge

Create 10 short manual gradient questions.

Examples:

```text
y = 5x
x = 3

dy/dx = ?
```

```text
y = x² + 2x
x = 4

dy/dx = ?
```

```text
z = xy
x = 2
y = 5

dz/dx = ?
dz/dy = ?
```

After manual prediction, validate using Autograd.

The goal is to help learners trust and understand what `.backward()` calculates.

---

# 44. Final Autograd Challenge

Create a larger final exercise.

Start with a tiny linear regression model manually:

```python
torch.manual_seed(42)

X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0]
])
```

This corresponds approximately to:

```text
y = 2x + 1
```

Create parameters manually:

```python
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

Ask the learner to implement:

1. Forward prediction:

```text
prediction = X * w + b
```

2. Mean squared error manually.

3. Call backward.

4. Inspect `w.grad`.

5. Inspect `b.grad`.

6. Update parameters using `torch.no_grad()`.

7. Clear gradients.

8. Repeat for multiple epochs.

9. Print final `w` and `b`.

10. Verify they approach:

```text
w ≈ 2
b ≈ 1
```

Use appropriate tolerance rather than requiring exact values.

This final exercise should demonstrate:

```text
forward
→ loss
→ backward
→ gradients
→ update
→ zero gradients
```

without using a PyTorch optimizer.

---

# 45. Final Cheat Sheet

End with a concise Autograd cheat sheet.

Include:

```python
# Enable gradient tracking
x = torch.tensor(2.0, requires_grad=True)

# Check
x.requires_grad

# Forward computation
y = x ** 2

# Backward pass
y.backward()

# Read gradient
x.grad

# Clear one tensor's gradient
x.grad.zero_()

# Disable gradients temporarily
with torch.no_grad():
    y = model(x)

# Detach tensor
y = x.detach()

# Enable gradients in place
x.requires_grad_(True)

# Freeze model
for param in model.parameters():
    param.requires_grad = False

# Re-enable parameters
for param in model.parameters():
    param.requires_grad = True

# Typical training gradient flow
optimizer.zero_grad()
prediction = model(X)
loss = loss_fn(prediction, y)
loss.backward()
optimizer.step()
```

Then include:

```text
backward()
→ computes gradients

zero_grad()
→ clears accumulated gradients

step()
→ updates parameters

no_grad()
→ temporarily disables gradient tracking

detach()
→ disconnects a tensor from the current graph
```

---

# 46. Important Mental Model

Include a concise section:

```text
Forward:
parameters
   ↓
model
   ↓
prediction
   ↓
loss

Backward:
loss
   ↓
backward()
   ↓
parameter.grad

Update:
parameter.grad
   ↓
optimizer.step()
   ↓
new parameters
```

This should be easy to memorize before interviews.

---

# 47. Completion Checklist

End with:

```text
## Before Moving to 04_neural_networks.ipynb
```

Add:

- [ ] I understand `requires_grad`.
- [ ] I understand computational graphs.
- [ ] I can explain forward vs backward pass.
- [ ] I can use `.backward()`.
- [ ] I know where gradients are stored.
- [ ] I understand the chain rule conceptually.
- [ ] I understand gradient accumulation.
- [ ] I know why gradients must be cleared.
- [ ] I understand leaf tensors.
- [ ] I understand `torch.no_grad()`.
- [ ] I understand `.detach()`.
- [ ] I know the difference between `detach()` and `no_grad()`.
- [ ] I can freeze model parameters.
- [ ] I understand `backward()` vs `optimizer.step()`.
- [ ] I can manually implement a basic gradient-descent loop.
- [ ] I can identify common Autograd bugs.

---

# 48. Difficulty Labels

Use:

```text
🟢 Easy
🟡 Medium
🔴 Hard
```

Most of the notebook should be Easy to Medium.

The following can be Medium:

- leaf tensors
- explicit gradients for non-scalar outputs
- `retain_graph`
- `retain_grad`
- in-place Autograd problems

Do not make these advanced topics dominate the notebook.

---

# 49. Quality Requirements

The notebook must:

- be a valid `.ipynb`
- execute from top to bottom
- work on CPU-only systems
- require only PyTorch
- not require internet access
- contain no empty placeholder cells
- include real runnable examples
- include automatic `assert` validation
- contain approximately 50–70 exercises/questions
- emphasize interview-relevant concepts
- teach gradient behavior through experimentation
- keep calculus simple
- avoid unnecessary internal Autograd implementation details

The notebook should take approximately **2–3 hours** to study properly.

The core principle is:

```text
Do not teach Autograd as something magical.

Make the learner repeatedly:

predict the gradient
      ↓
run backward()
      ↓
inspect .grad
      ↓
compare with the prediction
      ↓
understand why
```

Finally save the notebook as:

```text
03_autograd.ipynb
```