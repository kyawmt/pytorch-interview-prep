Create a Jupyter Notebook named:

```text
01_tensor_basics.ipynb
```

The notebook is part of a larger **PyTorch Interview Preparation** course for someone preparing for **AI Engineer / Machine Learning Engineer interviews**.

The notebook should teach PyTorch tensor fundamentals through:

- concise cheat-sheet style explanations
- runnable examples
- active-recall exercises
- automatic validation using `assert`
- interview-focused notes
- short quizzes
- debugging questions
- a final challenge

The notebook must run with normal Python + PyTorch in Jupyter.

Do NOT create a website. Create an actual `.ipynb` notebook.

# Main Goal

By the end of this notebook, the learner should be comfortable writing common PyTorch tensor syntax from memory and should understand the tensor concepts commonly asked in AI Engineer interviews.

Prioritize practical PyTorch knowledge over theory.

The notebook should follow this learning loop:

```text
Learn syntax
    ↓
Run example
    ↓
Predict result
    ↓
Write code yourself
    ↓
Run validation
    ↓
Understand mistakes
    ↓
Interview question
```

# Notebook Style

Use a clean structure with Markdown headings.

Use this approximate hierarchy:

```text
# PyTorch Tensor Basics

## 1. Setup
## 2. What Is a Tensor?
## 3. Creating Tensors
## 4. Tensor Shapes and Dimensions
## 5. Tensor Data Types
## 6. Tensor Devices
## 7. Basic Tensor Operations
## 8. Indexing and Slicing
## 9. Reshaping Tensors
## 10. Combining Tensors
## 11. NumPy and PyTorch
## 12. Common Mistakes
## 13. Interview Questions
## 14. Practice Challenge
## 15. Cheat Sheet
```

Keep explanations concise.

Do not turn the notebook into a textbook.

# 1. Setup

Start with:

```python
import torch
import numpy as np
```

Display the PyTorch version:

```python
print(torch.__version__)
```

Add a short Markdown explanation that `torch` is the main PyTorch package.

# 2. What Is a Tensor?

Explain simply that a tensor is a multidimensional array.

Show the relationship:

```text
0D tensor → scalar
1D tensor → vector
2D tensor → matrix
3D+ tensor → higher-dimensional tensor
```

Show examples:

```python
scalar = torch.tensor(5)

vector = torch.tensor([1, 2, 3])

matrix = torch.tensor([
    [1, 2],
    [3, 4]
])

tensor_3d = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
```

Print:

```python
scalar.shape
vector.shape
matrix.shape
tensor_3d.shape
```

Add a short interview note:

> In deep learning, tensors are usually organized by dimensions such as batch, features, channels, height, width, sequence length, and embedding dimension.

# 3. Creating Tensors

Create separate subsections for these important functions:

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
```

For each function:

1. Give one-sentence explanation.
2. Show one runnable example.
3. Show resulting tensor or shape.
4. Add an exercise immediately after it.

Example:

Markdown:

```text
### torch.zeros()

Creates a tensor filled with zeros.
```

Example:

```python
x = torch.zeros(2, 3)
print(x)
```

Exercise:

```python
# Exercise:
# Create a 3 x 4 tensor filled with zeros.

answer = None
```

Validation cell:

```python
assert isinstance(answer, torch.Tensor), "answer must be a PyTorch tensor"
assert answer.shape == (3, 4), "Expected shape (3, 4)"
assert torch.all(answer == 0), "All values should be zero"

print("✅ Correct!")
```

Use this exercise + validation structure throughout the notebook.

# 4. Exercises for Tensor Creation

Include exercises such as:

### Exercise 1

Create:

```text
tensor([1, 2, 3, 4])
```

using `torch.tensor()`.

### Exercise 2

Create a tensor with shape:

```text
(2, 5)
```

filled with zeros.

### Exercise 3

Create a:

```text
3 x 3
```

tensor filled with ones.

### Exercise 4

Create:

```text
tensor([0, 2, 4, 6, 8])
```

using `torch.arange()`.

### Exercise 5

Create 5 evenly spaced numbers from 0 to 1 using `torch.linspace()`.

### Exercise 6

Create a random tensor with shape:

```text
(4, 3)
```

using values between 0 and 1.

### Exercise 7

Create a random tensor sampled from a normal distribution with shape:

```text
(2, 3)
```

### Exercise 8

Create random integer values between 0 and 9 with shape:

```text
(3, 4)
```

### Exercise 9

Create a:

```text
4 x 4
```

identity matrix.

Use appropriate validation for each.

Do not validate random values exactly. Validate properties such as shape, dtype, and valid range.

# 5. Tensor Shapes and Dimensions

Teach:

```python
x.shape
x.size()
x.ndim
x.numel()
```

Use:

```python
x = torch.randn(32, 3, 224, 224)
```

Explain:

```text
32  → batch size
3   → channels
224 → height
224 → width
```

Show:

```python
print(x.shape)
print(x.ndim)
print(x.numel())
```

Explain that:

```python
x.shape
```

and:

```python
x.size()
```

generally provide the same shape information.

Add exercises.

Example:

```python
x = torch.randn(16, 10)
```

Ask the learner to assign:

```python
batch_size = ...
num_features = ...
```

Validate:

```python
assert batch_size == 16
assert num_features == 10
```

Add shape-prediction questions such as:

```python
x = torch.randn(64, 3, 32, 32)
```

Ask:

```text
How many dimensions does x have?

What is x.shape[0]?

What is x.shape[-1]?
```

# 6. Tensor Data Types

Teach important PyTorch dtypes:

```python
torch.float32
torch.float64
torch.float16
torch.bfloat16
torch.int64
torch.int32
torch.bool
```

Explain aliases where useful:

```python
torch.long
```

is commonly equivalent to:

```python
torch.int64
```

Teach:

```python
x.dtype
```

Show examples:

```python
x = torch.tensor([1, 2, 3])
print(x.dtype)

y = torch.tensor([1.0, 2.0, 3.0])
print(y.dtype)
```

Explain that integer literals and floating-point literals can produce different default dtypes.

Teach conversion:

```python
x.float()
x.long()
x.int()
x.bool()
x.to(torch.float32)
```

Include exercises.

Example:

```python
x = torch.tensor([1, 2, 3])
```

Ask learner to convert it to `float32`.

Validation:

```python
assert answer.dtype == torch.float32
```

Add an interview note:

> Classification targets used with `nn.CrossEntropyLoss` are commonly stored as `torch.long` / `torch.int64`.

# 7. Tensor Devices

Teach:

```python
x.device
```

and:

```python
torch.cuda.is_available()
```

Show:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)
```

Teach:

```python
x = x.to(device)
```

Keep this section basic because GPU usage will have its own notebook later.

Mention that Apple Silicon may use MPS:

```python
torch.backends.mps.is_available()
```

Do not require CUDA for notebook validation.

Exercises must work on CPU-only systems.

# 8. Basic Tensor Operations

Teach element-wise operations:

```python
x + y
x - y
x * y
x / y
x ** 2
```

Make clear that:

```python
x * y
```

is element-wise multiplication.

It is NOT necessarily matrix multiplication.

Teach common reductions:

```python
torch.sum()
torch.mean()
torch.max()
torch.min()
torch.argmax()
torch.argmin()
```

Example:

```python
x = torch.tensor([3, 7, 2, 9])

print(torch.max(x))
print(torch.argmax(x))
```

Explain the difference:

```text
torch.max(x)    → maximum value
torch.argmax(x) → index of maximum value
```

Add exercises.

# 9. Matrix Multiplication

Introduce:

```python
torch.matmul()
```

and:

```python
x @ y
```

Example:

```python
A = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

B = torch.tensor([
    [5.0, 6.0],
    [7.0, 8.0]
])

C = A @ B
print(C)
```

Explain matrix shape rule:

```text
(m, n) @ (n, p) → (m, p)
```

Add shape reasoning exercises.

Example:

```text
(32, 128) @ (128, 64)
```

Expected output:

```text
(32, 64)
```

Include at least 5 matrix-shape questions.

# 10. Indexing and Slicing

Teach:

```python
x[0]
x[1]
x[:, 0]
x[:, 1]
x[0, :]
x[1:3]
x[:, 1:3]
x[-1]
```

Use a simple matrix:

```python
x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
```

Show clearly what each expression returns.

Include exercises such as:

> Select the second row.

> Select the first column.

> Select values 50 and 60.

> Select the final row.

Validate using:

```python
torch.equal(...)
```

# 11. Boolean Indexing

Teach:

```python
x[x > 0]
```

Example:

```python
x = torch.tensor([-2, -1, 0, 1, 2])

positive = x[x > 0]
print(positive)
```

Exercise:

> Select all values greater than 5.

Validation should compare tensors.

# 12. Reshaping Tensors

This should be one of the most important sections.

Teach:

```python
reshape()
view()
flatten()
squeeze()
unsqueeze()
transpose()
permute()
```

Start with:

```python
x = torch.arange(12)
```

Show:

```python
x.reshape(3, 4)
```

Explain that the total number of elements must remain unchanged.

Teach `-1` dimension inference:

```python
x.reshape(3, -1)
```

Explain that PyTorch calculates the missing dimension automatically.

# 13. reshape vs view

Explain concisely:

```text
view()
- Requires compatible/contiguous memory layout.
- Often used for simple reshaping.

reshape()
- More flexible.
- May return a view or create a copy when necessary.
```

Do not go too deeply into storage internals.

For interview preparation, emphasize:

> `reshape()` is generally safer when you do not specifically need `view()` semantics.

# 14. flatten

Teach:

```python
x.flatten()
x.flatten(start_dim=1)
```

Use:

```python
x = torch.randn(32, 3, 28, 28)
```

Explain:

```python
x.flatten(1)
```

results in:

```text
(32, 2352)
```

because:

```text
3 × 28 × 28 = 2352
```

Include several shape prediction exercises.

# 15. squeeze and unsqueeze

Teach:

```python
x.unsqueeze(0)
x.unsqueeze(1)
x.squeeze()
x.squeeze(0)
```

Explain visually.

Example:

```text
(3, 4)
```

after:

```python
x.unsqueeze(0)
```

becomes:

```text
(1, 3, 4)
```

After:

```python
x.unsqueeze(1)
```

becomes:

```text
(3, 1, 4)
```

Create at least 5 shape exercises.

# 16. transpose and permute

Teach:

```python
x.transpose(0, 1)
```

and:

```python
x.permute(0, 2, 3, 1)
```

Explain:

```text
transpose
→ swaps two dimensions

permute
→ rearranges all dimensions
```

Use image data:

```python
x = torch.randn(32, 3, 224, 224)
```

Show converting:

```text
NCHW
```

to:

```text
NHWC
```

using:

```python
x.permute(0, 2, 3, 1)
```

Output shape:

```text
(32, 224, 224, 3)
```

Mark this as:

```text
🔥 Interview Essential
```

# 17. Combining Tensors

Teach:

```python
torch.cat()
torch.stack()
```

Explain very clearly.

Example:

```python
a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
```

Show:

```python
torch.cat([a, b])
```

result:

```text
tensor([1, 2, 3, 4])
```

Show:

```python
torch.stack([a, b])
```

result shape:

```text
(2, 2)
```

Explain:

```text
cat
→ joins along an existing dimension

stack
→ creates a new dimension
```

Mark this as:

```text
🔥 Interview Essential
```

Add multiple shape-prediction exercises.

Example:

```text
a.shape = (3, 4)
b.shape = (3, 4)

torch.cat([a, b], dim=0)
```

Expected:

```text
(6, 4)
```

and:

```python
torch.stack([a, b], dim=0)
```

Expected:

```text
(2, 3, 4)
```

# 18. Broadcasting

Introduce basic broadcasting.

Use:

```python
x = torch.randn(3, 4)
bias = torch.randn(4)

y = x + bias
```

Explain that `bias` is applied to every row.

Show shapes:

```text
x     → (3, 4)
bias  →    (4,)
result→ (3, 4)
```

Include shape reasoning exercises.

Do not go deeply into formal broadcasting rules yet.

# 19. Cloning and Copies

Introduce:

```python
x.clone()
```

Briefly explain that it creates a copy of the tensor data.

Do not introduce autograd details such as `.detach()` deeply because they belong in the autograd notebook.

You may mention:

> `.detach()` will be covered later when discussing autograd.

# 20. NumPy and PyTorch

Teach conversion:

```python
torch.from_numpy()
tensor.numpy()
```

Example:

```python
arr = np.array([1, 2, 3])

tensor = torch.from_numpy(arr)
print(tensor)
```

and:

```python
tensor.numpy()
```

Mention briefly that NumPy arrays and tensors created this way may share memory.

Add one simple exercise.

# 21. Common Shape Patterns

Create a Markdown section with common AI/ML tensor shapes.

Use a simple table.

Include:

```text
Tabular data:
(batch_size, features)

Image:
(batch_size, channels, height, width)

Sequence:
(batch_size, sequence_length)

Transformer hidden states:
(batch_size, sequence_length, hidden_size)

Classification output:
(batch_size, num_classes)
```

Create short questions:

> If a classifier processes a batch of 32 samples and predicts 10 classes, what output shape is expected?

Expected:

```text
(32, 10)
```

# 22. Common Mistakes

Create a dedicated section.

Include examples of these mistakes.

## Mistake 1

Confusing:

```python
x * y
```

with matrix multiplication.

Explain:

```python
x * y
```

is element-wise.

```python
x @ y
```

performs matrix multiplication.

## Mistake 2

Changing the number of elements during reshape.

Example:

```python
x = torch.arange(12)
x.reshape(5, 3)
```

Explain why this fails.

## Mistake 3

Confusing `cat()` with `stack()`.

## Mistake 4

Using the wrong dimension in `softmax`, `argmax`, or concatenation.

Only introduce the general idea here.

## Mistake 5

Forgetting that image tensors are commonly:

```text
NCHW
```

in PyTorch.

# 23. Debugging Exercises

Include at least 8 short debugging exercises.

Example:

```python
x = torch.arange(10)
y = x.reshape(3, 4)
```

Ask:

> What is wrong?

Provide a hidden answer after a Markdown `<details>` section or a separate solution cell.

Another example:

```python
A = torch.randn(3, 4)
B = torch.randn(5, 2)

C = A @ B
```

Ask:

> Why does matrix multiplication fail?

Expected concept:

```text
The inner dimensions do not match: 4 != 5.
```

# 24. Interview Questions

Create approximately 15 interview questions.

Questions should include:

1. What is a PyTorch tensor?
2. How is a tensor different from a Python list?
3. What is the difference between `shape`, `size()`, and `ndim`?
4. What is the difference between `torch.rand()` and `torch.randn()`?
5. What is the difference between `reshape()` and `view()`?
6. What does `-1` mean in `reshape()`?
7. What is the difference between `squeeze()` and `unsqueeze()`?
8. What is the difference between `transpose()` and `permute()`?
9. What is the difference between `torch.cat()` and `torch.stack()`?
10. What is broadcasting?
11. What does `torch.argmax()` return?
12. What is the difference between element-wise multiplication and matrix multiplication?
13. What is the common image tensor format in PyTorch?
14. What dtype is commonly used for class labels?
15. How do you move a tensor to another device?

For every question, provide:

### Short interview answer

A concise answer that could realistically be spoken in an interview.

### Explanation

A slightly more detailed explanation.

Use collapsible Markdown `<details>` sections where supported.

Example:

```html
<details>
<summary>Show answer</summary>

`torch.cat()` joins tensors along an existing dimension, while
`torch.stack()` creates a new dimension.

</details>
```

# 25. Knowledge Check Quiz

Create approximately 15 multiple-choice questions.

Example:

```text
What is the output shape?

x = torch.randn(32, 3, 28, 28)
y = x.flatten(1)

A. (32, 3, 28, 28)
B. (32, 2352)
C. (2352, 32)
D. (32, 784)
```

Do not immediately place the answer next to the question.

Put answers in a separate section after the quiz.

# 26. Write-from-Memory Exercises

Add a section called:

```text
## Write From Memory
```

Include about 10 prompts.

Examples:

> Create a tensor containing `[1, 2, 3]`.

> Create a random tensor of shape `(32, 10)` from a standard normal distribution.

> Convert tensor `x` to float32.

> Reshape tensor `x` to `(3, 4)`.

> Flatten all dimensions except the batch dimension.

> Add a dimension at index 1.

> Convert NCHW to NHWC.

> Concatenate tensors `a` and `b` along dimension 0.

> Stack tensors `a` and `b` along a new dimension.

> Find the index of the maximum value in tensor `x`.

Provide executable validation cells after each.

# 27. Final Challenge

Create a section:

```text
# Final Tensor Challenge
```

The learner should complete approximately 10 steps using one tensor pipeline.

Start with something like:

```python
x = torch.arange(24)
```

Ask the learner to:

1. Reshape it to `(2, 3, 4)`.
2. Add a dimension at the front.
3. Rearrange dimensions.
4. Flatten selected dimensions.
5. Extract a slice.
6. Calculate a mean.
7. Find a maximum value.
8. Find the index of the maximum.
9. Concatenate with another compatible tensor.
10. Verify the final shape.

Use asserts after the solution.

Make the challenge meaningful but not unnecessarily complicated.

# 28. Automatic Validation Rules

For every coding exercise:

- Give the learner a cell containing incomplete code.
- Follow it with a separate validation cell.
- Validation should provide useful error messages.
- Do not reveal the answer in the assertion unless necessary.
- Print:

```python
print("✅ Correct!")
```

when correct.

Example:

```python
# Exercise:
# Create a tensor with shape (2, 3) filled with ones.

answer = None
```

Then:

```python
assert isinstance(answer, torch.Tensor), "answer must be a tensor"
assert answer.shape == (2, 3), "Check the shape"
assert torch.all(answer == 1), "Check the values"

print("✅ Correct!")
```

# 29. Exercise Difficulty

Mark exercises using labels such as:

```text
🟢 Easy
🟡 Medium
🔴 Hard
```

Most exercises in this notebook should be Easy or Medium.

# 30. Interview Priority

Mark important sections with:

```text
🔥 Interview Essential
```

Use this especially for:

- tensor shapes
- dtype
- reshape
- flatten
- squeeze / unsqueeze
- transpose / permute
- matrix multiplication
- cat vs stack
- broadcasting
- common AI tensor shapes

# 31. Cheat Sheet at the End

End the notebook with a compact PyTorch Tensor Cheat Sheet.

It should contain syntax such as:

```python
# Create tensors
torch.tensor([1, 2, 3])
torch.zeros(2, 3)
torch.ones(2, 3)
torch.arange(0, 10, 2)
torch.linspace(0, 1, 5)
torch.rand(2, 3)
torch.randn(2, 3)
torch.randint(0, 10, (2, 3))
torch.eye(3)

# Properties
x.shape
x.size()
x.ndim
x.dtype
x.device
x.numel()

# Type conversion
x.float()
x.long()
x.to(torch.float32)

# Reshape
x.reshape(...)
x.view(...)
x.flatten()
x.flatten(1)
x.squeeze()
x.unsqueeze(0)
x.transpose(0, 1)
x.permute(...)

# Index
x[0]
x[:, 0]
x[1:3]
x[x > 0]

# Combine
torch.cat([a, b], dim=0)
torch.stack([a, b], dim=0)

# Operations
x + y
x * y
x @ y
torch.sum(x)
torch.mean(x)
torch.max(x)
torch.argmax(x)
```

Keep this section compact enough to use for quick revision.

# 32. Completion Checklist

At the end add:

```text
## Before Moving to 02_tensor_operations.ipynb

You should be able to do these without looking up the syntax:
```

Add a Markdown checklist:

```text
- Create common tensor types
- Read tensor shape and dimensions
- Change dtype
- Index and slice tensors
- Reshape tensors
- Use flatten
- Use squeeze and unsqueeze
- Use transpose and permute
- Explain cat vs stack
- Perform matrix multiplication
- Predict common tensor shapes
```

# 33. Quality Requirements

The generated notebook must:

- be a valid `.ipynb` file
- execute from top to bottom
- not contain broken cells
- not depend on internet access
- require only PyTorch and NumPy
- work on CPU-only systems
- not require a GPU
- contain meaningful real exercises
- contain real validation code
- contain no empty placeholder sections
- avoid unnecessary advanced topics
- focus on interview-relevant material

The notebook should be comprehensive enough to take roughly **1.5–3 hours** to study properly.

Do not simply explain PyTorch syntax. Make the learner repeatedly **write the syntax themselves**.

The most important principle is:

```text
Reading PyTorch is not enough.

The learner should repeatedly type PyTorch code,
run it,
make mistakes,
and fix those mistakes.
```

Finally, save the completed file as:

```text
01_tensor_basics.ipynb
```