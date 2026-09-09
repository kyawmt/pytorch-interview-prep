Create a Jupyter Notebook named:

```text
02_tensor_operations.ipynb
```

This notebook is the second part of a **PyTorch Interview Preparation** course for someone preparing for **AI Engineer / Machine Learning Engineer interviews**.

Assume the learner has already completed:

```text
01_tensor_basics.ipynb
```

and already knows:

- tensor creation
- tensor shapes
- dtype basics
- indexing and slicing
- reshape
- view
- flatten
- squeeze / unsqueeze
- transpose / permute
- basic `cat()` and `stack()`
- basic matrix multiplication

Do not repeat those topics extensively.

The purpose of this notebook is to build stronger practical skill with **tensor operations, broadcasting, dimensions, reductions, vectorization, masking, advanced indexing, and shape reasoning**.

The learner should repeatedly predict shapes and values, write PyTorch code, run it, and validate it.

---

# Main Learning Goals

By the end of this notebook, the learner should be able to confidently use:

```python
torch.sum()
torch.mean()
torch.max()
torch.min()
torch.argmax()
torch.argmin()

torch.any()
torch.all()

torch.abs()
torch.sqrt()
torch.exp()
torch.log()

torch.clamp()

torch.where()

torch.eq()
torch.ne()
torch.gt()
torch.ge()
torch.lt()
torch.le()

torch.unique()

torch.sort()
torch.argsort()
torch.topk()

torch.cat()
torch.stack()

torch.chunk()
torch.split()

torch.gather()

torch.matmul()
torch.mm()
torch.bmm()

torch.einsum()
```

The notebook should also teach:

- broadcasting
- the importance of the `dim` argument
- `keepdim=True`
- boolean masks
- advanced indexing
- vectorization
- batch matrix multiplication
- tensor aggregation
- selecting top values
- common dimension mistakes
- practical AI/ML tensor operations

---

# Notebook Structure

Use approximately this structure:

```text
# PyTorch Tensor Operations

## 1. Setup and Quick Review
## 2. Element-wise Operations
## 3. Comparison Operations
## 4. Reductions
## 5. Understanding dim
## 6. keepdim
## 7. Broadcasting
## 8. Boolean Masks
## 9. torch.where
## 10. Sorting and Top-K
## 11. Combining and Splitting Tensors
## 12. Matrix Operations
## 13. Batch Matrix Multiplication
## 14. Advanced Indexing and Gather
## 15. Vectorization
## 16. Basic einsum
## 17. Common AI/ML Tensor Patterns
## 18. Common Mistakes
## 19. Debugging Exercises
## 20. Interview Questions
## 21. Knowledge Check Quiz
## 22. Write From Memory
## 23. Final Challenge
## 24. Cheat Sheet
```

Keep explanations concise and practical.

The notebook should be focused on active practice rather than long theoretical explanations.

---

# 1. Setup

Start with:

```python
import torch

torch.manual_seed(42)
```

Add a short note explaining why using a fixed seed makes examples reproducible.

---

# 2. Element-wise Operations

Teach:

```python
x + y
x - y
x * y
x / y
x ** 2
```

Also show function equivalents where useful:

```python
torch.add()
torch.sub()
torch.mul()
torch.div()
```

Do not overemphasize the function forms because operators are more common.

Use:

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
```

Show results.

Then teach common mathematical functions:

```python
torch.abs()
torch.sqrt()
torch.exp()
torch.log()
```

Example:

```python
x = torch.tensor([1.0, 4.0, 9.0])

torch.sqrt(x)
```

Add short exercises after each concept.

---

# 3. Comparison Operations

Teach:

```python
x == y
x != y
x > y
x >= y
x < y
x <= y
```

Also mention:

```python
torch.eq()
torch.ne()
torch.gt()
torch.ge()
torch.lt()
torch.le()
```

Explain that comparison operations return boolean tensors.

Example:

```python
x = torch.tensor([1, 5, 3, 8])

mask = x > 3
print(mask)
```

Expected:

```text
tensor([False, True, False, True])
```

Add exercises asking learners to construct boolean masks.

---

# 4. torch.any() and torch.all()

Teach:

```python
torch.any()
torch.all()
```

Example:

```python
x = torch.tensor([True, True, False])

torch.any(x)
torch.all(x)
```

Explain simply:

```text
any → Is at least one value True?

all → Are all values True?
```

Show practical examples such as:

```python
x = torch.tensor([2, 4, 6])

torch.all(x > 0)
```

Add exercises.

---

# 5. Reductions

This should be an important section.

Teach:

```python
torch.sum()
torch.mean()
torch.max()
torch.min()
torch.argmax()
torch.argmin()
```

Use a matrix:

```python
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
```

Show:

```python
torch.sum(x)
torch.mean(x)
```

Then introduce reductions by dimension:

```python
torch.sum(x, dim=0)
torch.sum(x, dim=1)
```

Clearly show the resulting shapes and values.

---

# 6. Understanding `dim`

This should be marked:

```text
🔥 Interview Essential
```

Spend meaningful time teaching `dim`.

Use:

```python
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
```

Shape:

```text
(2, 3)
```

Explain:

```python
torch.sum(x, dim=0)
```

means collapse dimension 0.

Result:

```text
[5, 7, 9]
```

Shape:

```text
(3,)
```

And:

```python
torch.sum(x, dim=1)
```

means collapse dimension 1.

Result:

```text
[6, 15]
```

Shape:

```text
(2,)
```

Use diagrams such as:

```text
dim=0
↓
rows collapse
```

and:

```text
dim=1
→ columns collapse
```

But also make clear that the most robust way to think about it is:

> The specified dimension disappears after the reduction unless `keepdim=True`.

Create at least 10 shape-prediction exercises involving `dim`.

---

# 7. `keepdim=True`

Teach:

```python
torch.sum(x, dim=1, keepdim=True)
```

Compare:

```text
without keepdim:
(2, 3) → (2,)

with keepdim:
(2, 3) → (2, 1)
```

Explain why this is useful for:

- broadcasting
- normalization
- maintaining tensor rank

Include exercises.

---

# 8. Broadcasting

Make this one of the largest sections.

Mark:

```text
🔥 Interview Essential
```

Start with simple examples.

### Example 1

```text
(3, 4)
+
(4,)
→
(3, 4)
```

Use:

```python
x = torch.randn(3, 4)
bias = torch.randn(4)

y = x + bias
```

### Example 2

```text
(2, 3, 4)
+
(4,)
→
(2, 3, 4)
```

### Example 3

```text
(32, 10)
+
(10,)
→
(32, 10)
```

Explain this as adding a bias vector to a batch of predictions.

Teach the broadcasting rule simply:

Compare dimensions from the right.

Two dimensions are compatible if:

```text
they are equal
OR
one of them is 1
```

Add examples of valid and invalid broadcasting.

Valid:

```text
(3, 4)
(1, 4)

(5, 1)
(1, 7)

(32, 10)
(10,)
```

Invalid:

```text
(3, 4)
(3,)
```

Explain why.

Create at least 15 broadcasting prediction exercises.

Ask questions such as:

> Will these tensors broadcast?

> If yes, what is the output shape?

---

# 9. Normalization Using Broadcasting

Show a practical example.

```python
x = torch.randn(4, 3)

mean = x.mean(dim=0, keepdim=True)
std = x.std(dim=0, keepdim=True)

normalized = (x - mean) / std
```

Explain shape flow:

```text
x     → (4, 3)
mean  → (1, 3)
std   → (1, 3)
```

Broadcasting makes the operation possible.

This is an excellent practical example for interviews.

---

# 10. Boolean Masks

Teach:

```python
x[x > 0]
```

Use:

```python
x = torch.tensor([-3, -1, 0, 2, 5])
```

Show:

```python
positive = x[x > 0]
```

Also teach multi-condition masks:

```python
mask = (x > 0) & (x < 5)
```

Explain that Python:

```python
and
or
```

should generally not be used for element-wise tensor conditions.

Instead use:

```python
&
|
~
```

with parentheses.

Example:

```python
(x > 0) & (x < 10)
```

Mark this as a common debugging issue.

Add exercises.

---

# 11. `torch.where()`

Teach:

```python
torch.where(condition, value_if_true, value_if_false)
```

Example:

```python
x = torch.tensor([-2, -1, 0, 1, 2])

result = torch.where(
    x > 0,
    x,
    torch.tensor(0)
)
```

Show a simpler valid form if supported:

```python
torch.where(x > 0, x, 0)
```

Explain this can implement conditional element-wise transformations.

Example:

```python
torch.where(x >= 0, 1, 0)
```

Add exercises such as:

> Replace all negative values with zero.

---

# 12. `torch.clamp()`

Teach:

```python
torch.clamp(x, min=0)
torch.clamp(x, min=0, max=1)
```

Explain that this is often simpler than `torch.where` for clipping values.

Example:

```python
x = torch.tensor([-2.0, 0.5, 3.0])

torch.clamp(x, 0, 1)
```

Include one exercise comparing `clamp()` and `where()`.

---

# 13. Unique Values

Teach:

```python
torch.unique()
```

Example:

```python
x = torch.tensor([1, 2, 2, 3, 3, 3])

torch.unique(x)
```

Also optionally demonstrate:

```python
torch.unique(x, return_counts=True)
```

Keep this section brief.

---

# 14. Sorting

Teach:

```python
torch.sort()
torch.argsort()
```

Example:

```python
x = torch.tensor([30, 10, 20])
```

Show:

```python
values, indices = torch.sort(x)
```

Explain the difference:

```text
sort()
→ returns sorted values and their original indices

argsort()
→ returns only the indices that would sort the tensor
```

Add exercises.

---

# 15. Top-K

Mark:

```text
🔥 Interview Useful
```

Teach:

```python
torch.topk()
```

Example:

```python
scores = torch.tensor([0.2, 0.9, 0.4, 0.7])

values, indices = torch.topk(scores, k=2)
```

Explain this is common for:

- top predictions
- ranking
- retrieval
- beam-search-like operations

Add exercises such as:

> Find the indices of the top 3 class scores.

---

# 16. Combining Tensors

Review and deepen:

```python
torch.cat()
torch.stack()
```

Do not repeat basics extensively.

Focus on dimension reasoning.

Examples:

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)
```

Ask the learner to predict:

```python
torch.cat([a, b], dim=0)
torch.cat([a, b], dim=1)
torch.stack([a, b], dim=0)
torch.stack([a, b], dim=1)
```

Require prediction before showing answers.

Add at least 8 shape exercises.

---

# 17. Splitting Tensors

Teach:

```python
torch.chunk()
torch.split()
```

Explain:

```text
chunk
→ divide tensor into a requested number of chunks

split
→ divide using specific split sizes or chunk length
```

Examples:

```python
x = torch.arange(12)

torch.chunk(x, 3)
```

and:

```python
torch.split(x, 4)
```

Also show:

```python
torch.split(x, [2, 4, 6])
```

when appropriate.

Add exercises.

---

# 18. Matrix Multiplication Review

Review:

```python
torch.matmul()
x @ y
torch.mm()
```

Explain:

```text
torch.mm()
→ specifically 2D matrix multiplication

torch.matmul()
→ supports more general tensor dimensions and broadcasting
```

Do not overcomplicate this section.

Use shape reasoning:

```text
(5, 10) @ (10, 3)
→ (5, 3)
```

Add several exercises.

---

# 19. Batch Matrix Multiplication

Mark:

```text
🔥 Interview Essential for Deep Learning
```

Teach:

```python
torch.bmm()
```

Use:

```python
A = torch.randn(32, 10, 64)
B = torch.randn(32, 64, 20)

C = torch.bmm(A, B)
```

Explain:

```text
(32, 10, 64)
@
(32, 64, 20)

→

(32, 10, 20)
```

Explain that each batch performs its own matrix multiplication.

Use a diagram:

```text
batch 0:
(10, 64) @ (64, 20)

batch 1:
(10, 64) @ (64, 20)

...

batch 31:
(10, 64) @ (64, 20)
```

Explain that this pattern appears frequently in:

- attention
- sequence models
- batched neural-network operations

Add at least 8 batch matrix shape questions.

---

# 20. `matmul()` With Batched Tensors

Show that:

```python
torch.matmul(A, B)
```

can also support batched matrix multiplication.

Briefly contrast:

```text
bmm
→ explicitly expects 3D batched tensors

matmul
→ supports broader broadcasting behavior
```

Keep the explanation practical.

---

# 21. Dot Product

Briefly teach:

```python
torch.dot()
```

for 1D tensors.

Example:

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

torch.dot(a, b)
```

Explain the result.

Do not spend too much time on this.

---

# 22. Advanced Indexing

Use a matrix:

```python
x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
```

Teach selecting specific rows:

```python
x[[0, 2]]
```

Teach selecting specific coordinates:

```python
rows = torch.tensor([0, 2])
cols = torch.tensor([1, 2])

x[rows, cols]
```

Expected:

```text
tensor([20, 90])
```

Explain that paired index tensors select corresponding positions.

Add exercises.

---

# 23. `torch.gather()`

Introduce `gather()` at an interview-relevant level.

Example:

```python
scores = torch.tensor([
    [0.1, 0.7, 0.2],
    [0.8, 0.1, 0.1]
])

targets = torch.tensor([
    [1],
    [0]
])

selected = torch.gather(scores, 1, targets)
```

Explain that this selects specific positions along a given dimension.

Relate it to selecting:

```text
the score corresponding to each sample's target class
```

Keep the explanation intuitive.

Add 2–3 exercises.

---

# 24. Vectorization

Mark:

```text
🔥 Interview Essential
```

Explain that PyTorch code should generally use tensor operations instead of Python loops when possible.

Bad example:

```python
x = torch.arange(1000, dtype=torch.float32)

result = []

for value in x:
    result.append(value * 2)
```

Better:

```python
result = x * 2
```

Explain advantages:

- shorter code
- faster execution
- GPU-friendly
- lets PyTorch use optimized kernels

Add exercises asking learners to convert loops into vectorized operations.

Examples:

### Exercise

Convert:

```python
result = []

for value in x:
    if value > 0:
        result.append(value)
```

into a boolean-mask expression.

### Exercise

Replace:

```python
for i in range(len(x)):
    x[i] = x[i] + 5
```

with one tensor expression.

Include at least 8 vectorization exercises.

---

# 25. Basic `torch.einsum()`

Introduce only the basics.

Do NOT make this section mathematically overwhelming.

Explain:

```python
torch.einsum()
```

provides a compact way to describe tensor multiplications and reductions using dimension labels.

Start with dot product:

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

torch.einsum("i,i->", a, b)
```

Then matrix multiplication:

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)

C = torch.einsum("ij,jk->ik", A, B)
```

Explain labels:

```text
i → rows
j → shared dimension
k → output columns
```

Then batch matrix multiplication:

```python
torch.einsum("bij,bjk->bik", A, B)
```

where shapes are compatible.

Clearly state:

> For normal code, `@`, `matmul()`, and `bmm()` are often easier to read. `einsum()` becomes especially useful for complex tensor operations such as attention.

Include only 3–5 exercises.

---

# 26. Softmax Dimension Preview

Because `dim` is important, briefly introduce:

```python
torch.softmax()
```

using:

```python
logits = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 0.0]
])
```

Show:

```python
probs = torch.softmax(logits, dim=1)
```

Explain:

```text
Each row represents one sample.
Each column represents one class.

dim=1 means normalize across the classes.
```

Validate:

```python
probs.sum(dim=1)
```

should be approximately:

```text
tensor([1., 1.])
```

Mark:

```text
🔥 Interview Essential
```

Do not go deeply into loss functions yet.

---

# 27. Argmax for Classification

Use:

```python
predicted_classes = torch.argmax(logits, dim=1)
```

Explain output shape:

```text
logits:
(batch_size, num_classes)

argmax(dim=1):
(batch_size,)
```

Example:

```text
(32, 10)
→
(32,)
```

Create several shape/value exercises.

---

# 28. Practical Pattern: Classification Accuracy

Show:

```python
predictions = torch.argmax(logits, dim=1)

correct = predictions == targets

accuracy = correct.float().mean()
```

Explain each step.

This is highly useful for interview preparation.

Add an exercise where the learner calculates accuracy from prediction logits and labels.

---

# 29. Practical Pattern: Masking Sequence Values

Show an introductory NLP-style example.

```python
tokens = torch.tensor([
    [5, 8, 3, 0, 0],
    [9, 4, 2, 7, 0]
])

mask = tokens != 0
```

Explain:

```text
1 / True → real token
0 / False → padding
```

Do not introduce full attention masking yet.

Add one exercise.

---

# 30. Practical Pattern: Selecting Batch Labels

Use:

```python
scores = torch.randn(4, 5)
labels = torch.tensor([1, 4, 0, 3])
```

Show how to select the score corresponding to each label using advanced indexing:

```python
batch_indices = torch.arange(scores.shape[0])

selected = scores[batch_indices, labels]
```

Explain the shapes.

This is a useful interview-style tensor operation.

---

# 31. Shape Reasoning Section

Create a dedicated section containing approximately 20 rapid shape questions.

Example:

```python
x = torch.randn(32, 10)

x.mean(dim=0)
```

Expected:

```text
(10,)
```

Another:

```python
x.mean(dim=0, keepdim=True)
```

Expected:

```text
(1, 10)
```

Another:

```python
a = torch.randn(8, 3, 5)
b = torch.randn(8, 5, 7)

torch.bmm(a, b)
```

Expected:

```text
(8, 3, 7)
```

Another:

```python
x = torch.randn(16, 10)

torch.argmax(x, dim=1)
```

Expected:

```text
(16,)
```

Use automatic assert-based validation where practical.

---

# 32. Common Mistakes

Create a section covering at least these mistakes.

## Mistake 1 — Wrong reduction dimension

Example:

```python
logits = torch.randn(32, 10)

pred = torch.argmax(logits, dim=0)
```

Explain why this is usually wrong for classification.

Correct:

```python
torch.argmax(logits, dim=1)
```

---

## Mistake 2 — Forgetting `keepdim`

Show a normalization example where shape becomes inconvenient.

---

## Mistake 3 — Using `and` instead of `&`

Wrong:

```python
(x > 0) and (x < 10)
```

Correct:

```python
(x > 0) & (x < 10)
```

---

## Mistake 4 — Broadcasting incompatible shapes

Show:

```text
(3, 4)
+
(3,)
```

and explain the mismatch.

---

## Mistake 5 — Confusing `sort()` and `argsort()`

---

## Mistake 6 — Confusing `max()` and `argmax()`

---

## Mistake 7 — `cat()` along the wrong dimension

---

## Mistake 8 — Wrong batch matrix dimensions

Example:

```text
(32, 10, 64)
@
(32, 20, 64)
```

Explain why the inner matrix dimensions are incorrect.

---

# 33. Debugging Exercises

Create at least 12 short debugging exercises.

Example:

```python
logits = torch.randn(32, 10)

pred = logits.argmax(dim=0)
```

Question:

> We want one predicted class per sample. What is wrong?

Another:

```python
x = torch.randn(5, 4)
bias = torch.randn(5)

y = x + bias
```

Question:

> Why does this fail?

Another:

```python
mask = x > 0 and x < 1
```

Question:

> What should be used instead?

Another:

```python
A = torch.randn(8, 3, 5)
B = torch.randn(8, 7, 5)

torch.bmm(A, B)
```

Question:

> What dimension mismatch exists?

For each debugging question, include a collapsible answer or separate answer cell.

---

# 34. Coding Exercise Format

Use the same exercise structure as the first notebook.

Example:

```python
# Exercise:
# Find the mean of each row of x.

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

answer = None
```

Validation:

```python
expected = torch.tensor([2.0, 5.0])

assert isinstance(answer, torch.Tensor), "answer should be a tensor"
assert torch.allclose(answer, expected), "Check the dimension used for mean()"

print("✅ Correct!")
```

Do not reveal the solution directly inside the exercise.

---

# 35. Interview Questions

Create approximately 20 interview questions.

Include:

1. What is broadcasting?
2. What are PyTorch broadcasting rules?
3. What does `dim` mean in a reduction operation?
4. What does `keepdim=True` do?
5. What is the difference between `max()` and `argmax()`?
6. What is the difference between `sort()` and `argsort()`?
7. What does `topk()` return?
8. What is boolean masking?
9. Why use `&` instead of `and` for tensor conditions?
10. What does `torch.where()` do?
11. What is the difference between `cat()` and `stack()`?
12. What is the difference between `chunk()` and `split()`?
13. What is the difference between `torch.mm()` and `torch.matmul()`?
14. What does `torch.bmm()` do?
15. Where is batched matrix multiplication used in deep learning?
16. Why is vectorized tensor code preferred to Python loops?
17. What does `torch.gather()` do?
18. What does `argmax(logits, dim=1)` usually represent?
19. Why is `keepdim=True` useful during normalization?
20. When might `einsum()` be useful?

For each question provide:

### Short Interview Answer

A concise spoken answer.

### Detailed Explanation

A slightly deeper explanation.

Use collapsible answer sections.

---

# 36. Knowledge Check Quiz

Create approximately 20 multiple-choice questions.

Include many questions about:

- output shapes
- broadcasting
- `dim`
- `argmax`
- `keepdim`
- matrix multiplication
- boolean masks
- sorting
- top-k

Example:

```text
x.shape = (32, 10)

What is the shape of:

x.mean(dim=1, keepdim=True)

A. (10,)
B. (32,)
C. (32, 1)
D. (1, 10)
```

Correct:

```text
C
```

Put answers at the end rather than immediately below each question.

---

# 37. Write From Memory

Create approximately 15 coding prompts.

Examples:

> Sum tensor `x` across dimension 1.

> Find the class index with the highest score for every sample in `logits`.

> Keep only positive values from `x`.

> Replace negative values in `x` with zero.

> Find the top 3 values in `scores`.

> Concatenate `a` and `b` along dimension 1.

> Split `x` into 4 chunks.

> Perform batch matrix multiplication between `A` and `B`.

> Normalize each feature using feature-wise mean.

> Create a mask identifying all non-zero tokens.

Each should include automatic validation.

---

# 38. Mini Interview Challenge

Create a timed-style section:

```text
## 10-Minute Tensor Operations Challenge
```

Include 10 rapid questions.

Mix:

- shape prediction
- syntax
- debugging
- broadcasting
- classification operations
- matrix multiplication

Do not provide answers until a final solution section.

---

# 39. Final Challenge

Create a larger end-to-end tensor challenge.

Start with:

```python
torch.manual_seed(42)

logits = torch.randn(8, 5)
targets = torch.tensor([1, 0, 4, 2, 3, 1, 2, 0])
```

Ask the learner to perform steps such as:

1. Compute softmax probabilities across classes.
2. Verify each row sums to approximately 1.
3. Find the predicted class for every sample.
4. Create a boolean tensor showing correct predictions.
5. Calculate accuracy.
6. Find the maximum probability for every sample.
7. Find the two most likely classes for every sample.
8. Select the probability assigned to each target label.
9. Compute the average target-class probability.
10. Identify samples where the target-class probability is below a threshold.

Use validation cells for every stage.

This challenge should combine many of the skills taught in the notebook.

---

# 40. Final Cheat Sheet

End with a compact cheat sheet containing:

```python
# Element-wise
x + y
x - y
x * y
x / y
x ** 2

torch.abs(x)
torch.sqrt(x)
torch.exp(x)
torch.log(x)

# Comparisons
x > 0
x == y
(x > 0) & (x < 1)

# Reductions
x.sum()
x.mean()
x.max()
x.min()
x.argmax()
x.argmin()

x.sum(dim=0)
x.mean(dim=1)
x.mean(dim=1, keepdim=True)

# Conditions
torch.any(x)
torch.all(x)
torch.where(condition, a, b)
torch.clamp(x, min=0, max=1)

# Sorting
torch.sort(x)
torch.argsort(x)
torch.topk(x, k=3)

# Combine / split
torch.cat([a, b], dim=0)
torch.stack([a, b], dim=0)
torch.chunk(x, 4)
torch.split(x, 4)

# Matrix operations
A @ B
torch.matmul(A, B)
torch.mm(A, B)
torch.bmm(A, B)
torch.dot(a, b)

# Selection
x[mask]
torch.gather(x, dim, index)

# Classification
torch.softmax(logits, dim=1)
torch.argmax(logits, dim=1)

# einsum
torch.einsum("ij,jk->ik", A, B)
torch.einsum("bij,bjk->bik", A, B)
```

Keep this compact.

---

# 41. Completion Checklist

End with:

```text
## Before Moving to 03_autograd.ipynb
```

Add a checklist:

- [ ] I understand what `dim` means.
- [ ] I can predict reduction output shapes.
- [ ] I understand `keepdim=True`.
- [ ] I understand PyTorch broadcasting.
- [ ] I can use boolean masks.
- [ ] I can use `torch.where()`.
- [ ] I understand `max()` vs `argmax()`.
- [ ] I can use `topk()`.
- [ ] I can predict `cat()` and `stack()` output shapes.
- [ ] I understand `matmul()` and `bmm()`.
- [ ] I can perform batched matrix multiplication.
- [ ] I can calculate classification predictions and accuracy.
- [ ] I can replace simple Python loops with vectorized tensor operations.
- [ ] I understand the basic purpose of `einsum()`.

---

# 42. Quality Requirements

The notebook must:

- be a valid `.ipynb`
- execute successfully from top to bottom
- require only PyTorch
- not depend on internet access
- work on CPU
- use reproducible examples where needed
- contain real runnable code
- contain automatic validation
- contain no placeholder sections
- contain approximately 60–80 exercises/questions in total
- emphasize shape reasoning
- emphasize `dim`
- emphasize broadcasting
- emphasize practical AI/ML use cases
- avoid unnecessary obscure APIs

The notebook should take approximately **2–3 hours** to study thoroughly.

The key principle is:

```text
Do not merely show tensor operations.

Make the learner repeatedly predict the shape,
write the operation,
run the code,
and verify the result.
```

Finally save the completed notebook as:

```text
02_tensor_operations.ipynb
```