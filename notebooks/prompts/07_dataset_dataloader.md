Create a Jupyter Notebook named:

```text
07_dataset_dataloader.ipynb
```

This notebook is the seventh part of a **PyTorch Interview Preparation** course for someone preparing for **AI Engineer / Machine Learning Engineer interviews**.

Assume the learner has already completed:

```text
01_tensor_basics.ipynb
02_tensor_operations.ipynb
03_autograd.ipynb
04_neural_networks.ipynb
05_losses_optimizers.ipynb
06_training_loop.ipynb
```

and already understands:

- tensors and shapes
- model construction
- Autograd
- losses and optimizers
- training loops
- validation loops
- epochs and batches
- device handling

Do not spend significant time reteaching those topics.

The purpose of this notebook is to teach the **PyTorch data pipeline**, especially:

```python
Dataset
TensorDataset
DataLoader
```

and the practical concepts needed to load, batch, shuffle, preprocess, and feed data into training loops.

This notebook should prepare the learner to both **write custom datasets from memory** and **explain DataLoader behavior during AI Engineer interviews**.

---

# Main Learning Goals

By the end of the notebook, the learner should confidently understand and use:

```python
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
```

The learner should understand:

- what a Dataset is
- what a DataLoader is
- Dataset vs DataLoader
- `__len__()`
- `__getitem__()`
- custom `Dataset`
- `TensorDataset`
- batching
- shuffling
- batch size
- final incomplete batch
- `drop_last`
- `num_workers`
- `pin_memory`
- custom preprocessing
- transforms at the Dataset level
- `collate_fn`
- variable-length data
- custom batching
- train / validation DataLoaders
- why training data is usually shuffled
- why validation data usually is not
- moving batches to a device
- common DataLoader mistakes

Mark these as:

```text
🔥 Interview Essential
```

- Dataset vs DataLoader
- custom Dataset
- `__len__`
- `__getitem__`
- batching
- `shuffle=True`
- `batch_size`
- `num_workers`
- `drop_last`
- `collate_fn`
- train vs validation loaders

---

# Notebook Structure

Use approximately:

```text
# PyTorch Dataset and DataLoader

## 1. Setup
## 2. What Is a Dataset?
## 3. What Is a DataLoader?
## 4. Dataset vs DataLoader
## 5. TensorDataset
## 6. DataLoader Basics
## 7. Batch Size
## 8. Shuffling
## 9. Iterating Through DataLoader
## 10. Final Incomplete Batch
## 11. drop_last
## 12. Building a Custom Dataset
## 13. __len__()
## 14. __getitem__()
## 15. Adding Preprocessing
## 16. Train and Validation Loaders
## 17. num_workers
## 18. pin_memory
## 19. collate_fn
## 20. Variable-Length Data
## 21. Custom Batch Structures
## 22. DataLoader + Device Handling
## 23. Dataset Patterns for Tabular Data
## 24. Dataset Patterns for Images
## 25. Dataset Patterns for Text
## 26. Common Mistakes
## 27. Debugging Exercises
## 28. Interview Questions
## 29. Knowledge Check Quiz
## 30. Write From Memory
## 31. Final Data Pipeline Challenge
## 32. Cheat Sheet
```

---

# 1. Setup

Start with:

```python
import torch

from torch.utils.data import (
    Dataset,
    TensorDataset,
    DataLoader
)

torch.manual_seed(42)
```

The notebook must:

- work on CPU
- require no internet
- use synthetic data
- execute quickly

Do not require:

- torchvision
- pandas
- scikit-learn
- external datasets

unless used only in optional sections.

---

# 2. What Is a Dataset?

Mark:

```text
🔥 Interview Essential
```

Explain simply:

> A PyTorch Dataset represents a collection of samples and defines how individual samples are retrieved.

Explain the basic contract:

```text
Dataset
must know:

1. How many samples exist?
2. How do I retrieve sample i?
```

This corresponds to:

```python
__len__()
__getitem__()
```

Show conceptual example:

```text
dataset[0]
dataset[1]
dataset[2]
...
```

---

# 3. What Is a DataLoader?

Mark:

```text
🔥 Interview Essential
```

Explain:

> A DataLoader takes a Dataset and provides convenient iteration over batches of samples.

Visualize:

```text
Dataset
   ↓
DataLoader
   ↓
Batch 1
Batch 2
Batch 3
...
```

Explain that DataLoader handles things such as:

- batching
- shuffling
- multiprocessing
- collation

---

# 4. Dataset vs DataLoader

Make this a dedicated interview section.

Use a concise comparison:

| Dataset | DataLoader |
|---|---|
| represents samples | produces batches |
| defines how to retrieve one sample | iterates over many samples |
| implements `__len__` and `__getitem__` | handles batching/shuffling |
| `dataset[i]` | `for batch in loader` |

Short interview answer:

> Dataset defines how individual samples are accessed; DataLoader wraps the Dataset and handles batching, shuffling, parallel loading, and iteration.

Mark:

```text
🔥 Very Common Interview Question
```

---

# 5. TensorDataset

Teach:

```python
TensorDataset
```

Example:

```python
X = torch.randn(100, 10)
y = torch.randint(0, 3, (100,))

dataset = TensorDataset(X, y)
```

Show:

```python
print(len(dataset))
```

Expected:

```text
100
```

Show:

```python
sample_X, sample_y = dataset[0]

print(sample_X.shape)
print(sample_y)
```

Explain:

> `TensorDataset` is useful when all your data is already stored in aligned tensors.

---

# 6. TensorDataset Requirement

Explain that the first dimension must have matching length.

Example:

```python
X.shape
# (100, 10)

y.shape
# (100,)
```

Both have:

```text
100 samples
```

Show a debugging example where lengths differ.

Ask why it fails or is invalid.

---

# 7. Exercise — Create TensorDataset

Give:

```python
features = torch.randn(200, 5)
targets = torch.randint(0, 2, (200,))

dataset = None
```

Ask learner to create the dataset.

Validation:

```python
assert isinstance(dataset, TensorDataset)
assert len(dataset) == 200

x0, y0 = dataset[0]

assert x0.shape == (5,)
assert y0.ndim == 0

print("✅ Correct!")
```

---

# 8. DataLoader Basics

Teach:

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

Explain arguments:

```text
dataset
→ source of samples

batch_size
→ number of samples per batch

shuffle
→ whether sample order is randomized
```

---

# 9. Iterating Through DataLoader

Show:

```python
for X_batch, y_batch in loader:
    print(X_batch.shape)
    print(y_batch.shape)
    break
```

Expected:

```text
X_batch:
(32, 10)

y_batch:
(32,)
```

Explain that DataLoader automatically stacks compatible individual samples into batches.

---

# 10. Batch Size

Mark:

```text
🔥 Interview Essential
```

Explain:

> Batch size is the number of samples processed together in one training step.

Show:

```text
Dataset:
1000 samples

batch_size:
100

approximately:
10 batches
```

Explain trade-offs at a practical level:

### Smaller batches

- less memory
- noisier gradients
- more optimizer steps per epoch

### Larger batches

- more memory
- fewer optimizer steps
- can use hardware more efficiently

Do not claim larger or smaller is universally better.

---

# 11. Number of Batches

Teach:

```python
len(loader)
```

Example:

```python
dataset = TensorDataset(
    torch.randn(100, 5),
    torch.randn(100)
)

loader = DataLoader(
    dataset,
    batch_size=32
)

print(len(loader))
```

Expected:

```text
4
```

Explain:

```text
32
32
32
4
```

Create several batch-count exercises.

Examples:

```text
Dataset = 1000
Batch = 128
```

Ask how many batches.

Expected:

```text
8
```

because DataLoader includes the final partial batch by default.

---

# 12. Final Incomplete Batch

Demonstrate:

```python
dataset = TensorDataset(
    torch.arange(10)
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False
)
```

Print each batch.

Expected sizes:

```text
4
4
2
```

Explain:

> By default, DataLoader keeps the final smaller batch.

---

# 13. `drop_last`

Teach:

```python
DataLoader(
    dataset,
    batch_size=4,
    drop_last=True
)
```

Then expected:

```text
4
4
```

final 2 samples are not included in that epoch.

Explain common use cases:

- when equal batch shapes are required
- BatchNorm sensitivity in very small final batches
- some distributed/training setups

But emphasize:

> `drop_last=True` discards samples, so do not enable it without a reason.

Mark:

```text
🟡 Interview Useful
```

---

# 14. Shuffling

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

Explain:

> Shuffling changes the sample order each epoch so the model does not repeatedly see data in the same sequence.

Typical pattern:

```text
Training:
shuffle=True

Validation:
shuffle=False
```

Explain that validation does not normally require shuffling because parameters are not being updated.

---

# 15. Demonstrating Shuffle

Use:

```python
dataset = TensorDataset(
    torch.arange(10)
)
```

Create:

```python
loader_no_shuffle = DataLoader(
    dataset,
    batch_size=5,
    shuffle=False
)
```

and:

```python
loader_shuffle = DataLoader(
    dataset,
    batch_size=5,
    shuffle=True
)
```

Print order.

Explain random order can differ between runs/epochs.

---

# 16. Important Shuffle Nuance

Explain:

> `shuffle=True` changes order, but it does not change the samples themselves.

It is not data augmentation.

Create a short quiz question distinguishing:

```text
shuffling
vs
augmentation
```

---

# 17. Building a Custom Dataset

Make this one of the largest sections.

Mark:

```text
🔥 Interview Essential
```

Teach standard structure:

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

Explain each method.

---

# 18. `__init__`

Explain:

> `__init__` stores information the Dataset needs to retrieve samples.

This may include:

- tensors
- filenames
- labels
- preprocessing objects
- tokenizers
- metadata

For this notebook, use tensors and simple Python lists.

---

# 19. `__len__()`

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
def __len__(self):
    return len(self.X)
```

Explain:

> It returns the number of samples in the Dataset.

Show:

```python
len(dataset)
```

calls this method.

---

# 20. `__getitem__()`

Mark:

```text
🔥 Interview Essential
```

Teach:

```python
def __getitem__(self, idx):
    return self.X[idx], self.y[idx]
```

Explain:

> DataLoader repeatedly calls `dataset[idx]` to retrieve samples.

Show:

```python
dataset[5]
```

and explain this invokes:

```python
__getitem__(5)
```

---

# 21. Build a Custom Dataset Exercise

Starter:

```python
class ClassificationDataset(Dataset):

    def __init__(self, X, y):
        # TODO
        pass

    def __len__(self):
        # TODO
        pass

    def __getitem__(self, idx):
        # TODO
        pass
```

Validation:

```python
X = torch.randn(50, 8)
y = torch.randint(0, 4, (50,))

dataset = ClassificationDataset(X, y)

assert len(dataset) == 50

sample_x, sample_y = dataset[10]

assert sample_x.shape == (8,)
assert sample_y.ndim == 0

print("✅ Correct!")
```

---

# 22. Custom Dataset With Preprocessing

Show:

```python
class NormalizedDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.mean = X.mean(dim=0)
        self.std = X.std(dim=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]

        x = (
            x - self.mean
        ) / (self.std + 1e-8)

        return x, self.y[idx]
```

Explain preprocessing can happen inside `__getitem__`.

But discuss an important practical point:

> Expensive global preprocessing should usually not be recomputed for every sample inside `__getitem__`.

Compute reusable statistics once when appropriate.

---

# 23. Transform Pattern

Without requiring torchvision, demonstrate a transform function:

```python
def add_noise(x):
    return x + 0.01 * torch.randn_like(x)
```

Dataset:

```python
class CustomDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        transform=None
    ):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y
```

Explain this pattern generalizes to image and text preprocessing.

---

# 24. Avoid Doing Too Much in `__getitem__`

Explain practical considerations.

`__getitem__` may run many times, potentially across multiple worker processes.

Avoid unnecessarily:

- loading huge resources repeatedly
- recomputing dataset-wide statistics
- initializing models/tokenizers repeatedly
- expensive setup work per sample

Do one-time setup in `__init__` where appropriate.

---

# 25. Train / Validation Split

Create synthetic data:

```python
X = torch.randn(1000, 10)
y = torch.randint(0, 3, (1000,))
```

Split using tensor indices:

```python
split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]

X_val = X[split:]
y_val = y[split:]
```

Mention that real-world random splitting is often preferable, but keep this deterministic example simple.

Optionally show:

```python
torch.utils.data.random_split
```

later.

---

# 26. `random_split`

Teach:

```python
from torch.utils.data import random_split
```

Example:

```python
train_dataset, val_dataset = random_split(
    dataset,
    [800, 200],
    generator=torch.Generator().manual_seed(42)
)
```

Explain seeded generator provides reproducible split.

Add one exercise.

---

# 27. Train and Validation DataLoaders

Teach standard pattern:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)
```

Mark:

```text
🔥 Interview Essential
```

Ask:

> Why shuffle training data but usually not validation data?

Provide concise interview answer later.

---

# 28. DataLoader + Training Loop

Connect to previous notebook:

```python
for X_batch, y_batch in train_loader:

    optimizer.zero_grad()

    logits = model(X_batch)

    loss = loss_fn(logits, y_batch)

    loss.backward()

    optimizer.step()
```

Explain DataLoader's role is simply to provide each batch.

---

# 29. DataLoader Returns Batch Structures

Explain DataLoader does not always have to return:

```python
X, y
```

A Dataset can return:

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "label": ...
}
```

Then DataLoader may produce a batched dictionary.

Show a simple custom Dataset returning:

```python
{
    "features": self.X[idx],
    "label": self.y[idx]
}
```

Then:

```python
for batch in loader:
    X = batch["features"]
    y = batch["label"]
```

This is important for modern NLP / LLM codebases.

---

# 30. `num_workers`

Mark:

```text
🔥 Interview Useful
```

Teach:

```python
DataLoader(
    dataset,
    batch_size=32,
    num_workers=4
)
```

Explain:

> `num_workers` controls how many worker processes load/preprocess data in parallel.

Basic interpretation:

```text
num_workers=0
→ loading happens in main process

num_workers>0
→ worker processes load samples in parallel
```

Explain potential benefits:

- keep GPU fed with data
- overlap data preparation with training
- improve throughput when preprocessing/loading is expensive

---

# 31. `num_workers` Is Not Always Faster

Important nuance:

> More workers are not automatically better.

Too many can cause:

- process overhead
- extra memory usage
- CPU contention
- storage contention
- platform-specific issues

Interview answer:

> I usually benchmark `num_workers` because the best value depends on CPU cores, storage, preprocessing cost, and batch size.

---

# 32. Notebook / OS Caveat

Mention that multiprocessing behavior can differ across:

- Linux
- Windows
- macOS
- Jupyter

For exercises, use:

```python
num_workers=0
```

so the notebook runs reliably everywhere.

Use larger values only as conceptual examples.

---

# 33. `pin_memory`

Mark:

```text
🟡 Interview Useful
```

Teach:

```python
DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True
)
```

Explain:

> Pinned CPU memory can make CPU-to-CUDA transfers faster.

Typical CUDA training pattern:

```text
DataLoader:
pin_memory=True

then:
batch.to("cuda", non_blocking=True)
```

But clearly state:

> `pin_memory=True` is mainly useful when transferring CPU tensors to a CUDA GPU. It is not universally beneficial for CPU-only training.

Do not require CUDA.

---

# 34. `persistent_workers`

Briefly introduce:

```python
DataLoader(
    dataset,
    num_workers=4,
    persistent_workers=True
)
```

Explain:

> Keeps worker processes alive between epochs instead of recreating them.

Useful when worker startup is expensive.

Mark as optional/intermediate.

---

# 35. `prefetch_factor`

Mention briefly:

```python
prefetch_factor
```

Explain conceptually:

> Controls how many batches each worker prepares ahead of time.

Do not spend more than a short section on this.

Do not require code exercises for it.

---

# 36. `collate_fn`

Make this an important intermediate section.

Mark:

```text
🔥 Interview Useful, especially for NLP
```

Explain:

> `collate_fn` defines how individual samples are combined into a batch.

By default DataLoader can stack same-shaped tensors automatically.

Example individual samples:

```text
x1 shape: (10,)
x2 shape: (10,)
x3 shape: (10,)
```

default batching:

```text
batch shape:
(3, 10)
```

But variable-length samples cannot simply be stacked.

---

# 37. Simple Custom `collate_fn`

Create a Dataset returning variable-length tensors:

```python
samples = [
    torch.tensor([1, 2, 3]),
    torch.tensor([4, 5]),
    torch.tensor([6, 7, 8, 9])
]
```

Custom Dataset:

```python
class SequenceDataset(Dataset):

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]
```

Explain default batching fails because lengths differ.

---

# 38. Padding Variable-Length Sequences

Create:

```python
def collate_sequences(batch):

    max_len = max(
        len(sequence)
        for sequence in batch
    )

    padded = torch.zeros(
        len(batch),
        max_len,
        dtype=torch.long
    )

    for i, sequence in enumerate(batch):
        padded[i, :len(sequence)] = sequence

    return padded
```

Use:

```python
loader = DataLoader(
    dataset,
    batch_size=3,
    collate_fn=collate_sequences
)
```

Expected:

```text
[
 [1, 2, 3, 0],
 [4, 5, 0, 0],
 [6, 7, 8, 9]
]
```

Explain:

> `collate_fn` is where many NLP pipelines perform dynamic padding.

---

# 39. Attention Mask Preview

For padded sequences, create:

```python
attention_mask = padded != 0
```

Explain shape:

```text
tokens:
(batch, sequence_length)

attention mask:
(batch, sequence_length)
```

Do not teach transformer attention deeply yet.

Mention this pattern appears frequently in LLM training/inference code.

---

# 40. `pad_sequence`

Optionally teach:

```python
from torch.nn.utils.rnn import pad_sequence
```

Example:

```python
padded = pad_sequence(
    batch,
    batch_first=True,
    padding_value=0
)
```

Explain this is simpler than manually padding variable-length tensors.

Mark it useful for sequence/NLP workloads.

---

# 41. Custom Collate Returning Dictionary

Show:

```python
def collate_fn(batch):

    sequences = [
        item["tokens"]
        for item in batch
    ]

    labels = torch.tensor([
        item["label"]
        for item in batch
    ])

    tokens = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    )

    attention_mask = tokens != 0

    return {
        "input_ids": tokens,
        "attention_mask": attention_mask,
        "labels": labels
    }
```

This is useful for AI Engineer / LLM interview familiarity.

---

# 42. Dataset Pattern — Tabular Data

Show:

```python
class TabularDataset(Dataset):

    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

Explain common dtype handling:

```text
features
→ float32

multiclass labels
→ long / int64
```

---

# 43. Dataset Pattern — Images

Do not require actual image files.

Explain conceptual Dataset:

```python
class ImageDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels,
        transform=None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # load image from image_paths[idx]
        image = ...

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
```

Do not leave this as executable broken code. Put it in a Markdown/code example rather than an executed cell.

Explain typical flow:

```text
filename
↓
load image
↓
transform
↓
tensor
↓
label
```

---

# 44. Dataset Pattern — Text

Show conceptual example without requiring tokenizer libraries:

```python
class TextDataset(Dataset):

    def __init__(self, token_sequences, labels):
        self.tokens = token_sequences
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "tokens": self.tokens[idx],
            "label": self.labels[idx]
        }
```

Then batching can use custom `collate_fn`.

Explain typical real-world flow:

```text
raw text
↓
tokenization
↓
token IDs
↓
padding
↓
attention mask
↓
model
```

---

# 45. Map-Style vs Iterable Datasets

Introduce briefly.

Explain the normal custom Dataset taught so far is a **map-style Dataset**:

```python
dataset[index]
```

PyTorch also supports:

```python
IterableDataset
```

for streaming/sequential data.

Use cases:

- very large datasets
- streams
- data generated on the fly
- files that are naturally read sequentially

Do not deeply teach implementation.

Mark:

```text
🟡 Interview Awareness
```

---

# 46. Samplers

Briefly introduce the concept of:

```python
Sampler
```

Explain:

> A sampler determines which indices the DataLoader retrieves and in what order.

Mention:

- `shuffle=True` is enough for many cases
- custom samplers are useful for specialized sampling strategies
- `WeightedRandomSampler` may be used for imbalanced datasets

Do not make sampler implementation a major topic.

---

# 47. `WeightedRandomSampler`

Optionally show:

```python
from torch.utils.data import WeightedRandomSampler
```

Explain conceptually:

> Samples with higher weights are more likely to be selected.

Useful for class imbalance.

But add:

> It changes sampling frequency; it is different from class-weighted loss.

Keep this concise.

---

# 48. Shuffle vs Sampler

Explain:

> Do not normally specify both `shuffle=True` and a custom sampler that controls order.

Mention DataLoader may reject incompatible configurations.

This is a good debugging/interview detail.

---

# 49. DataLoader + Device Handling

Connect to Notebook 06.

Teach:

```python
for X, y in train_loader:

    X = X.to(device)
    y = y.to(device)

    ...
```

Explain:

> DataLoader usually yields CPU tensors. Moving the batch to GPU/MPS normally happens inside the training loop.

Do not put the entire Dataset on GPU as a default strategy.

---

# 50. Why Not Move Everything to GPU in Dataset?

Explain practical reasons:

- dataset may be larger than GPU memory
- DataLoader workers generally operate on CPU-side data
- batching lets you move only the current batch
- preprocessing may happen on CPU

Typical flow:

```text
disk / CPU dataset
       ↓
DataLoader
       ↓
CPU batch
       ↓
.to(device)
       ↓
GPU model
```

Mark:

```text
🔥 Interview Useful
```

---

# 51. Non-Blocking Transfer

Mention:

```python
X = X.to(
    device,
    non_blocking=True
)
```

Explain this can work with pinned memory for potentially asynchronous CUDA transfers.

Keep it optional and do not require CUDA.

---

# 52. Performance Pipeline Mental Model

Show:

```text
Storage
   ↓
Dataset.__getitem__
   ↓
DataLoader workers
   ↓
collate_fn
   ↓
CPU batch
   ↓
GPU transfer
   ↓
Model
```

Explain a slow training job may actually be **data-loader-bound**, not compute-bound.

This is useful for AI Engineer interviews.

---

# 53. Data Loader Bottleneck Example

Explain conceptually:

```text
GPU utilization:
30%

GPU waits between batches
```

Possible cause:

```text
data preprocessing/loading too slow
```

Potential adjustments:

- more workers
- faster storage
- cheaper preprocessing
- caching
- pinned memory
- prefetching

Do not imply these always solve the issue.

---

# 54. Common Mistakes

Include at least these.

## Mistake 1 — Dataset lengths do not match

```text
X: 100 samples
y: 90 labels
```

---

## Mistake 2 — `__len__()` returns feature dimension

Wrong:

```python
return self.X.shape[1]
```

instead of number of samples.

---

## Mistake 3 — `__getitem__()` returns entire dataset

Wrong:

```python
return self.X, self.y
```

instead of one indexed sample.

---

## Mistake 4 — Training loader uses `shuffle=False` accidentally

Not always fatal, but generally undesirable for standard supervised training.

---

## Mistake 5 — Validation loader uses random augmentation

Explain that stochastic training augmentation is usually disabled or changed during validation.

---

## Mistake 6 — Assuming every batch has exact `batch_size`

Final batch may be smaller.

---

## Mistake 7 — Wrong use of `drop_last=True`

Accidentally discarding validation/test samples.

Usually avoid `drop_last=True` for evaluation.

---

## Mistake 8 — Returning inconsistent shapes from Dataset

Default DataLoader collation may fail.

---

## Mistake 9 — Variable-length sequences without custom collation/padding

---

## Mistake 10 — Excessive work inside `__getitem__`

---

## Mistake 11 — Too many workers

May reduce performance or cause memory issues.

---

## Mistake 12 — Assuming `pin_memory=True` automatically moves tensors to GPU

It does not.

---

## Mistake 13 — Forgetting to move batch tensors to model device

---

## Mistake 14 — Putting `model.to(device)` inside every batch iteration

Usually move the model once before the loop.

---

## Mistake 15 — Returning wrong dtype

Example:

multiclass target returned as float instead of long.

---

## Mistake 16 — `shuffle=True` with incompatible sampler setup

---

## Mistake 17 — Padding variable-length sequences globally to an unnecessarily huge maximum length

Explain dynamic per-batch padding can reduce wasted computation.

---

# 55. Debugging Exercises

Create at least 20 debugging exercises.

Example:

```python
class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

Ask:

> What is wrong?

Expected:

```text
__len__ should return number of samples, typically X.shape[0].
```

---

Another:

```python
class MyDataset(Dataset):

    def __getitem__(self, idx):
        return self.X, self.y
```

Ask what is wrong.

---

Another:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False
)
```

Ask:

> What setting would usually be changed for standard training?

Expected:

```text
shuffle=True
```

---

Another:

```python
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    drop_last=True
)
```

Ask:

> Why could this be a bad idea for validation?

Expected:

> Some validation samples would never be evaluated.

---

Another:

```python
for X, y in train_loader:
    logits = model(X)
```

when model is CUDA.

Ask what is missing.

---

Another:

```python
sequences = [
    torch.tensor([1, 2]),
    torch.tensor([3, 4, 5])
]

loader = DataLoader(
    sequences,
    batch_size=2
)
```

Ask why default collation fails.

---

# 56. Exercise Format

Use coding exercises like:

```python
# Exercise:
# Build a DataLoader with:
# batch_size=16
# shuffle=True

loader = None
```

Validation:

```python
assert isinstance(loader, DataLoader)

assert loader.batch_size == 16

print("✅ Correct!")
```

Where internal attributes are not reliable, validate by observing actual batches instead.

---

# 57. Build Custom Dataset From Memory

Create:

```text
## Write a Custom Dataset From Memory
```

Prompt:

> Implement a Dataset that accepts `X` and `y` and returns one `(x, y)` pair by index.

Give only:

```python
class MyDataset(Dataset):
    pass
```

Validation should test:

- length
- several random indices
- returned values
- returned shape

This should be repeated more than once later in the notebook.

---

# 58. Batch Shape Exercises

Create at least 15.

Example:

```text
Dataset feature shape:
(1000, 20)

batch_size:
64

What is the normal feature batch shape?
```

Expected:

```text
(64, 20)
```

Another:

```text
Images:
(5000, 3, 224, 224)

batch_size:
32
```

Expected:

```text
(32, 3, 224, 224)
```

Another:

```text
Token IDs:
individual sample shape = (128,)
batch_size = 16
```

Expected:

```text
(16, 128)
```

---

# 59. Interview Questions

Create approximately 30–35 interview questions.

Include:

1. What is a PyTorch Dataset?
2. What is a DataLoader?
3. Dataset vs DataLoader?
4. What methods must a standard custom Dataset implement?
5. What does `__len__()` do?
6. What does `__getitem__()` do?
7. What is TensorDataset?
8. When would you use a custom Dataset instead of TensorDataset?
9. What is batch size?
10. Why use mini-batches?
11. What does `shuffle=True` do?
12. Why shuffle training data?
13. Do you need to shuffle validation data?
14. What happens to the final incomplete batch?
15. What does `drop_last=True` do?
16. Why might `drop_last=True` be used?
17. Why is `drop_last=True` usually undesirable for validation?
18. What does `num_workers` control?
19. Does a larger `num_workers` always improve speed?
20. What does `pin_memory=True` do?
21. When is pinned memory useful?
22. Does pinning memory move tensors to GPU?
23. What does `collate_fn` do?
24. Why is custom collation useful for NLP?
25. How do you batch variable-length sequences?
26. What is dynamic padding?
27. What does `pad_sequence` do?
28. What is the difference between a map-style Dataset and an IterableDataset?
29. What is a sampler?
30. Why shouldn't you move the entire Dataset to GPU by default?
31. Where should `.to(device)` usually happen?
32. What can cause DataLoader to become the training bottleneck?
33. How would you speed up a slow input pipeline?
34. What happens if Dataset samples have inconsistent shapes?
35. What dtype should multiclass labels commonly use?

For each provide:

### Short Interview Answer

and:

### Detailed Explanation

Use collapsible `<details>` sections.

---

# 60. Knowledge Check Quiz

Create approximately 25–30 multiple-choice questions.

Example:

```text
Which method tells PyTorch how many samples are in a custom Dataset?

A. __init__
B. __getitem__
C. __len__
D. forward
```

Correct:

```text
C
```

Another:

```text
What does DataLoader normally do?

A. Defines model architecture
B. Computes gradients
C. Batches and iterates over Dataset samples
D. Updates optimizer parameters
```

Correct:

```text
C
```

Another:

```text
Dataset size = 100
batch_size = 32
drop_last = False

How many batches?

A. 3
B. 4
C. 32
D. 100
```

Correct:

```text
B
```

Put answers in a separate section.

---

# 61. Write From Memory

Create approximately 20 prompts.

Examples:

> Import Dataset and DataLoader.

> Create a TensorDataset from `X` and `y`.

> Create a training DataLoader with batch size 32 and shuffle enabled.

> Create a validation DataLoader with batch size 64 and no shuffle.

> Write a custom Dataset with `__init__`, `__len__`, and `__getitem__`.

> Return one feature-label pair from `__getitem__`.

> Split a Dataset using `random_split`.

> Create a DataLoader with `drop_last=True`.

> Create a DataLoader with four workers.

> Create a DataLoader using pinned memory.

> Iterate through a DataLoader and move batches to `device`.

> Write a custom `collate_fn` that pads sequences.

> Generate an attention mask for padded tokens.

Every practical prompt should have validation where reasonable.

---

# 62. Data Pipeline Reading Exercise

Show:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

Ask learner to explain every argument verbally.

Expected explanation:

```text
train_dataset
→ source of samples

64
→ 64 samples per batch

shuffle=True
→ randomize training order

num_workers=4
→ four worker processes load data

pin_memory=True
→ use pinned CPU memory, potentially speeding CUDA transfers
```

This should resemble an interview question.

---

# 63. Data Pipeline Design Scenarios

Create approximately 10 scenarios.

Example:

### Scenario 1

Small tabular dataset already stored as tensors.

Ask:

> Custom Dataset or TensorDataset?

Likely:

```text
TensorDataset is sufficient.
```

---

### Scenario 2

One million image filenames stored on disk.

Ask:

> Should all images be loaded into RAM in `__init__`?

Expected:

```text
Usually no. Store paths and load each sample on demand.
```

---

### Scenario 3

Sentences have different lengths.

Ask:

> What DataLoader feature can help batch them?

Expected:

```text
custom collate_fn with dynamic padding
```

---

### Scenario 4

GPU is frequently idle while waiting for data.

Ask for likely pipeline improvements.

Possible answers:

- increase `num_workers`
- optimize preprocessing
- faster storage
- pinned memory
- prefetching

Do not require one exact answer.

---

# 64. Final Data Pipeline Challenge

Create a complete synthetic text-classification-style pipeline without external tokenizers.

Generate variable-length token sequences.

Example:

```python
torch.manual_seed(42)

num_samples = 200
vocab_size = 1000
num_classes = 3
```

For each sample, generate sequence length between:

```text
5 and 25
```

Each sequence should be a 1D `torch.long` tensor.

Create random class labels.

---

# 65. Final Challenge — Custom Dataset

Ask learner to implement:

```python
class TokenDataset(Dataset):
```

It should:

- store sequences
- store labels
- return dictionary:

```python
{
    "tokens": ...,
    "label": ...
}
```

Validation should test random samples.

---

# 66. Final Challenge — Custom `collate_fn`

Ask learner to create:

```python
def collate_batch(batch):
```

Requirements:

1. Extract token sequences.
2. Pad them dynamically.
3. Create attention mask.
4. Stack labels.
5. Return:

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...
}
```

Expected shapes:

```text
input_ids:
(batch, max_sequence_length_in_batch)

attention_mask:
same shape

labels:
(batch,)
```

Use:

```python
torch.nn.utils.rnn.pad_sequence
```

if desired.

---

# 67. Final Challenge — DataLoaders

Split dataset:

```text
80% training
20% validation
```

Create:

```python
train_loader
val_loader
```

Training:

```text
batch_size=16
shuffle=True
```

Validation:

```text
batch_size=32
shuffle=False
```

Use custom collate function for both.

---

# 68. Final Challenge — Model Connection

Build a very small model:

```text
Embedding
↓
masked mean pooling
↓
Linear classifier
```

Example architecture:

```python
class TextClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_classes
    ):
        ...
```

Use:

```text
embedding_dim = 32
```

Input:

```text
input_ids:
(batch, sequence)
```

Embedding output:

```text
(batch, sequence, 32)
```

Use attention mask to calculate mean only across real tokens.

Do not require advanced NLP.

---

# 69. Final Challenge — One Training Epoch

Connect the DataLoader to the previous training-loop knowledge.

Require:

```python
for batch in train_loader:

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    ...
```

Train with:

```python
CrossEntropyLoss
AdamW
```

Run only a few epochs.

The purpose is to prove the full data pipeline works.

---

# 70. Final Challenge Validation

Validate:

```python
batch = next(iter(train_loader))

assert batch["input_ids"].ndim == 2
assert batch["attention_mask"].shape == batch["input_ids"].shape
assert batch["labels"].ndim == 1
assert batch["input_ids"].dtype == torch.long
```

Also confirm:

```text
padding mask correctly marks real vs padded positions
```

and model output shape:

```text
(batch, num_classes)
```

---

# 71. Optional Hard Exercise — Balanced Sampling

Add:

```text
🔴 Hard / Optional
```

Create an imbalanced synthetic dataset and ask learner to explore:

```python
WeightedRandomSampler
```

Explain:

> This is optional and should not distract from core Dataset/DataLoader mastery.

---

# 72. Final Cheat Sheet

End with:

```python
from torch.utils.data import (
    Dataset,
    TensorDataset,
    DataLoader,
    random_split
)

# TensorDataset
dataset = TensorDataset(X, y)

# DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# Custom Dataset
class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Train / validation
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# Keep final batch
drop_last=False

# Parallel loading
num_workers=4

# CUDA transfer optimization
pin_memory=True

# Iterate
for X, y in train_loader:
    X = X.to(device)
    y = y.to(device)
```

Add variable-length example:

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):

    sequences = [
        item["tokens"]
        for item in batch
    ]

    labels = torch.tensor([
        item["label"]
        for item in batch
    ])

    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    )

    attention_mask = padded != 0

    return {
        "input_ids": padded,
        "attention_mask": attention_mask,
        "labels": labels
    }
```

---

# 73. Critical Mental Model

Include:

```text
Dataset
→ defines one sample

DataLoader
→ builds batches from samples

Training loop
→ consumes batches
```

And:

```text
Storage / tensors
       ↓
Dataset
       ↓
__getitem__
       ↓
collate_fn
       ↓
DataLoader batch
       ↓
.to(device)
       ↓
model
```

And:

```text
Training:
shuffle=True

Validation:
shuffle=False
```

---

# 74. Interview Quick Reference

Add:

```text
Dataset
→ how to retrieve one sample

__len__
→ number of samples

__getitem__
→ retrieve sample by index

DataLoader
→ batching + iteration

batch_size
→ samples per batch

shuffle
→ randomize sample order

drop_last
→ discard final partial batch

num_workers
→ parallel CPU data loading

pin_memory
→ potentially faster CPU→CUDA transfer

collate_fn
→ combine samples into a batch

TensorDataset
→ simple Dataset for aligned tensors
```

---

# 75. Completion Checklist

End with:

```text
## Before Moving to 08_gpu_and_devices.ipynb
```

Add:

- [ ] I can explain Dataset vs DataLoader.
- [ ] I can create a TensorDataset.
- [ ] I can create a DataLoader.
- [ ] I understand batch size.
- [ ] I understand shuffling.
- [ ] I know what happens to the final partial batch.
- [ ] I understand `drop_last`.
- [ ] I can write a custom Dataset from memory.
- [ ] I can implement `__len__()`.
- [ ] I can implement `__getitem__()`.
- [ ] I can apply preprocessing in a Dataset.
- [ ] I can create train and validation DataLoaders.
- [ ] I understand `num_workers`.
- [ ] I understand `pin_memory`.
- [ ] I understand `collate_fn`.
- [ ] I can batch variable-length sequences.
- [ ] I understand dynamic padding.
- [ ] I can create attention masks for padded sequences.
- [ ] I know where `.to(device)` belongs.
- [ ] I can identify common DataLoader bottlenecks.
- [ ] I can connect a DataLoader to a training loop.

---

# 76. Difficulty Labels

Use:

```text
🟢 Easy
🟡 Medium
🔴 Hard
```

Most content should be Easy to Medium.

Easy:

- TensorDataset
- DataLoader
- batch size
- shuffle
- custom Dataset basics

Medium:

- `num_workers`
- `pin_memory`
- custom transforms
- `collate_fn`
- variable-length batching
- dynamic padding

Hard / optional:

- custom samplers
- IterableDataset
- prefetch tuning
- weighted sampling

Do not go deeply into:

- distributed samplers
- WebDataset
- DataPipes
- massive distributed input pipelines
- multi-node data loading

Those can be covered later.

---

# 77. Quality Requirements

The notebook must:

- be a valid `.ipynb`
- execute successfully from top to bottom
- work on CPU-only systems
- require no internet
- use synthetic data only
- contain no broken placeholder code in executable cells
- include real Dataset implementations
- include real DataLoader usage
- include automatic `assert` validation
- contain approximately 70–90 exercises/questions
- heavily emphasize writing custom Dataset code from memory
- heavily emphasize batching and shape reasoning
- include realistic NLP-style variable-length batching
- include interview-focused explanations
- connect naturally to the previous training-loop notebook

The notebook should take approximately **2–3 hours** to study thoroughly.

The core learning principle is:

```text
Do not only read DataLoader syntax.

Repeatedly:

build a Dataset
      ↓
retrieve samples
      ↓
create a DataLoader
      ↓
inspect batches
      ↓
predict batch shapes
      ↓
customize collation
      ↓
feed batches into a model
      ↓
debug the pipeline
```

By the end of the notebook, the learner should be comfortable seeing code such as:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

and immediately understand what each part does and why it might be used.

Finally save the notebook as:

```text
07_dataset_dataloader.ipynb
```