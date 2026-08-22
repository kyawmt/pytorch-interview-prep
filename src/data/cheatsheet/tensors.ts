import type { CheatSheetEntry } from "@/lib/types";

export const tensorEntries: CheatSheetEntry[] = [
  {
    id: "cs-imports",
    topic: "tensors",
    section: "Imports",
    title: "The four imports you always write",
    description:
      "Almost every PyTorch training script starts with these. Interviewers notice when you hesitate here.",
    syntax: "import torch",
    example: `import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader`,
    interviewNote:
      "nn holds stateful layers (nn.ReLU, nn.Linear); F holds the stateless function versions (F.relu). Layers with learnable parameters belong in nn.",
    importance: "high",
    tags: ["import", "torch.nn", "functional", "setup"],
    relatedExercises: ["ex-imports-1"],
  },
  {
    id: "cs-tensor",
    topic: "tensors",
    section: "Tensor Creation",
    title: "torch.tensor()",
    description:
      "Creates a tensor by copying data from a Python list, NumPy array or scalar. The dtype is inferred unless you pass one.",
    syntax: "torch.tensor(data, dtype=None, device=None, requires_grad=False)",
    example: `x = torch.tensor([[1, 2], [3, 4]])
print(x, x.dtype, x.shape)`,
    result: "tensor([[1, 2], [3, 4]]) torch.int64 torch.Size([2, 2])",
    interviewNote:
      "Integers in, torch.int64 out. Pass dtype=torch.float32 when the tensor feeds a model, otherwise nn.Linear raises a dtype error.",
    importance: "high",
    tags: ["create", "tensor", "dtype"],
    relatedExercises: ["ex-tensor-1", "ex-tensor-2"],
  },
  {
    id: "cs-zeros",
    topic: "tensors",
    section: "Tensor Creation",
    title: "torch.zeros() / torch.ones()",
    description:
      "Allocate a tensor of a given shape filled with 0 or 1. Shape can be passed as separate ints or as a tuple.",
    syntax: "torch.zeros(*size)  ·  torch.ones(*size)",
    example: `a = torch.zeros(3, 4)
b = torch.ones((2, 3), dtype=torch.float32)
print(a.shape, b.shape)`,
    result: "torch.Size([3, 4]) torch.Size([2, 3])",
    interviewNote: "Default dtype is torch.float32, unlike torch.tensor() which infers from the data.",
    importance: "high",
    tags: ["zeros", "ones", "create"],
    relatedExercises: ["ex-tensor-3", "ex-tensor-4"],
  },
  {
    id: "cs-empty",
    topic: "tensors",
    section: "Tensor Creation",
    title: "torch.empty()",
    description:
      "Allocates memory without initialising it. Slightly faster than zeros when you are about to overwrite every element.",
    syntax: "torch.empty(*size)",
    example: `buf = torch.empty(2, 3)   # contents are arbitrary
buf.fill_(0.0)`,
    interviewNote:
      "Low interview priority. Never print an empty tensor and assume zeros — the values are whatever was in memory.",
    importance: "low",
    tags: ["empty", "uninitialised"],
  },
  {
    id: "cs-full",
    topic: "tensors",
    section: "Tensor Creation",
    title: "torch.full()",
    description: "Fills a shape with one constant value.",
    syntax: "torch.full(size, fill_value)",
    example: `mask = torch.full((2, 3), -1e9)
print(mask[0])`,
    result: "tensor([-1.0000e+09, -1.0000e+09, -1.0000e+09])",
    interviewNote:
      "A large negative fill is the standard trick for attention masks: add it to logits before softmax so those positions get ~0 weight.",
    importance: "medium",
    tags: ["full", "mask", "attention"],
  },
  {
    id: "cs-arange",
    topic: "tensors",
    section: "Tensor Creation",
    title: "torch.arange() / torch.linspace()",
    description:
      "arange steps by a fixed increment and excludes the endpoint. linspace picks N evenly spaced points and includes both ends.",
    syntax: "torch.arange(start, end, step)  ·  torch.linspace(start, end, steps)",
    example: `print(torch.arange(0, 5))
print(torch.linspace(0, 1, 5))`,
    result: `tensor([0, 1, 2, 3, 4])
tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])`,
    interviewNote:
      "arange with integer args gives int64; linspace always gives floats. Positional-encoding code uses both constantly.",
    importance: "medium",
    tags: ["arange", "linspace", "range", "positions"],
    relatedExercises: ["ex-tensor-5"],
  },
  {
    id: "cs-rand",
    topic: "tensors",
    section: "Tensor Creation",
    title: "torch.rand() / randn() / randint()",
    description:
      "rand draws uniform [0, 1). randn draws from the standard normal. randint draws integers in [low, high).",
    syntax: "torch.rand(*size)  ·  torch.randn(*size)  ·  torch.randint(low, high, size)",
    example: `x = torch.randn(32, 128)          # fake batch of embeddings
y = torch.randint(0, 10, (32,))   # fake class labels
print(x.shape, y.shape, y.dtype)`,
    result: "torch.Size([32, 128]) torch.Size([32]) torch.int64",
    interviewNote:
      "randn is the standard way to fabricate a batch when you sanity-check a model's shapes. Note randint takes the size as a tuple.",
    importance: "high",
    tags: ["random", "randn", "rand", "randint", "dummy data"],
    relatedExercises: ["ex-tensor-6", "ex-tensor-7"],
  },
  {
    id: "cs-eye",
    topic: "tensors",
    section: "Tensor Creation",
    title: "torch.eye()",
    description: "Identity matrix — 1s on the diagonal, 0s elsewhere.",
    syntax: "torch.eye(n)",
    example: `I = torch.eye(3)
print(I)`,
    result: `tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])`,
    interviewNote: "Handy for one-hot targets and for checking that a learned transform is close to identity.",
    importance: "low",
    tags: ["eye", "identity", "one-hot"],
  },
  {
    id: "cs-like",
    topic: "tensors",
    section: "Tensor Creation",
    title: "*_like() constructors",
    description:
      "zeros_like / ones_like / rand_like / randn_like copy the shape, dtype AND device of an existing tensor.",
    syntax: "torch.zeros_like(x)  ·  torch.ones_like(x)  ·  torch.rand_like(x)",
    example: `x = torch.randn(4, 5, device="cpu")
mask = torch.zeros_like(x)
print(mask.shape, mask.dtype, mask.device)`,
    result: "torch.Size([4, 5]) torch.float32 cpu",
    interviewNote:
      "The _like family is the safe way to build companion tensors on GPU: torch.zeros(x.shape) lands on the CPU and causes a device-mismatch error.",
    importance: "high",
    tags: ["zeros_like", "ones_like", "device", "mismatch"],
    relatedExercises: ["ex-tensor-8"],
  },
  {
    id: "cs-props",
    topic: "tensors",
    section: "Tensor Properties",
    title: "shape · size() · ndim · numel()",
    description:
      "shape is an attribute, size() is the method form and also accepts a dimension index. ndim is the rank, numel() the total element count.",
    syntax: "x.shape  ·  x.size()  ·  x.size(0)  ·  x.ndim  ·  x.numel()",
    example: `x = torch.randn(32, 3, 224, 224)
print(x.shape)
print(x.size(0), x.ndim, x.numel())`,
    result: `torch.Size([32, 3, 224, 224])
32 4 4816896`,
    interviewNote:
      "x.shape[0] is the batch size in nearly every training loop. torch.Size is a tuple subclass, so you can unpack it: B, C, H, W = x.shape.",
    importance: "high",
    tags: ["shape", "size", "ndim", "numel", "batch"],
    relatedExercises: ["ex-tensor-9", "ex-tensor-10"],
  },
  {
    id: "cs-dtype-device",
    topic: "tensors",
    section: "Tensor Properties",
    title: "dtype · device · requires_grad",
    description:
      "The three attributes to print first when something breaks. Most runtime errors in PyTorch are one of these three disagreeing.",
    syntax: "x.dtype  ·  x.device  ·  x.requires_grad",
    example: `x = torch.randn(2, 2, requires_grad=True)
print(x.dtype, x.device, x.requires_grad)`,
    result: "torch.float32 cpu True",
    interviewNote:
      "\"Expected all tensors to be on the same device\" and \"expected scalar type Long but found Float\" are both diagnosed by printing these.",
    importance: "high",
    tags: ["dtype", "device", "requires_grad", "debug"],
    relatedExercises: ["ex-tensor-11"],
  },
  {
    id: "cs-dtypes",
    topic: "tensors",
    section: "Data Types",
    title: "The dtypes that matter",
    description:
      "float32 is the default for activations and weights. int64 (torch.long) is required for class labels and embedding indices. bool is produced by comparisons.",
    syntax: "torch.float32 · torch.float64 · torch.float16 · torch.bfloat16 · torch.int64 · torch.int32 · torch.bool",
    example: `logits = torch.randn(8, 10)               # float32
targets = torch.randint(0, 10, (8,))      # int64 == torch.long
mask = targets > 4                        # bool
print(logits.dtype, targets.dtype, mask.dtype)`,
    result: "torch.float32 torch.int64 torch.bool",
    interviewNote:
      "nn.CrossEntropyLoss and nn.Embedding both require int64 indices. Passing float targets raises \"expected scalar type Long\". float64 tensors silently halve your throughput — avoid them on GPU.",
    importance: "high",
    tags: ["dtype", "long", "int64", "float32", "bfloat16", "labels"],
    relatedExercises: ["ex-dtype-1", "ex-dtype-2"],
  },
  {
    id: "cs-precision",
    topic: "tensors",
    section: "Data Types",
    title: "float16 vs bfloat16",
    description:
      "Both are 16-bit. float16 has more mantissa bits but a small exponent range and overflows easily. bfloat16 keeps float32's exponent range, so it rarely overflows and needs no loss scaling.",
    syntax: "x.half()  ·  x.to(torch.bfloat16)",
    example: `x = torch.randn(4, 4)
print(x.half().dtype, x.to(torch.bfloat16).dtype)`,
    result: "torch.float16 torch.bfloat16",
    interviewNote:
      "Modern LLM training uses bfloat16 on Ampere+ GPUs and TPUs precisely because it avoids the gradient-scaling machinery float16 needs.",
    importance: "medium",
    tags: ["float16", "bfloat16", "mixed precision", "amp"],
  },
  {
    id: "cs-cast",
    topic: "tensors",
    section: "Type Conversion",
    title: "Casting: .float() · .long() · .to()",
    description:
      "Shorthand casts exist for the common dtypes; .to() is the general form and also moves devices. All of them return a new tensor.",
    syntax: "x.float()  ·  x.long()  ·  x.int()  ·  x.bool()  ·  x.to(torch.float32)",
    example: `y = torch.tensor([1, 2, 3])       # int64
y = y.float()                     # float32
print(y.dtype, y.to(torch.long).dtype)`,
    result: "torch.float32 torch.int64",
    interviewNote:
      "Casts are not in-place: `x.float()` alone does nothing, you must reassign. The same trap applies to `x.to(device)`.",
    importance: "high",
    tags: ["cast", "float", "long", "to", "convert"],
    relatedExercises: ["ex-dtype-3", "ex-dtype-4"],
  },
  {
    id: "cs-item",
    topic: "tensors",
    section: "Tensor Properties",
    title: ".item() and .tolist()",
    description:
      "Pull Python numbers out of a tensor. .item() works on single-element tensors, .tolist() converts any tensor to nested Python lists.",
    syntax: "loss.item()  ·  x.tolist()",
    example: `loss = torch.tensor(0.3125)
running_loss = loss.item()
print(running_loss, torch.tensor([1, 2]).tolist())`,
    result: "0.3125 [1, 2]",
    interviewNote:
      "Always accumulate epoch losses with .item(). Summing the tensors themselves keeps the whole autograd graph alive and leaks GPU memory.",
    importance: "high",
    tags: ["item", "tolist", "logging", "memory leak"],
    relatedExercises: ["ex-tensor-12"],
  },
  {
    id: "cs-numpy",
    topic: "interop",
    section: "NumPy Interop",
    title: "torch.from_numpy() / .numpy()",
    description:
      "Convert between NumPy and PyTorch. Both directions share memory on the CPU, so mutating one mutates the other.",
    syntax: "torch.from_numpy(arr)  ·  tensor.numpy()",
    example: `import numpy as np
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)
arr[0] = 99.0
print(t)`,
    result: "tensor([99.,  2.,  3.], dtype=torch.float64)",
    interviewNote:
      "Two gotchas: NumPy defaults to float64 so you usually need .float(), and .numpy() fails on GPU or grad-tracking tensors — use x.detach().cpu().numpy().",
    importance: "medium",
    tags: ["numpy", "from_numpy", "interop", "shared memory"],
    relatedExercises: ["ex-interop-1", "ex-interop-2"],
  },
  {
    id: "cs-torch-vs-numpy",
    topic: "interop",
    section: "NumPy Interop",
    title: "What PyTorch adds over NumPy",
    description:
      "The tensor API deliberately mirrors ndarray. The three things NumPy cannot do are what make PyTorch a deep-learning framework.",
    syntax: "GPU execution · autograd · nn modules",
    example: `# NumPy                      # PyTorch
# np.array([1, 2])           torch.tensor([1, 2])
# np.zeros((2, 3))           torch.zeros(2, 3)
# arr.reshape(3, 2)          x.reshape(3, 2)
# arr.T                      x.T / x.transpose(0, 1)
# np.matmul(a, b)            torch.matmul(a, b)`,
    interviewNote:
      "A crisp answer: PyTorch tensors run on accelerators, record a backward graph for automatic differentiation, and plug into nn.Module/optimizers. Everything else is API parity.",
    importance: "medium",
    tags: ["numpy", "comparison", "gpu", "autograd"],
  },
  {
    id: "cs-seed",
    topic: "tensors",
    section: "Reproducibility",
    title: "torch.manual_seed()",
    description:
      "Seeds the CPU RNG (and, in recent versions, all GPUs) so weight init, dropout masks and shuffling repeat across runs.",
    syntax: "torch.manual_seed(42)  ·  torch.cuda.manual_seed_all(42)",
    example: `import random, numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)`,
    interviewNote:
      "Seeding is necessary but not sufficient. cuDNN picks nondeterministic kernels, atomics reorder float additions, and DataLoader workers have their own seeds — use torch.use_deterministic_algorithms(True) when you truly need bit-exact runs.",
    importance: "medium",
    tags: ["seed", "reproducibility", "determinism", "random"],
    relatedExercises: ["ex-repro-1"],
  },
];
