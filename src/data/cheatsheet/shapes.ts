import type { CheatSheetEntry } from "@/lib/types";

export const shapeEntries: CheatSheetEntry[] = [
  {
    id: "cs-reshape",
    topic: "shapes",
    section: "Reshaping",
    title: "reshape() vs view()",
    description:
      "Both reinterpret the same elements under a new shape. view() requires the tensor to be contiguous and always shares storage; reshape() falls back to a copy when it cannot create a view.",
    syntax: "x.reshape(*shape)  ·  x.view(*shape)  ·  -1 infers one dimension",
    example: `x = torch.randn(4, 6)
print(x.view(2, 12).shape)
print(x.reshape(-1).shape)      # -1 = "work it out"
y = x.t()                       # transpose -> non-contiguous
print(y.reshape(4, 6).shape)    # works (copies)
# y.view(4, 6)                  # RuntimeError`,
    result: `torch.Size([2, 12])
torch.Size([24])
torch.Size([4, 6])`,
    interviewNote:
      "The answer interviewers want: view is a zero-copy re-interpretation that needs contiguous memory; reshape does the same when possible and silently copies otherwise. Use reshape unless you specifically want the failure when a copy would occur.",
    importance: "high",
    tags: ["reshape", "view", "contiguous", "-1"],
    relatedExercises: ["ex-shape-1", "ex-shape-2", "ex-shape-3"],
  },
  {
    id: "cs-contiguous",
    topic: "shapes",
    section: "Reshaping",
    title: ".contiguous() and memory layout",
    description:
      "permute/transpose only change the strides — the underlying buffer is untouched. .contiguous() materialises a new tensor whose memory order matches its shape.",
    syntax: "x.is_contiguous()  ·  x.contiguous()",
    example: `x = torch.randn(2, 3, 4)
y = x.permute(1, 0, 2)
print(y.is_contiguous())
print(y.contiguous().is_contiguous())`,
    result: `False
True`,
    interviewNote:
      "The classic transformer line `x.transpose(1, 2).contiguous().view(B, T, C)` exists precisely because view() refuses non-contiguous input.",
    importance: "medium",
    tags: ["contiguous", "stride", "memory", "permute"],
    relatedExercises: ["ex-shape-4"],
  },
  {
    id: "cs-flatten",
    topic: "shapes",
    section: "Reshaping",
    title: "flatten()",
    description:
      "Collapses a range of dimensions into one. flatten(1) is the standard bridge from a conv feature map to a linear classifier.",
    syntax: "x.flatten(start_dim=0, end_dim=-1)",
    example: `x = torch.randn(32, 3, 224, 224)
print(x.flatten().shape)
print(x.flatten(1).shape)     # keep the batch dimension`,
    result: `torch.Size([4816896])
torch.Size([32, 150528])`,
    interviewNote:
      "Forgetting the `1` flattens the batch away and produces a single giant vector — an extremely common bug in hand-written CNN heads. nn.Flatten() defaults to start_dim=1 for exactly this reason.",
    importance: "high",
    tags: ["flatten", "cnn", "batch", "classifier"],
    relatedExercises: ["ex-shape-5", "ex-shape-6"],
  },
  {
    id: "cs-squeeze",
    topic: "shapes",
    section: "Reshaping",
    title: "squeeze() / unsqueeze()",
    description:
      "squeeze removes size-1 dimensions; unsqueeze inserts one at the given position. Both are pure shape operations — no data moves.",
    syntax: "x.squeeze(dim=None)  ·  x.unsqueeze(dim)",
    example: `x = torch.randn(1, 3, 1)
print(x.squeeze().shape)        # all size-1 dims
print(x.squeeze(0).shape)       # only dim 0
y = torch.randn(3)
print(y.unsqueeze(0).shape)     # add a batch dim`,
    result: `torch.Size([3])
torch.Size([3, 1])
torch.Size([1, 3])`,
    interviewNote:
      "Always pass an explicit dim to squeeze() in library code: a bare squeeze() on a batch of size 1 silently deletes the batch dimension. unsqueeze(0) is how you feed a single sample to a model that expects a batch.",
    importance: "high",
    tags: ["squeeze", "unsqueeze", "batch dimension", "None indexing"],
    relatedExercises: ["ex-shape-7", "ex-shape-8"],
  },
  {
    id: "cs-permute",
    topic: "shapes",
    section: "Reshaping",
    title: "permute() vs transpose()",
    description:
      "transpose swaps exactly two dimensions. permute reorders all of them at once, and you must list every dimension.",
    syntax: "x.transpose(dim0, dim1)  ·  x.permute(*dims)",
    example: `x = torch.randn(32, 224, 224, 3)   # NHWC (e.g. from PIL/NumPy)
print(x.permute(0, 3, 1, 2).shape)  # -> NCHW for Conv2d
y = torch.randn(8, 100, 512)        # (B, T, C)
print(y.transpose(1, 2).shape)      # -> (B, C, T)`,
    result: `torch.Size([32, 3, 224, 224])
torch.Size([8, 512, 100])`,
    interviewNote:
      "permute/transpose move axes and keep the data; reshape/view reinterpret the flat buffer. Using reshape where you needed permute scrambles the image — it does not error, it silently produces garbage.",
    importance: "high",
    tags: ["permute", "transpose", "NCHW", "NHWC", "axes"],
    relatedExercises: ["ex-shape-9", "ex-shape-10", "ex-shape-11"],
  },
  {
    id: "cs-indexing",
    topic: "shapes",
    section: "Indexing & Slicing",
    title: "Basic indexing and slicing",
    description:
      "Indexing follows NumPy. A plain integer index removes that dimension; a slice keeps it.",
    syntax: "x[0]  ·  x[:, 0]  ·  x[1:4]  ·  x[..., 0]  ·  x[:, None]",
    example: `x = torch.randn(8, 3, 32, 32)
print(x[0].shape)        # first sample
print(x[:, 0].shape)     # first channel of every sample
print(x[1:4].shape)      # a slice of the batch
print(x[..., 0].shape)   # last dim indexed, rest untouched`,
    result: `torch.Size([3, 32, 32])
torch.Size([8, 32, 32])
torch.Size([3, 3, 32, 32])
torch.Size([8, 3, 32])`,
    interviewNote:
      "x[0] drops a dimension, x[0:1] keeps it. Ellipsis (...) means \"all remaining dimensions\" and keeps code rank-agnostic.",
    importance: "high",
    tags: ["indexing", "slicing", "ellipsis", "dimension"],
    relatedExercises: ["ex-index-1", "ex-index-2"],
  },
  {
    id: "cs-bool-index",
    topic: "shapes",
    section: "Indexing & Slicing",
    title: "Boolean masking",
    description:
      "A bool tensor used as an index selects the elements where it is True and always returns a 1-D tensor.",
    syntax: "x[x > 0]  ·  x.masked_fill(mask, value)  ·  torch.where(cond, a, b)",
    example: `x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
print(x[x > 0])
print(torch.where(x > 0, x, torch.zeros_like(x)))`,
    result: `tensor([2., 3.])
tensor([[0., 2.],
        [3., 0.]])`,
    interviewNote:
      "Boolean indexing loses the shape; torch.where keeps it. masked_fill(mask, -inf) before softmax is how attention masking and padding masks are implemented.",
    importance: "medium",
    tags: ["boolean", "mask", "where", "masked_fill", "attention"],
    relatedExercises: ["ex-index-3"],
  },
  {
    id: "cs-gather",
    topic: "shapes",
    section: "Indexing & Slicing",
    title: "Fancy indexing & gather()",
    description:
      "Index with a tensor of positions to pick arbitrary rows, or use gather() to pull one value per row along a dimension.",
    syntax: "x[idx]  ·  torch.gather(x, dim, index)",
    example: `logits = torch.randn(4, 10)
targets = torch.tensor([3, 1, 9, 0])
chosen = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
print(chosen.shape)`,
    result: "torch.Size([4])",
    interviewNote:
      "gather is how you pick the logit of the correct class per sample (the core of a manual cross-entropy or of RL policy losses). The index tensor must be int64 and match the rank of the source.",
    importance: "medium",
    tags: ["gather", "fancy indexing", "index_select", "cross entropy"],
  },
  {
    id: "cs-cat",
    topic: "shapes",
    section: "Combining Tensors",
    title: "torch.cat() vs torch.stack()",
    description:
      "cat joins tensors along an EXISTING dimension and the rank stays the same. stack creates a NEW dimension and the rank increases by one.",
    syntax: "torch.cat(tensors, dim=0)  ·  torch.stack(tensors, dim=0)",
    example: `a = torch.randn(2, 3)
b = torch.randn(2, 3)
print(torch.cat([a, b], dim=0).shape)
print(torch.cat([a, b], dim=1).shape)
print(torch.stack([a, b], dim=0).shape)`,
    result: `torch.Size([4, 3])
torch.Size([2, 6])
torch.Size([2, 2, 3])`,
    interviewNote:
      "cat requires matching sizes on every dim except the concat dim; stack requires the shapes to be identical. Collecting per-batch predictions across an epoch → cat. Turning a list of samples into a batch → stack.",
    importance: "high",
    tags: ["cat", "stack", "concat", "batch", "shape"],
    relatedExercises: ["ex-combine-1", "ex-combine-2", "ex-combine-3"],
  },
  {
    id: "cs-split",
    topic: "shapes",
    section: "Combining Tensors",
    title: "chunk() / split() / unbind()",
    description: "The inverses of cat and stack: cut a tensor into pieces along a dimension.",
    syntax: "x.chunk(n, dim=0)  ·  x.split(size, dim=0)  ·  x.unbind(dim)",
    example: `qkv = torch.randn(8, 100, 3 * 64)
q, k, v = qkv.chunk(3, dim=-1)
print(q.shape)`,
    result: "torch.Size([8, 100, 64])",
    interviewNote:
      "chunk(3, dim=-1) on a fused QKV projection is the standard trick in compact attention implementations — one matmul instead of three.",
    importance: "medium",
    tags: ["chunk", "split", "unbind", "qkv", "attention"],
  },
  {
    id: "cs-expand",
    topic: "shapes",
    section: "Combining Tensors",
    title: "expand() vs repeat()",
    description:
      "expand creates a zero-copy view by setting a stride of 0 on size-1 dimensions. repeat physically copies the data.",
    syntax: "x.expand(*sizes)  ·  x.repeat(*repeats)  ·  x.repeat_interleave(n, dim)",
    example: `x = torch.randn(1, 4)
print(x.expand(3, 4).shape)   # no new memory
print(x.repeat(3, 1).shape)   # 3x the memory`,
    result: `torch.Size([3, 4])
torch.Size([3, 4])`,
    interviewNote:
      "expand only works on dimensions of size 1 and the result shares memory, so writing into it is unsafe. Prefer expand for broadcasting-style work; use repeat only when you truly need independent copies.",
    importance: "medium",
    tags: ["expand", "repeat", "memory", "broadcast"],
  },
  {
    id: "cs-broadcast-rules",
    topic: "broadcasting",
    section: "Broadcasting",
    title: "The broadcasting rules",
    description:
      "Align shapes from the RIGHT. Two dimensions are compatible when they are equal, or one of them is 1. Missing leading dimensions are treated as 1.",
    syntax: "(3, 4) + (4,) -> (3, 4)   ·   (32, 10) + (10,) -> (32, 10)",
    example: `a = torch.randn(3, 4)
b = torch.randn(4)
print((a + b).shape)

x = torch.randn(32, 10)
bias = torch.randn(10)
print((x + bias).shape)`,
    result: `torch.Size([3, 4])
torch.Size([32, 10])`,
    interviewNote:
      "This is exactly how a Linear layer's bias is added to a batch. Say the rule out loud in the interview: right-align, then each pair must be equal or contain a 1.",
    importance: "high",
    tags: ["broadcasting", "shapes", "bias", "rules"],
    relatedExercises: ["ex-broadcast-1", "ex-broadcast-2", "ex-broadcast-3"],
  },
  {
    id: "cs-broadcast-trap",
    topic: "broadcasting",
    section: "Broadcasting",
    title: "The silent broadcasting bug",
    description:
      "Broadcasting can make a shape mistake succeed instead of raising. The classic case is a loss computed between (N,) predictions and (N, 1) targets.",
    syntax: "pred.squeeze() vs target.view(-1, 1)",
    example: `pred = torch.randn(32)        # (32,)
target = torch.randn(32, 1)   # (32, 1)
print((pred - target).shape)  # NOT (32,)`,
    result: "torch.Size([32, 32])",
    interviewNote:
      "An MSE over a (32, 32) tensor still returns a number, so training silently produces nonsense. When a loss looks strange, print the shapes of both arguments first.",
    importance: "high",
    tags: ["broadcasting", "bug", "loss", "shape mismatch"],
    relatedExercises: ["ex-broadcast-4"],
  },
  {
    id: "cs-broadcast-newaxis",
    topic: "broadcasting",
    section: "Broadcasting",
    title: "Making broadcasting explicit",
    description:
      "Insert size-1 dimensions with unsqueeze or None to control which axes align, e.g. for pairwise distances or attention masks.",
    syntax: "a[:, None] - b[None, :]",
    example: `a = torch.randn(5, 3)
b = torch.randn(7, 3)
diff = a[:, None, :] - b[None, :, :]
print(diff.shape)     # pairwise differences`,
    result: "torch.Size([5, 7, 3])",
    interviewNote:
      "Being able to derive (5, 1, 3) vs (1, 7, 3) → (5, 7, 3) on a whiteboard is a common shape-reasoning question.",
    importance: "medium",
    tags: ["broadcasting", "None", "unsqueeze", "pairwise"],
  },
  {
    id: "cs-shape-patterns",
    topic: "shapes",
    section: "Common Shape Patterns",
    title: "Shape conventions to memorise",
    description:
      "Almost every tensor you meet fits one of these layouts. Knowing them makes shape errors instantly readable.",
    syntax: "(B, F) · (B, C, H, W) · (B, T, C) · (B, num_classes)",
    example: `# Tabular / MLP input        (batch_size, num_features)
# Images (PyTorch NCHW)      (batch_size, channels, height, width)
# Text / transformers        (batch_size, seq_len, hidden_size)
# Classification logits      (batch_size, num_classes)
# Class labels               (batch_size,)          dtype int64
# Attention scores           (batch, heads, seq, seq)`,
    interviewNote:
      "PyTorch is channels-first (NCHW) while TensorFlow and PIL are channels-last (NHWC) — that permute is where most image bugs live. Note labels are 1-D, not one-hot, for CrossEntropyLoss.",
    importance: "high",
    tags: ["shapes", "NCHW", "batch", "conventions", "logits"],
    relatedExercises: ["ex-shape-12", "ex-shape-13"],
  },
  {
    id: "cs-conv-shape",
    topic: "shapes",
    section: "Common Shape Patterns",
    title: "Conv2d and pooling output size",
    description:
      "The formula every CNN question reduces to: out = floor((in + 2*padding - kernel) / stride) + 1.",
    syntax: "out = (in + 2p - k) // s + 1",
    example: `# (B, 3, 32, 32) -> Conv2d(3, 16, kernel_size=3, padding=1)
# (32 + 2 - 3) // 1 + 1 = 32   ->  (B, 16, 32, 32)
# -> MaxPool2d(2)              ->  (B, 16, 16, 16)
x = torch.randn(8, 3, 32, 32)
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
print(nn.MaxPool2d(2)(conv(x)).shape)`,
    result: "torch.Size([8, 16, 16, 16])",
    interviewNote:
      "kernel_size=3, padding=1, stride=1 preserves H and W — that is why it is the default choice in almost every CNN. MaxPool2d(2) halves both.",
    importance: "high",
    tags: ["conv2d", "maxpool", "output size", "cnn", "formula"],
    relatedExercises: ["ex-shape-14", "ex-shape-15"],
  },
];
