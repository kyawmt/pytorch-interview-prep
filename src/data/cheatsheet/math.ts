import type { CheatSheetEntry } from "@/lib/types";

export const mathEntries: CheatSheetEntry[] = [
  {
    id: "cs-elementwise",
    topic: "math",
    section: "Element-wise Math",
    title: "Arithmetic and in-place ops",
    description:
      "+ - * / are element-wise and broadcast. Methods ending in an underscore mutate the tensor in place.",
    syntax: "x + y  ·  x * y  ·  x.add_(y)  ·  torch.abs / sqrt / exp / log / clamp",
    example: `x = torch.tensor([1.0, 4.0, 9.0])
print(torch.sqrt(x))
print(x.clamp(min=2.0, max=5.0))`,
    result: `tensor([1., 2., 3.])
tensor([2., 4., 5.])`,
    interviewNote:
      "In-place ops (add_, relu_) save memory but raise \"a variable needed for gradient computation has been modified\" if the original value is needed by backward. When autograd complains, remove the underscore first.",
    importance: "medium",
    tags: ["elementwise", "in-place", "clamp", "sqrt", "autograd error"],
    relatedExercises: ["ex-math-1"],
  },
  {
    id: "cs-reductions",
    topic: "math",
    section: "Reductions",
    title: "sum / mean / max / min and dim",
    description:
      "Without dim they reduce the whole tensor to a scalar. With dim they collapse that axis; keepdim=True keeps it as size 1.",
    syntax: "x.sum(dim=None, keepdim=False)  ·  x.mean(dim)  ·  x.max(dim)",
    example: `x = torch.randn(32, 10)
print(x.sum().shape)
print(x.mean(dim=1).shape)          # per-sample mean
print(x.mean(dim=1, keepdim=True).shape)`,
    result: `torch.Size([])
torch.Size([32])
torch.Size([32, 1])`,
    interviewNote:
      "Rule of thumb: `dim` names the dimension that disappears. mean() on an integer tensor errors — cast to float first. keepdim=True is what lets the result broadcast back against the original.",
    importance: "high",
    tags: ["sum", "mean", "dim", "keepdim", "reduction"],
    relatedExercises: ["ex-math-2", "ex-math-3", "ex-math-4"],
  },
  {
    id: "cs-argmax",
    topic: "math",
    section: "Reductions",
    title: "argmax / argmin and max with dim",
    description:
      "argmax returns the index of the largest value. max(dim=...) returns both values and indices as a named tuple.",
    syntax: "x.argmax(dim)  ·  values, indices = x.max(dim)  ·  torch.topk(x, k, dim)",
    example: `logits = torch.randn(4, 10)
preds = logits.argmax(dim=1)
print(preds.shape, preds.dtype)
values, indices = logits.max(dim=1)
print(values.shape, indices.shape)`,
    result: `torch.Size([4]) torch.int64
torch.Size([4]) torch.Size([4])`,
    interviewNote:
      "argmax(dim=1) on (B, num_classes) logits gives predicted class ids — the standard accuracy line is `(logits.argmax(1) == y).float().mean()`. Because softmax is monotonic, applying it before argmax changes nothing.",
    importance: "high",
    tags: ["argmax", "accuracy", "topk", "prediction", "logits"],
    relatedExercises: ["ex-math-5", "ex-math-6"],
  },
  {
    id: "cs-matmul",
    topic: "math",
    section: "Matrix Operations",
    title: "matmul / @ / mm / bmm / dot",
    description:
      "matmul (and its @ operator) is the general form: it handles 1-D, 2-D and batched inputs with broadcasting. mm is strictly 2-D, bmm strictly 3-D batched, dot strictly 1-D.",
    syntax: "torch.matmul(a, b)  ·  a @ b  ·  torch.mm  ·  torch.bmm  ·  torch.dot",
    example: `a = torch.randn(8, 100, 64)   # (B, T, C)
b = torch.randn(8, 64, 100)
print(torch.bmm(a, b).shape)
print((a @ b).shape)          # matmul handles it too

x = torch.randn(32, 128)
w = torch.randn(128, 10)
print((x @ w).shape)`,
    result: `torch.Size([8, 100, 100])
torch.Size([8, 100, 100])
torch.Size([32, 10])`,
    interviewNote:
      "Use @ / matmul by default; reach for mm or bmm when you want the stricter version to catch rank mistakes. The inner dimensions must match: (n, k) @ (k, m) -> (n, m).",
    importance: "high",
    tags: ["matmul", "mm", "bmm", "dot", "attention", "linear"],
    relatedExercises: ["ex-math-7", "ex-math-8", "ex-math-9"],
  },
  {
    id: "cs-softmax",
    topic: "math",
    section: "Matrix Operations",
    title: "torch.softmax(x, dim)",
    description:
      "Turns raw scores into a probability distribution along one dimension: values in (0, 1) summing to 1 over that axis.",
    syntax: "torch.softmax(x, dim)  ·  F.softmax(x, dim=-1)  ·  F.log_softmax(x, dim=-1)",
    example: `logits = torch.randn(4, 10)
probs = torch.softmax(logits, dim=1)
print(probs.shape, probs.sum(dim=1))`,
    result: "torch.Size([4, 10]) tensor([1., 1., 1., 1.])",
    interviewNote:
      "`dim` is the axis that sums to 1 — for (B, num_classes) logits that is dim=1 (or dim=-1). Do NOT apply softmax before nn.CrossEntropyLoss: it already does log_softmax internally.",
    importance: "high",
    tags: ["softmax", "dim", "probabilities", "logits", "cross entropy"],
    relatedExercises: ["ex-math-10", "ex-math-11"],
  },
  {
    id: "cs-norm",
    topic: "math",
    section: "Matrix Operations",
    title: "Norms and normalisation",
    description:
      "torch.norm/linalg.norm measure magnitude; F.normalize rescales vectors to unit length, which is what cosine similarity needs.",
    syntax: "torch.linalg.norm(x, dim=-1)  ·  F.normalize(x, p=2, dim=-1)",
    example: `emb = torch.randn(4, 128)
unit = F.normalize(emb, dim=-1)
cos = unit @ unit.T
print(cos.shape)`,
    result: "torch.Size([4, 4])",
    interviewNote:
      "Cosine similarity = normalize + matmul. This is the entire retrieval/embedding-search primitive behind RAG systems.",
    importance: "medium",
    tags: ["norm", "normalize", "cosine similarity", "embeddings", "rag"],
  },
];
