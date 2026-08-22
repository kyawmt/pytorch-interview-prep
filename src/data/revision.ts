import type { RevisionCard } from "@/lib/types";

/**
 * The "15-Minute PyTorch Review" deck: the ~30 things worth having fresh in
 * your head an hour before an interview. Each card is a prompt you should be
 * able to answer out loud before flipping it.
 */
export const REVISION_CARDS: RevisionCard[] = [
  {
    id: "rv-1",
    topic: "tensors",
    prompt: "Create a tensor; what dtype do you get?",
    answer:
      "torch.tensor() infers: integers → int64, floats → float32. torch.zeros/ones/randn are float32 by default.",
    code: "x = torch.tensor([1, 2, 3])        # int64\ny = torch.zeros(3, 4)              # float32",
  },
  {
    id: "rv-2",
    topic: "tensors",
    prompt: "How do you read shape, rank and element count?",
    answer: "x.shape / x.size(0) for one dimension, x.ndim for the rank, x.numel() for the total.",
    code: "x.shape[0]   # batch size\nx.ndim       # rank\nx.numel()    # total elements",
  },
  {
    id: "rv-3",
    topic: "shapes",
    prompt: "reshape vs view?",
    answer:
      "view needs contiguous memory and always shares storage; reshape copies when it cannot make a view. Use reshape by default.",
    code: "x.view(2, 12)\nx.t().reshape(4, 6)   # view would raise",
  },
  {
    id: "rv-4",
    topic: "shapes",
    prompt: "permute vs transpose vs reshape?",
    answer:
      "transpose swaps two axes, permute reorders all of them, reshape reinterprets the flat buffer. permute/transpose move axes; reshape does not.",
    code: "x.permute(0, 3, 1, 2)   # NHWC -> NCHW\nx.transpose(1, 2)       # swap two axes",
  },
  {
    id: "rv-5",
    topic: "shapes",
    prompt: "squeeze and unsqueeze?",
    answer:
      "squeeze(dim) removes a size-1 dimension, unsqueeze(dim) inserts one. unsqueeze(0) adds the batch dimension for single-sample inference.",
    code: "img.unsqueeze(0)     # (3,224,224) -> (1,3,224,224)\nout.squeeze(1)       # (32,1) -> (32,)",
  },
  {
    id: "rv-6",
    topic: "shapes",
    prompt: "cat vs stack?",
    answer:
      "cat joins along an existing dimension (rank unchanged); stack creates a new dimension (rank + 1).",
    code: "torch.cat([a, b], dim=0)    # (2,3)+(2,3) -> (4,3)\ntorch.stack([a, b])         # -> (2,2,3)",
  },
  {
    id: "rv-7",
    topic: "broadcasting",
    prompt: "State the broadcasting rule.",
    answer:
      "Align shapes from the RIGHT. Each pair must be equal or contain a 1; missing leading dimensions count as 1.",
    code: "(3, 4) + (4,)     -> (3, 4)\n(5, 1) * (1, 7)   -> (5, 7)",
  },
  {
    id: "rv-8",
    topic: "math",
    prompt: "matmul, mm, bmm — which is which?",
    answer:
      "matmul (and @) is the general form with broadcasting; mm is strictly 2-D; bmm is strictly 3-D batched.",
    code: "x @ w                  # (32,128)@(128,10) -> (32,10)\ntorch.bmm(a, b)        # (8,100,64)@(8,64,100)",
  },
  {
    id: "rv-9",
    topic: "math",
    prompt: "What does dim mean for reductions vs softmax?",
    answer:
      "For sum/mean/max, dim is the axis that DISAPPEARS. For softmax, dim is the axis that sums to 1 — the shape is unchanged.",
    code: "x.mean(dim=1)                  # (32,10) -> (32,)\ntorch.softmax(x, dim=1)        # (32,10) -> (32,10)",
  },
  {
    id: "rv-10",
    topic: "autograd",
    prompt: "What does requires_grad=True do?",
    answer:
      "It records operations on that tensor in the computational graph so backward() can fill its .grad. Model parameters have it set automatically.",
    code: "x = torch.tensor(2.0, requires_grad=True)\ny = x ** 2\ny.backward()\nx.grad   # tensor(4.)",
  },
  {
    id: "rv-11",
    topic: "autograd",
    prompt: "Where do gradients go after backward()?",
    answer:
      "Into the .grad attribute of every leaf tensor with requires_grad=True — i.e. the model parameters, which is what the optimizer reads.",
    code: "loss.backward()\nfor p in model.parameters():\n    print(p.grad.shape)",
  },
  {
    id: "rv-12",
    topic: "autograd",
    prompt: "Why do gradients accumulate?",
    answer:
      "backward() adds into .grad, which enables gradient accumulation and multi-loss training. The cost is you must call zero_grad() each step.",
    code: "optimizer.zero_grad()   # every step",
  },
  {
    id: "rv-13",
    topic: "autograd",
    prompt: "no_grad() vs detach()?",
    answer:
      "no_grad() is a block that prevents graph construction; detach() cuts one existing tensor out of the graph.",
    code: "with torch.no_grad(): ...\nlogged = loss.detach()",
  },
  {
    id: "rv-14",
    topic: "nn",
    prompt: "Sketch an nn.Module.",
    answer:
      "Subclass nn.Module, super().__init__() first, layers as attributes in __init__, data flow in forward(). Call model(x), not model.forward(x).",
    code: `class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x)`,
  },
  {
    id: "rv-15",
    topic: "nn",
    prompt: "What does nn.Linear do, and what shape is its weight?",
    answer:
      "y = xWᵀ + b applied to the LAST dimension; every leading dimension is treated as batch. weight is (out_features, in_features).",
    code: "fc = nn.Linear(128, 10)\nfc(torch.randn(8, 100, 128)).shape   # (8, 100, 10)",
  },
  {
    id: "rv-16",
    topic: "nn",
    prompt: "BatchNorm vs LayerNorm?",
    answer:
      "BatchNorm normalises per channel across the batch and keeps running statistics. LayerNorm normalises per sample across features — batch independent, so transformers use it.",
    code: "nn.BatchNorm2d(16)   # CNNs\nnn.LayerNorm(768)    # transformers",
  },
  {
    id: "rv-17",
    topic: "losses",
    prompt: "CrossEntropyLoss — inputs and traps?",
    answer:
      "Raw logits (B, C) and int64 targets (B,). It applies log_softmax internally, so never softmax first.",
    code: "loss = nn.CrossEntropyLoss()(logits, y)",
  },
  {
    id: "rv-18",
    topic: "losses",
    prompt: "Binary classification loss?",
    answer:
      "BCEWithLogitsLoss on raw logits — it fuses sigmoid for numerical stability. Targets are FLOAT and the same shape as the logits.",
    code: "nn.BCEWithLogitsLoss()(logits, y.float())",
  },
  {
    id: "rv-19",
    topic: "optimizers",
    prompt: "Adam vs AdamW?",
    answer:
      "AdamW decouples weight decay — it subtracts it from the weights instead of adding it to the gradient. It is the default for transformers.",
    code: "torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)",
  },
  {
    id: "rv-20",
    topic: "training",
    prompt: "Write the training loop.",
    answer: "train() → zero_grad → forward → loss → backward → step. Move the batch to the device first.",
    code: `model.train()
for X, y in loader:
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()`,
  },
  {
    id: "rv-21",
    topic: "training",
    prompt: "What do zero_grad / backward / step each do?",
    answer:
      "zero_grad clears p.grad; backward fills it via the chain rule; step reads it and updates the weights.",
    code: "optimizer.zero_grad()\nloss.backward()\noptimizer.step()",
  },
  {
    id: "rv-22",
    topic: "evaluation",
    prompt: "eval() vs no_grad()?",
    answer:
      "eval() switches dropout off and BatchNorm to running statistics. no_grad() stops graph recording. You want both, and they are independent.",
    code: "model.eval()\nwith torch.no_grad():\n    pred = model(X)",
  },
  {
    id: "rv-23",
    topic: "evaluation",
    prompt: "Accuracy in one line?",
    answer: "argmax over the class dimension, compare to the labels, cast to float, take the mean.",
    code: "(logits.argmax(1) == y).float().mean().item()",
  },
  {
    id: "rv-24",
    topic: "data",
    prompt: "What must a Dataset implement?",
    answer: "__len__ and __getitem__. __getitem__ returns ONE sample; the DataLoader batches them.",
    code: `class DS(Dataset):
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]`,
  },
  {
    id: "rv-25",
    topic: "data",
    prompt: "Key DataLoader arguments?",
    answer:
      "batch_size, shuffle (training only), num_workers for parallel loading, pin_memory for faster host→device copies, drop_last for uniform batches.",
    code: "DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)",
  },
  {
    id: "rv-26",
    topic: "gpu",
    prompt: "Device-agnostic setup?",
    answer:
      "One device variable; model.to(device) is in place, tensor .to(device) is NOT and must be reassigned.",
    code: 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\nmodel.to(device)\nX = X.to(device)',
  },
  {
    id: "rv-27",
    topic: "saving",
    prompt: "Save and load a model.",
    answer:
      "Save the state_dict, rebuild the architecture, load into it, then call eval(). Add the optimizer state if you want to resume training.",
    code: 'torch.save(model.state_dict(), "m.pth")\nmodel.load_state_dict(torch.load("m.pth"))\nmodel.eval()',
  },
  {
    id: "rv-28",
    topic: "debugging",
    prompt: "First three things to print when something breaks?",
    answer: "shape, dtype, device — they account for nearly every PyTorch runtime error.",
    code: "print(x.shape, x.dtype, x.device)",
  },
  {
    id: "rv-29",
    topic: "debugging",
    prompt: "Loss is NaN. Checklist?",
    answer:
      "Lower the learning rate, clip gradients, check for log/div by zero, check the inputs for NaN, use the fused *WithLogits losses, and enable anomaly detection to find the op.",
    code: "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\ntorch.autograd.set_detect_anomaly(True)",
  },
  {
    id: "rv-30",
    topic: "transformers",
    prompt: "Scaled dot-product attention?",
    answer:
      "softmax(QKᵀ/√d)V. The scaling stops softmax saturating; the softmax is over the key axis (dim=-1).",
    code: "attn = torch.softmax(Q @ K.transpose(-2, -1) / d ** 0.5, dim=-1)\nout = attn @ V",
  },
];
