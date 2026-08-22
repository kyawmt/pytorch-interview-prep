import type { CheatSheetEntry } from "@/lib/types";

export const trainingEntries: CheatSheetEntry[] = [
  {
    id: "cs-mse",
    topic: "losses",
    section: "Loss Functions",
    title: "nn.MSELoss / nn.L1Loss",
    description:
      "Regression losses. MSE squares the error (heavier penalty on outliers), L1 uses the absolute error (robust to them). SmoothL1/Huber sits in between.",
    syntax: "nn.MSELoss()  ·  nn.L1Loss()  ·  nn.SmoothL1Loss()",
    example: `loss_fn = nn.MSELoss()
pred = torch.randn(32, 1)
target = torch.randn(32, 1)
print(loss_fn(pred, target).shape)   # scalar`,
    result: "torch.Size([])",
    interviewNote:
      "pred and target must have the SAME shape — (32,) vs (32, 1) broadcasts to (32, 32) and silently returns a meaningless number. Check with pred.shape == target.shape.",
    importance: "high",
    tags: ["mse", "l1", "regression", "huber", "shape mismatch"],
    relatedExercises: ["ex-loss-1", "ex-broadcast-4"],
  },
  {
    id: "cs-crossentropy",
    topic: "losses",
    section: "Loss Functions",
    title: "nn.CrossEntropyLoss",
    description:
      "Multi-class classification. Takes RAW LOGITS of shape (B, num_classes) and integer class labels of shape (B,). Internally it is log_softmax + NLLLoss.",
    syntax: "nn.CrossEntropyLoss(weight=None, label_smoothing=0.0, ignore_index=-100)",
    example: `loss_fn = nn.CrossEntropyLoss()
logits = torch.randn(32, 10)              # NOT softmaxed
targets = torch.randint(0, 10, (32,))     # int64, class ids
loss = loss_fn(logits, targets)
print(loss.item())`,
    interviewNote:
      "🔥 The single most asked PyTorch loss question. Do NOT apply softmax first: you would take log of a softmax of a softmax, which flattens the distribution, weakens gradients and hurts accuracy. Targets must be int64 class indices, not one-hot.",
    importance: "high",
    tags: ["crossentropy", "logits", "softmax", "classification", "int64"],
    relatedExercises: ["ex-loss-2", "ex-loss-3", "ex-loss-4"],
  },
  {
    id: "cs-bce",
    topic: "losses",
    section: "Loss Functions",
    title: "nn.BCELoss vs nn.BCEWithLogitsLoss",
    description:
      "Binary / multi-label classification. BCELoss expects probabilities (after sigmoid). BCEWithLogitsLoss takes raw logits and applies sigmoid internally.",
    syntax: "nn.BCEWithLogitsLoss(pos_weight=None)",
    example: `loss_fn = nn.BCEWithLogitsLoss()
logits = torch.randn(32, 1)
targets = torch.randint(0, 2, (32, 1)).float()
print(loss_fn(logits, targets).item())`,
    interviewNote:
      "Always prefer BCEWithLogitsLoss: fusing sigmoid and BCE uses the log-sum-exp trick, so it stays numerically stable when logits are large. Targets must be FLOAT (0.0/1.0), unlike CrossEntropyLoss which wants int64.",
    importance: "high",
    tags: ["bce", "logits", "sigmoid", "binary", "stability", "multi-label"],
    relatedExercises: ["ex-loss-5", "ex-loss-6"],
  },
  {
    id: "cs-loss-choice",
    topic: "losses",
    section: "Loss Functions",
    title: "Choosing a loss (decision table)",
    description: "Match the task to the loss and the final layer. Most loss bugs are really a task/loss mismatch.",
    syntax: "task -> output shape -> loss",
    example: `# Regression            (B, 1) or (B, k)   MSELoss / L1Loss        no activation
# Binary classification  (B, 1) logits       BCEWithLogitsLoss       no activation
# Multi-class            (B, C) logits       CrossEntropyLoss        no activation
# Multi-label            (B, C) logits       BCEWithLogitsLoss       no activation
# Ranking / similarity   embeddings          CosineEmbeddingLoss     normalize`,
    interviewNote:
      "Notice every row says \"no activation\": in modern PyTorch the loss owns the final nonlinearity. You only apply sigmoid/softmax explicitly at inference time to read off probabilities.",
    importance: "high",
    tags: ["loss selection", "logits", "task", "table"],
  },
  {
    id: "cs-sgd",
    topic: "optimizers",
    section: "Optimizers",
    title: "optim.SGD",
    description:
      "The baseline. With momentum=0.9 it accumulates a velocity term that smooths the descent direction and speeds up convergence.",
    syntax: "optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)",
    example: `optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
)`,
    interviewNote:
      "SGD + momentum + a good LR schedule still gives the best final accuracy on many vision benchmarks — it generalises slightly better than Adam, at the cost of more tuning.",
    importance: "high",
    tags: ["sgd", "momentum", "learning rate", "weight decay"],
    relatedExercises: ["ex-optim-1"],
  },
  {
    id: "cs-adam",
    topic: "optimizers",
    section: "Optimizers",
    title: "optim.Adam vs optim.AdamW",
    description:
      "Adam adapts a per-parameter learning rate from running estimates of the first and second gradient moments. AdamW fixes how weight decay is applied.",
    syntax: "optim.Adam(params, lr=1e-3)  ·  optim.AdamW(params, lr=1e-3, weight_decay=0.01)",
    example: `optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
)`,
    interviewNote:
      "🔥 Adam adds weight decay INTO the gradient, so the adaptive scaling distorts it. AdamW subtracts it directly from the weights (decoupled), which is why every transformer recipe uses AdamW. Defaults worth knowing: lr=1e-3, betas=(0.9, 0.999), eps=1e-8.",
    importance: "high",
    tags: ["adam", "adamw", "weight decay", "decoupled", "transformer"],
    relatedExercises: ["ex-optim-2", "ex-optim-3"],
  },
  {
    id: "cs-param-groups",
    topic: "optimizers",
    section: "Optimizers",
    title: "Parameter groups",
    description:
      "Pass a list of dicts to give different layers different hyper-parameters — a lower LR for a pretrained backbone, no weight decay for biases and norms.",
    syntax: "optim.AdamW([{'params': ..., 'lr': ...}, ...])",
    example: `decay, no_decay = [], []
for name, p in model.named_parameters():
    (no_decay if p.ndim < 2 else decay).append(p)

optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": 0.01},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4)`,
    interviewNote:
      "Excluding biases and LayerNorm weights from weight decay is standard in LLM training recipes — decaying a normalisation scale toward zero actively hurts.",
    importance: "medium",
    tags: ["param groups", "weight decay", "fine-tuning", "llm"],
  },
  {
    id: "cs-scheduler",
    topic: "optimizers",
    section: "Schedulers",
    title: "Learning-rate schedulers",
    description:
      "Wrap an optimizer and change its LR over time. StepLR drops by a factor every N epochs, CosineAnnealingLR decays smoothly, ReduceLROnPlateau reacts to a metric.",
    syntax: "scheduler.step()  ·  scheduler.step(val_loss) for ReduceLROnPlateau",
    example: `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    train_one_epoch()
    val_loss = validate()
    scheduler.step()                 # per-epoch schedulers
print(scheduler.get_last_lr())`,
    interviewNote:
      "Order matters: optimizer.step() then scheduler.step(). ReduceLROnPlateau is the exception that takes the metric as an argument. Most schedulers step per epoch; warmup schedules step per batch.",
    importance: "medium",
    tags: ["scheduler", "steplr", "cosine", "plateau", "warmup"],
    relatedExercises: ["ex-optim-4"],
  },
  {
    id: "cs-training-loop",
    topic: "training",
    section: "The Training Loop",
    title: "The canonical training loop",
    description:
      "Five lines, always in this order. Being able to type this from memory is the single highest-value PyTorch skill for an interview.",
    syntax: "zero_grad -> forward -> loss -> backward -> step",
    example: `model.train()
for X, y in train_loader:
    X, y = X.to(device), y.to(device)

    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()`,
    interviewNote:
      "🔥 Say what each line does: clear stale gradients, compute predictions, measure error, backpropagate into .grad, apply the update. zero_grad can go at the top or right after step() — what matters is that it happens once per batch.",
    importance: "high",
    tags: ["training loop", "zero_grad", "backward", "step", "memorise"],
    relatedExercises: ["ex-train-1", "ex-train-2", "ex-train-3", "ex-train-8"],
  },
  {
    id: "cs-zero-grad",
    topic: "training",
    section: "The Training Loop",
    title: "optimizer.zero_grad()",
    description:
      "Clears .grad on every parameter the optimizer owns. Required because backward() accumulates instead of overwriting.",
    syntax: "optimizer.zero_grad(set_to_none=True)",
    example: `optimizer.zero_grad()   # set_to_none=True is the default in modern PyTorch
loss.backward()
optimizer.step()`,
    interviewNote:
      "Forget it and each step uses the sum of all previous gradients — the effective learning rate grows every batch and training diverges. set_to_none=True frees the gradient buffers instead of filling them with zeros (slightly faster, less memory).",
    importance: "high",
    tags: ["zero_grad", "accumulate", "set_to_none", "bug"],
    relatedExercises: ["ex-train-2", "ex-train-9"],
  },
  {
    id: "cs-step",
    topic: "training",
    section: "The Training Loop",
    title: "loss.backward() and optimizer.step()",
    description:
      "backward() fills p.grad for every parameter. step() reads those gradients and updates the weights in place. They are two separate objects with no direct link beyond the parameters.",
    syntax: "loss.backward()  ·  optimizer.step()",
    example: `loss.backward()               # writes into p.grad
print(model.fc1.weight.grad.shape)
optimizer.step()              # reads p.grad, updates p.data`,
    interviewNote:
      "Because the link is only through the parameter objects, the optimizer must be constructed with model.parameters() AFTER the model is on its final device. Recreating the model without recreating the optimizer is a classic silent bug.",
    importance: "high",
    tags: ["backward", "step", "optimizer", "gradient", "update"],
    relatedExercises: ["ex-train-4", "ex-train-5"],
  },
  {
    id: "cs-epoch-loop",
    topic: "training",
    section: "The Training Loop",
    title: "Full epoch loop with validation",
    description:
      "The shape of a real training script: train an epoch, validate under no_grad, track the best checkpoint.",
    syntax: "for epoch in range(epochs): train(); validate()",
    example: `best = float("inf")
for epoch in range(epochs):
    model.train()
    running = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        running += loss.item() * X.size(0)

    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(1) == y).sum().item()

    if running < best:
        best = running
        torch.save(model.state_dict(), "best.pth")`,
    interviewNote:
      "Note `loss.item() * X.size(0)`: losses are batch means, so weighting by batch size makes the epoch average correct even when the last batch is smaller.",
    importance: "high",
    tags: ["epoch", "validation", "checkpoint", "accuracy", "running loss"],
    relatedExercises: ["ex-train-6", "ex-train-10"],
  },
  {
    id: "cs-train-eval",
    topic: "evaluation",
    section: "Evaluation",
    title: "model.train() vs model.eval()",
    description:
      "A flag that switches the BEHAVIOUR of dropout and batch norm. It has nothing to do with gradients.",
    syntax: "model.train()  ·  model.eval()  ·  model.training",
    example: `model.train()   # dropout active, BatchNorm uses batch stats
model.eval()    # dropout off, BatchNorm uses running stats
print(model.training)`,
    result: "False",
    interviewNote:
      "🔥 Forgetting model.eval() gives noisy, worse validation numbers that change between runs. Forgetting to switch back to model.train() after validating silently disables dropout for the rest of training.",
    importance: "high",
    tags: ["train", "eval", "dropout", "batchnorm", "mode"],
    relatedExercises: ["ex-eval-1", "ex-eval-3"],
  },
  {
    id: "cs-eval-loop",
    topic: "evaluation",
    section: "Evaluation",
    title: "The evaluation loop",
    description:
      "Two independent things: eval() changes layer behaviour, no_grad() stops graph construction. You want BOTH.",
    syntax: "model.eval() + with torch.no_grad():",
    example: `model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        correct += (pred.argmax(dim=1) == y).sum().item()
        total += y.size(0)
print(correct / total)`,
    interviewNote:
      "🔥 The most common conceptual confusion in PyTorch interviews. eval() ≠ no_grad(). eval() without no_grad() is correct but wastes memory; no_grad() without eval() gives you dropout during validation.",
    importance: "high",
    tags: ["eval", "no_grad", "accuracy", "validation", "inference"],
    relatedExercises: ["ex-eval-2", "ex-eval-4", "ex-eval-5"],
  },
  {
    id: "cs-inference-mode",
    topic: "evaluation",
    section: "Evaluation",
    title: "torch.inference_mode()",
    description:
      "A stricter, faster no_grad(): it also skips version-counter bookkeeping. The recommended default for pure serving code.",
    syntax: "with torch.inference_mode(): ...",
    example: `model.eval()
with torch.inference_mode():
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)`,
    interviewNote:
      "The one caveat: tensors produced inside inference_mode cannot later take part in autograd. If you need to reuse outputs for training (e.g. a teacher model), use no_grad() instead.",
    importance: "medium",
    tags: ["inference_mode", "no_grad", "serving", "performance"],
  },
];
