import type { Exercise } from "@/lib/types";

/**
 * Training Loop Builder: ordering drills where some of the offered blocks are
 * deliberately wrong, so the learner has to reject them rather than just sort.
 */
export const BUILDER_DRILLS: Exercise[] = [
  {
    id: "bd-basic",
    title: "The basic training step",
    topic: "training",
    level: 5,
    difficulty: "easy",
    importance: "high",
    type: "ordering",
    question:
      "Build one training step. Three of these blocks do not belong in a training loop — leave them out.",
    blocks: [
      "optimizer.zero_grad()",
      "pred = model(X)",
      "loss = loss_fn(pred, y)",
      "loss.backward()",
      "optimizer.step()",
      "model.eval()",
      "with torch.no_grad():",
      "loss.detach().backward()",
    ],
    correctOrder: [0, 1, 2, 3, 4],
    hint: "Clear the gradients, predict, measure, backpropagate, update.",
    solution: `optimizer.zero_grad()
pred = model(X)
loss = loss_fn(pred, y)
loss.backward()
optimizer.step()`,
    explanation:
      "model.eval() and no_grad() belong to validation. Detaching before backward would cut the graph, so no gradients would reach the parameters at all.",
    interviewNote: "🔥 Be able to produce these five lines instantly, in this order.",
    tags: ["training loop", "ordering"],
  },
  {
    id: "bd-full",
    title: "A full epoch with device handling",
    topic: "training",
    level: 5,
    difficulty: "medium",
    importance: "high",
    type: "ordering",
    question: "Build a complete training epoch, including the mode switch, device move and logging.",
    blocks: [
      "model.train()",
      "for X, y in train_loader:",
      "X, y = X.to(device), y.to(device)",
      "optimizer.zero_grad()",
      "pred = model(X)",
      "loss = loss_fn(pred, y)",
      "loss.backward()",
      "optimizer.step()",
      "running_loss += loss.item()",
      "torch.cuda.empty_cache()",
      "model.zero_grad(set_to_none=False)",
    ],
    correctOrder: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    hint: "train() → loop → move → zero_grad → forward → loss → backward → step → log.",
    solution: `model.train()
for X, y in train_loader:
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()`,
    explanation:
      "empty_cache() per batch only adds a synchronisation point and does not prevent OOM. A second zero_grad would be redundant. Note loss.item() rather than loss — accumulating the tensor leaks the graph.",
    tags: ["training loop", "device", "ordering"],
  },
  {
    id: "bd-eval",
    title: "The validation loop",
    topic: "evaluation",
    level: 5,
    difficulty: "medium",
    importance: "high",
    type: "ordering",
    question: "Build the evaluation loop that computes accuracy. Two blocks do not belong.",
    blocks: [
      "model.eval()",
      "with torch.no_grad():",
      "for X, y in val_loader:",
      "pred = model(X)",
      "correct += (pred.argmax(1) == y).sum().item()",
      "loss.backward()",
      "optimizer.step()",
    ],
    correctOrder: [0, 1, 2, 3, 4],
    hint: "Switch the mode, disable the graph, then just measure.",
    solution: `model.eval()
with torch.no_grad():
    for X, y in val_loader:
        pred = model(X)
        correct += (pred.argmax(1) == y).sum().item()`,
    explanation:
      "Nothing is optimised during validation, so backward() and step() must not appear. eval() and no_grad() do different jobs and you want both.",
    tags: ["evaluation", "no_grad", "ordering"],
  },
  {
    id: "bd-clip",
    title: "Training step with gradient clipping",
    topic: "training",
    level: 7,
    difficulty: "medium",
    importance: "medium",
    type: "ordering",
    question: "Where exactly does gradient clipping go? Build the step.",
    blocks: [
      "optimizer.zero_grad()",
      "loss = loss_fn(model(X), y)",
      "loss.backward()",
      "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)",
      "optimizer.step()",
      "scheduler.step()",
    ],
    correctOrder: [0, 1, 2, 3, 4],
    hint: "Clipping needs gradients to exist, but must happen before they are applied.",
    solution: `optimizer.zero_grad()
loss = loss_fn(model(X), y)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()`,
    explanation:
      "Clipping sits strictly between backward() and step(). A per-epoch scheduler.step() belongs outside the batch loop, so it is not part of this step.",
    tags: ["clipping", "ordering"],
  },
  {
    id: "bd-amp",
    title: "Mixed-precision training step",
    topic: "performance",
    level: 7,
    difficulty: "hard",
    importance: "medium",
    type: "ordering",
    question: "Build an AMP training step with a GradScaler.",
    blocks: [
      "optimizer.zero_grad()",
      'with torch.autocast(device_type="cuda", dtype=torch.float16):',
      "loss = loss_fn(model(X), y)",
      "scaler.scale(loss).backward()",
      "scaler.step(optimizer)",
      "scaler.update()",
      "loss.backward()",
    ],
    correctOrder: [0, 1, 2, 3, 4, 5],
    hint: "Autocast wraps only the forward pass and the loss.",
    solution: `optimizer.zero_grad()
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = loss_fn(model(X), y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()`,
    explanation:
      "The backward pass and the optimizer step happen OUTSIDE the autocast block, and go through the scaler so gradients are unscaled before the update. A plain loss.backward() would bypass the scaler entirely.",
    tags: ["amp", "GradScaler", "ordering"],
  },
  {
    id: "bd-accum",
    title: "Gradient accumulation step",
    topic: "performance",
    level: 7,
    difficulty: "hard",
    importance: "medium",
    type: "ordering",
    question:
      "Build the body of a gradient-accumulation loop (accum_steps micro-batches per update).",
    blocks: [
      "loss = loss_fn(model(X), y) / accum_steps",
      "loss.backward()",
      "if (i + 1) % accum_steps == 0:",
      "optimizer.step()",
      "optimizer.zero_grad()",
      "optimizer.zero_grad()  # every batch",
    ],
    correctOrder: [0, 1, 2, 3, 4],
    hint: "Every micro-batch backprops; only every Nth one steps and clears.",
    solution: `loss = loss_fn(model(X), y) / accum_steps
loss.backward()
if (i + 1) % accum_steps == 0:
    optimizer.step()
    optimizer.zero_grad()`,
    explanation:
      "Clearing gradients every batch would defeat the whole point — accumulation relies on them summing. Dividing the loss keeps the gradient magnitude equal to one true large batch.",
    tags: ["accumulation", "ordering"],
  },
];
