import type { CheatSheetEntry } from "@/lib/types";

export const autogradEntries: CheatSheetEntry[] = [
  {
    id: "cs-requires-grad",
    topic: "autograd",
    section: "Autograd Basics",
    title: "requires_grad=True",
    description:
      "Marks a tensor as something to differentiate with respect to. Every operation on it is then recorded in a computational graph.",
    syntax: "torch.tensor(2.0, requires_grad=True)  ·  x.requires_grad_(True)",
    example: `x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
print(y.requires_grad, y.grad_fn)`,
    result: "True <PowBackward0 object at 0x...>",
    interviewNote:
      "Model parameters get requires_grad=True automatically (nn.Parameter). Input data normally does not — you only set it on inputs for adversarial examples or input optimisation.",
    importance: "high",
    tags: ["requires_grad", "autograd", "graph", "grad_fn"],
    relatedExercises: ["ex-autograd-1", "ex-autograd-2"],
  },
  {
    id: "cs-backward",
    topic: "autograd",
    section: "Autograd Basics",
    title: "loss.backward() and .grad",
    description:
      "backward() walks the graph backwards applying the chain rule and ACCUMULATES the result into the .grad attribute of every leaf tensor with requires_grad=True.",
    syntax: "loss.backward()  ·  x.grad",
    example: `x = torch.tensor(2.0, requires_grad=True)
y = x ** 2          # dy/dx = 2x
y.backward()
print(x.grad)`,
    result: "tensor(4.)",
    interviewNote:
      "Why 4? y = x², so dy/dx = 2x, and at x = 2 that is 4. backward() must be called on a scalar; for a non-scalar you have to pass a gradient argument, which is why you almost always reduce with .mean().",
    importance: "high",
    tags: ["backward", "grad", "chain rule", "gradient"],
    relatedExercises: ["ex-autograd-3", "ex-autograd-4", "ex-autograd-5"],
  },
  {
    id: "cs-accumulate",
    topic: "autograd",
    section: "Autograd Basics",
    title: "Why gradients accumulate",
    description:
      "backward() adds to .grad instead of replacing it. Without an explicit zero_grad() your gradients are the sum of every step so far.",
    syntax: "optimizer.zero_grad()  ·  model.zero_grad()  ·  optimizer.zero_grad(set_to_none=True)",
    example: `x = torch.tensor(2.0, requires_grad=True)
(x ** 2).backward()
print(x.grad)          # 4
(x ** 2).backward()
print(x.grad)          # 8 — accumulated, not reset`,
    result: `tensor(4.)
tensor(8.)`,
    interviewNote:
      "Accumulation is a feature: it lets you split a large batch across several forward passes (gradient accumulation) and sum gradients from multiple losses. The cost is that you must clear them yourself every step.",
    importance: "high",
    tags: ["zero_grad", "accumulate", "gradient accumulation", "training loop"],
    relatedExercises: ["ex-autograd-6", "ex-train-2"],
  },
  {
    id: "cs-no-grad",
    topic: "autograd",
    section: "Turning Autograd Off",
    title: "torch.no_grad()",
    description:
      "A context manager that stops the graph from being recorded. Operations still run, they just cannot be differentiated.",
    syntax: "with torch.no_grad(): ...  ·  @torch.no_grad()  ·  torch.inference_mode()",
    example: `model.eval()
with torch.no_grad():
    for X, y in val_loader:
        pred = model(X)`,
    interviewNote:
      "Saves memory and time during validation and inference, because activations no longer need to be kept for a backward pass. torch.inference_mode() is the newer, slightly faster version — but tensors created inside it cannot later be used in autograd.",
    importance: "high",
    tags: ["no_grad", "inference_mode", "eval", "memory", "validation"],
    relatedExercises: ["ex-autograd-7", "ex-eval-2"],
  },
  {
    id: "cs-detach",
    topic: "autograd",
    section: "Turning Autograd Off",
    title: ".detach()",
    description:
      "Returns a tensor that shares the same data but is cut out of the graph. Gradients stop flowing at that point.",
    syntax: "x.detach()  ·  x.detach().cpu().numpy()",
    example: `loss_value = loss.detach()          # keep for logging, drop the graph
arr = pred.detach().cpu().numpy()   # hand off to NumPy/sklearn`,
    interviewNote:
      "detach() acts on one tensor; no_grad() acts on a whole block. Classic uses: logging metrics, computing targets in a target network, and truncated BPTT where you detach the hidden state between chunks.",
    importance: "high",
    tags: ["detach", "graph", "numpy", "logging", "memory leak"],
    relatedExercises: ["ex-autograd-8", "ex-autograd-9"],
  },
  {
    id: "cs-graph",
    topic: "autograd",
    section: "Autograd Basics",
    title: "The computational graph",
    description:
      "PyTorch builds a dynamic DAG of operations during the forward pass: leaves are inputs and parameters, edges are the backward functions. backward() traverses it and then frees it.",
    syntax: "y.grad_fn  ·  loss.backward(retain_graph=True)",
    example: `x = torch.randn(3, requires_grad=True)
y = (x * 2).sum()
print(y.grad_fn)              # SumBackward0
y.backward()
# y.backward()                # RuntimeError: graph freed`,
    interviewNote:
      "\"Define-by-run\": the graph is rebuilt on every forward pass, which is why plain Python control flow works inside forward(). Calling backward twice needs retain_graph=True — but usually that error means you accidentally reused a graph across steps.",
    importance: "high",
    tags: ["graph", "dynamic", "define-by-run", "retain_graph", "grad_fn"],
    relatedExercises: ["ex-autograd-10"],
  },
  {
    id: "cs-grad-leaf",
    topic: "autograd",
    section: "Autograd Basics",
    title: "Where gradients actually live",
    description:
      "Only leaf tensors with requires_grad=True get a populated .grad. Intermediate results return None unless you call retain_grad().",
    syntax: "x.is_leaf  ·  x.grad  ·  y.retain_grad()",
    example: `x = torch.tensor(3.0, requires_grad=True)
y = x * 2
z = y ** 2
z.backward()
print(x.grad, y.grad)   # y.grad is None`,
    result: "tensor(24.) None",
    interviewNote:
      "In a model, the leaves are the parameters — so after backward() the gradients sit in p.grad for each p in model.parameters(), which is exactly what the optimizer reads.",
    importance: "medium",
    tags: ["leaf", "grad", "retain_grad", "parameters"],
  },
  {
    id: "cs-grad-clip",
    topic: "autograd",
    section: "Gradient Management",
    title: "Gradient clipping",
    description:
      "Rescales gradients whose global norm exceeds a threshold. Call it after backward() and before step().",
    syntax: "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
    example: `loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()`,
    interviewNote:
      "Standard for RNNs and transformers, where a single bad batch can produce an enormous gradient. Clipping preserves direction and only shrinks magnitude; clip_grad_value_ clips per element and distorts the direction.",
    importance: "medium",
    tags: ["clipping", "exploding gradients", "rnn", "transformer", "stability"],
    relatedExercises: ["ex-autograd-11"],
  },
  {
    id: "cs-grad-accum",
    topic: "autograd",
    section: "Gradient Management",
    title: "Gradient accumulation",
    description:
      "Simulate a larger batch on limited memory: run several small batches, scale each loss, and only step once.",
    syntax: "loss = loss / accum_steps  ·  step every accum_steps batches",
    example: `accum_steps = 4
optimizer.zero_grad()
for i, (X, y) in enumerate(loader):
    loss = loss_fn(model(X), y) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()`,
    interviewNote:
      "Dividing by accum_steps keeps the gradient magnitude equal to a true large batch. Forgetting the division effectively multiplies your learning rate by accum_steps.",
    importance: "medium",
    tags: ["gradient accumulation", "batch size", "memory", "oom"],
    relatedExercises: ["ex-perf-1"],
  },
];
