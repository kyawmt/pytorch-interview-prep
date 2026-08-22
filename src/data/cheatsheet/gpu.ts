import type { CheatSheetEntry } from "@/lib/types";

export const gpuEntries: CheatSheetEntry[] = [
  {
    id: "cs-device",
    topic: "gpu",
    section: "Devices",
    title: "Picking a device",
    description:
      "Write device-agnostic code once and it runs on CUDA, Apple Silicon (MPS) or CPU without edits.",
    syntax: 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    example: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Including Apple Silicon:
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)`,
    interviewNote:
      "Hardcoding .cuda() everywhere is the anti-pattern; a single `device` variable plus .to(device) is what reviewers expect to see.",
    importance: "high",
    tags: ["device", "cuda", "mps", "cpu", "is_available"],
    relatedExercises: ["ex-gpu-1", "ex-gpu-2"],
  },
  {
    id: "cs-to-device",
    topic: "gpu",
    section: "Devices",
    title: "model.to(device) vs tensor.to(device)",
    description:
      "Modules move IN PLACE; tensors do NOT. That asymmetry is the source of the most common PyTorch runtime error.",
    syntax: "model.to(device)  ·  X = X.to(device)",
    example: `model.to(device)              # in place — no reassignment needed
X = X.to(device)             # returns a copy — MUST reassign
y = y.to(device)

# Async copy from pinned memory:
X = X.to(device, non_blocking=True)`,
    interviewNote:
      "🔥 \"RuntimeError: Expected all tensors to be on the same device\" almost always means the model was moved but the batch was not — or that `X.to(device)` was written without assigning the result.",
    importance: "high",
    tags: ["to", "device", "in place", "mismatch", "RuntimeError"],
    relatedExercises: ["ex-gpu-3", "ex-gpu-4", "ex-gpu-5"],
  },
  {
    id: "cs-gpu-memory",
    topic: "gpu",
    section: "Devices",
    title: "GPU memory and OOM",
    description:
      "Activations dominate memory during training, and they scale with batch size. The usual first fix is a smaller batch plus gradient accumulation.",
    syntax: "torch.cuda.memory_allocated()  ·  torch.cuda.empty_cache()",
    example: `print(torch.cuda.memory_allocated() / 1e9, "GB")
print(torch.cuda.max_memory_allocated() / 1e9, "GB")
torch.cuda.empty_cache()   # returns cached blocks to the driver`,
    interviewNote:
      "CUDA OOM checklist: reduce batch size, use AMP, add gradient accumulation, call .item()/.detach() when logging, avoid keeping a list of loss tensors, and use gradient checkpointing for very deep models. empty_cache() does not fix a real leak.",
    importance: "medium",
    tags: ["oom", "memory", "cuda", "empty_cache", "debug"],
    relatedExercises: ["ex-gpu-6"],
  },
  {
    id: "cs-transfers",
    topic: "gpu",
    section: "Devices",
    title: "Avoiding CPU↔GPU stalls",
    description:
      "Every .item(), .cpu() or print() on a GPU tensor forces a synchronisation and stalls the pipeline. Keep them out of the inner loop.",
    syntax: "accumulate on device, transfer once",
    example: `# Slow: syncs every batch
# for X, y in loader: total += loss.item()

# Faster: keep it on the GPU, read once per epoch
total = torch.zeros((), device=device)
for X, y in loader:
    ...
    total += loss.detach()
print(total.item())`,
    interviewNote:
      "CUDA kernels are launched asynchronously; a .item() blocks until the queue drains. This is why naive per-batch logging can dominate the profile of a small model.",
    importance: "medium",
    tags: ["synchronisation", "item", "transfer", "performance", "async"],
  },
  {
    id: "cs-save",
    topic: "saving",
    section: "Checkpoints",
    title: "Saving a state_dict",
    description:
      "state_dict is an OrderedDict of parameter and buffer tensors. Saving it (rather than the model object) is the recommended approach.",
    syntax: 'torch.save(model.state_dict(), "model.pth")',
    example: `torch.save(model.state_dict(), "model.pth")

model = MyModel()                    # rebuild the architecture first
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()`,
    interviewNote:
      "🔥 torch.save(model) pickles the class path, so refactoring your module breaks every old checkpoint. A state_dict is just tensors — portable, inspectable and version-tolerant. Remember model.eval() after loading for inference.",
    importance: "high",
    tags: ["save", "load", "state_dict", "checkpoint", "pickle"],
    relatedExercises: ["ex-save-1", "ex-save-2", "ex-save-3"],
  },
  {
    id: "cs-checkpoint",
    topic: "saving",
    section: "Checkpoints",
    title: "Resumable checkpoints",
    description:
      "To resume training you need more than the weights: the optimizer state (Adam's moments), the epoch and the scheduler.",
    syntax: "torch.save({...}, path)  ·  load_state_dict on each component",
    example: `torch.save({
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "best_loss": best_loss,
}, "checkpoint.pth")

ckpt = torch.load("checkpoint.pth", map_location=device)
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
start_epoch = ckpt["epoch"] + 1`,
    interviewNote:
      "Dropping the optimizer state restarts Adam's moment estimates from zero, which causes a visible loss spike when you resume. map_location matters when a GPU checkpoint is loaded on a CPU-only machine.",
    importance: "medium",
    tags: ["checkpoint", "resume", "optimizer state", "map_location"],
    relatedExercises: ["ex-save-4"],
  },
  {
    id: "cs-params-vs-statedict",
    topic: "saving",
    section: "Checkpoints",
    title: "parameters() vs state_dict()",
    description:
      "parameters() yields only learnable tensors (what the optimizer needs). state_dict() also includes buffers such as BatchNorm running_mean/var (what a checkpoint needs).",
    syntax: "model.parameters()  ·  model.state_dict()  ·  model.buffers()",
    example: `n_params = sum(p.numel() for p in model.parameters())
keys = list(model.state_dict().keys())
print(n_params, keys[:3])`,
    interviewNote:
      "Saving only parameters loses BatchNorm running statistics, so the restored model produces different predictions in eval mode. That is the point of the distinction.",
    importance: "medium",
    tags: ["parameters", "state_dict", "buffers", "batchnorm", "difference"],
    relatedExercises: ["ex-save-5"],
  },
  {
    id: "cs-amp",
    topic: "performance",
    section: "Mixed Precision",
    title: "Automatic Mixed Precision (AMP)",
    description:
      "Runs most ops in 16-bit while keeping a float32 copy of the weights. Roughly halves activation memory and uses tensor cores for a large speedup.",
    syntax: "with torch.autocast(device_type='cuda', dtype=torch.bfloat16): ...",
    example: `scaler = torch.amp.GradScaler("cuda")

for X, y in loader:
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = loss_fn(model(X), y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()`,
    interviewNote:
      "float16 needs a GradScaler because small gradients underflow to zero; bfloat16 has float32's exponent range and usually needs no scaler. Autocast wraps only the FORWARD pass and the loss — never the backward call or the optimizer step.",
    importance: "medium",
    tags: ["amp", "autocast", "GradScaler", "float16", "bfloat16", "speed"],
    relatedExercises: ["ex-perf-2", "ex-perf-3"],
  },
  {
    id: "cs-perf-checklist",
    topic: "performance",
    section: "Performance",
    title: "Throughput checklist",
    description:
      "The ordered list of things to try when training is slower than it should be.",
    syntax: "profile -> batch -> AMP -> data pipeline -> compile",
    example: `# 1. Vectorise: replace Python loops over samples with batched ops
# 2. Raise batch size until memory is the limit
# 3. Enable AMP (autocast + GradScaler)
# 4. num_workers > 0, pin_memory=True, persistent_workers=True
# 5. Remove per-batch .item() / print() synchronisation
# 6. torch.backends.cudnn.benchmark = True   (fixed input sizes)
# 7. model = torch.compile(model)            (PyTorch 2.x)`,
    interviewNote:
      "Always establish whether you are GPU-bound or data-bound first. If GPU utilisation sits at 30%, the fix is in the DataLoader, not in the model.",
    importance: "medium",
    tags: ["performance", "throughput", "compile", "cudnn", "profiling"],
    relatedExercises: ["ex-perf-4"],
  },
  {
    id: "cs-compile",
    topic: "performance",
    section: "Performance",
    title: "torch.compile()",
    description:
      "PyTorch 2.x traces your model into an optimised graph and fuses kernels. Usually a one-line change.",
    syntax: "model = torch.compile(model)",
    example: `model = torch.compile(model)   # first batch is slow (compilation)`,
    interviewNote:
      "Expect a warm-up cost and recompilation whenever input shapes change — pad to fixed shapes or use dynamic=True. It composes with AMP and DDP.",
    importance: "low",
    tags: ["compile", "pytorch 2", "fusion", "speed"],
  },
];
