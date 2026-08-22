import type { InterviewQuestion } from "@/lib/types";

export const trainingQuestions: InterviewQuestion[] = [
  {
    id: "iq-train-1",
    category: "training loops",
    topic: "training",
    question: "Write and explain the standard PyTorch training loop.",
    shortAnswer:
      "Set model.train(), then for each batch: zero_grad to clear stale gradients, forward to get predictions, compute the loss, backward to fill .grad, and step to update the weights.",
    detailedAnswer:
      "The five lines map onto the maths directly: zero_grad resets the accumulator, the forward pass builds the graph, the loss reduces everything to a scalar, backward applies the chain rule to fill p.grad for every parameter, and step applies the update rule using those gradients. Everything else — device moves, AMP, clipping, schedulers, logging — hangs off this skeleton. Be ready to write it without a starter and to explain what breaks if any line is missing or reordered.",
    code: `model.train()
for X, y in train_loader:
    X, y = X.to(device), y.to(device)

    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()`,
    importance: "high",
    followUps: ["What happens if zero_grad is missing?", "Where would gradient clipping go?"],
    tags: ["training loop", "memorise"],
  },
  {
    id: "iq-train-2",
    category: "training loops",
    topic: "training",
    question: "What happens if you forget optimizer.zero_grad()?",
    shortAnswer:
      "Gradients from previous batches keep accumulating, so each update uses the running sum. The effective step size grows every iteration and training usually destabilises or diverges.",
    detailedAnswer:
      "It is a silent bug — nothing raises. The loss typically looks fine for a few steps, then climbs or turns into NaN. If you ever DO want accumulation (to simulate a bigger batch), do it deliberately: divide the loss by the number of micro-batches and call zero_grad only every N steps. A quick diagnostic is to print the gradient norm each step and watch it grow monotonically.",
    code: `for i, (X, y) in enumerate(loader):
    loss = loss_fn(model(X), y) / accum
    loss.backward()
    if (i + 1) % accum == 0:
        optimizer.step()
        optimizer.zero_grad()`,
    importance: "high",
    followUps: ["How would you detect this from the logs?", "When is accumulation desirable?"],
    tags: ["zero_grad", "bug", "accumulation"],
  },
  {
    id: "iq-train-3",
    category: "training loops",
    topic: "evaluation",
    question: "What is the difference between model.eval() and torch.no_grad()?",
    shortAnswer:
      "model.eval() switches layer behaviour — dropout off, BatchNorm on running statistics. torch.no_grad() stops autograd from recording the graph. They are independent and you normally want both.",
    detailedAnswer:
      "eval() without no_grad() gives correct predictions but wastes memory storing activations for a backward pass that never happens. no_grad() without eval() is worse: dropout stays active and BatchNorm keeps updating its running statistics from validation data, so your metrics are noisy and your model is quietly contaminated. Remember eval() is sticky — call model.train() again at the start of the next training epoch.",
    code: `model.eval()
with torch.no_grad():
    for X, y in val_loader:
        pred = model(X)`,
    importance: "high",
    followUps: ["What breaks if you only call one of them?", "What is inference_mode()?"],
    tags: ["eval", "no_grad", "validation"],
  },
  {
    id: "iq-train-4",
    category: "losses",
    topic: "losses",
    question: "Why shouldn't you apply softmax before nn.CrossEntropyLoss?",
    shortAnswer:
      "CrossEntropyLoss already applies log_softmax internally. Softmaxing first applies it twice, which flattens the distribution, weakens the gradients and loses the numerical stability of the fused version.",
    detailedAnswer:
      "nn.CrossEntropyLoss = log_softmax + NLLLoss. The fused implementation uses the log-sum-exp trick, so it stays stable even for large logits. Passing already-normalised probabilities does not raise an error — the model just trains measurably worse, which makes it a nasty silent bug. Apply softmax only at inference when you want to report probabilities; for the predicted class you do not even need it, because softmax is monotonic and argmax is unchanged.",
    code: `logits = model(X)              # raw scores
loss = loss_fn(logits, y)      # NOT softmax(logits)
probs = torch.softmax(logits, dim=1)   # inference only`,
    importance: "high",
    followUps: ["What is the equivalent trap for binary classification?", "What does log-sum-exp buy you?"],
    tags: ["crossentropy", "softmax", "logits"],
  },
  {
    id: "iq-train-5",
    category: "losses",
    topic: "losses",
    question: "Why use BCEWithLogitsLoss instead of Sigmoid + BCELoss?",
    shortAnswer:
      "It fuses the sigmoid into the loss and uses the log-sum-exp trick, so it stays numerically stable when logits are large. Separate sigmoid + BCELoss can produce inf or NaN.",
    detailedAnswer:
      "BCELoss takes log(p) and log(1-p). If sigmoid saturates to exactly 0.0 or 1.0 in float32, those logs become -inf and the gradient becomes NaN. The fused version never materialises the probability, computing everything from the logit in a stable form. It also supports pos_weight for class imbalance. Remember the dtype asymmetry: BCEWithLogitsLoss wants FLOAT targets shaped like the logits, while CrossEntropyLoss wants int64 indices of shape (B,).",
    code: `loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))
loss = loss_fn(logits, targets.float())`,
    importance: "high",
    followUps: ["How do you handle class imbalance here?", "When would you use plain BCELoss?"],
    tags: ["bce", "stability", "logits"],
  },
  {
    id: "iq-train-6",
    category: "losses",
    topic: "losses",
    question: "How do you choose a loss function for a new task?",
    shortAnswer:
      "Match it to the output: regression → MSE or L1/Huber; binary or multi-label → BCEWithLogitsLoss on logits; single-label multi-class → CrossEntropyLoss on logits. In all cases the final layer has no activation.",
    detailedAnswer:
      "MSE punishes outliers quadratically; L1 and SmoothL1 are more robust when the target has heavy tails. Multi-class means the classes are mutually exclusive, so softmax is appropriate; multi-label means they are independent, so each class gets its own sigmoid. For imbalanced data, add class weights (CrossEntropyLoss(weight=...)) or pos_weight for BCE, and consider focal loss for extreme imbalance. Label smoothing is a cheap regulariser for large-scale classification.",
    code: `nn.MSELoss()                # regression
nn.BCEWithLogitsLoss()      # binary / multi-label
nn.CrossEntropyLoss(label_smoothing=0.1)   # multi-class`,
    importance: "medium",
    followUps: ["How do you handle a 1:100 class imbalance?", "What does label smoothing do?"],
    tags: ["loss selection", "imbalance"],
  },
  {
    id: "iq-opt-1",
    category: "optimization",
    topic: "optimizers",
    question: "Adam vs SGD — which do you pick and why?",
    shortAnswer:
      "Adam converges faster and needs less tuning, so it is the default for transformers and most new problems. SGD with momentum plus a good schedule often generalises slightly better and is still standard for large-scale vision.",
    detailedAnswer:
      "Adam keeps per-parameter running averages of the gradient and its square, giving each weight an adaptive step size — very helpful when gradient magnitudes vary wildly across layers, as in transformers. SGD's uniform step size makes it more sensitive to the learning rate but the resulting solutions are often flatter and generalise a little better; ResNet-style vision recipes still use SGD + momentum + cosine decay. In practice: prototype with AdamW at 1e-3 (or 3e-4 for transformers), and only move to SGD if you are chasing the last point of accuracy.",
    code: `torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)`,
    importance: "high",
    followUps: ["What does momentum do?", "What learning rate would you start with?"],
    tags: ["adam", "sgd", "momentum"],
  },
  {
    id: "iq-opt-2",
    category: "optimization",
    topic: "optimizers",
    question: "Why AdamW instead of Adam?",
    shortAnswer:
      "AdamW decouples weight decay from the gradient. Adam folds L2 regularisation into the gradient, where the adaptive scaling distorts it; AdamW subtracts the decay directly from the weights so it regularises as intended.",
    detailedAnswer:
      "In Adam, the L2 term is divided by the same per-parameter denominator as the gradient, so parameters with large historical gradients receive proportionally less decay — the opposite of what you want. Loshchilov & Hutter showed decoupling recovers the generalisation benefit and makes weight decay tunable independently of the learning rate. Every modern transformer recipe uses AdamW, typically excluding biases and normalisation parameters from decay.",
    code: `decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
torch.optim.AdamW([
    {"params": decay, "weight_decay": 0.01},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4)`,
    importance: "high",
    followUps: ["Why exclude biases and LayerNorm from decay?", "What does weight decay actually do?"],
    tags: ["adamw", "weight decay", "decoupled"],
  },
  {
    id: "iq-opt-3",
    category: "optimization",
    topic: "optimizers",
    question: "What does the learning rate control, and how do you choose one?",
    shortAnswer:
      "It scales the size of each parameter update. Too high and the loss oscillates or diverges; too low and training crawls or gets stuck. Start from known defaults and use a short LR range test.",
    detailedAnswer:
      "Practical defaults: 1e-3 for Adam on small models, 3e-4 or lower for transformers, 0.1 for SGD+momentum on vision with a cosine or step schedule. An LR range test — increase the LR exponentially over a few hundred steps and plot the loss — gives a good bracket in minutes. Warmup for the first few hundred steps helps transformers, where early updates are large and unstable. Effective LR also interacts with batch size: the linear scaling rule says doubling the batch roughly allows doubling the LR.",
    code: `scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=epochs * len(loader))`,
    importance: "high",
    followUps: ["What is LR warmup for?", "How does batch size interact with LR?"],
    tags: ["learning rate", "schedule", "warmup"],
  },
  {
    id: "iq-opt-4",
    category: "optimization",
    topic: "optimizers",
    question: "What does weight decay do?",
    shortAnswer:
      "It shrinks weights toward zero at each step, penalising large weights and biasing the model toward simpler functions — a regulariser against overfitting.",
    detailedAnswer:
      "Mathematically it is L2 regularisation (in SGD they are identical; in Adam they are not, hence AdamW). Typical values are 1e-4 for vision with SGD and 0.01 for transformers with AdamW. Biases and normalisation scales are normally excluded — a LayerNorm gain pulled toward zero actively degrades the model, and biases have no capacity effect worth regularising.",
    code: `torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)`,
    importance: "medium",
    followUps: ["Is weight decay the same as L2 in Adam?", "Which parameters do you exclude?"],
    tags: ["weight decay", "regularisation"],
  },
  {
    id: "iq-opt-5",
    category: "optimization",
    topic: "optimizers",
    question: "What is a learning-rate scheduler and when would you use one?",
    shortAnswer:
      "It changes the learning rate over training — usually a high rate early for fast progress and a low rate late for fine convergence. Cosine decay and step decay are the common choices.",
    detailedAnswer:
      "StepLR multiplies by gamma every N epochs; CosineAnnealingLR decays smoothly to near zero; OneCycleLR warms up then anneals and works very well for short schedules; ReduceLROnPlateau reacts to a validation metric and is useful when you cannot predict the schedule length. Call scheduler.step() after optimizer.step() — per epoch for most schedulers, per batch for warmup/OneCycle, and with the metric for ReduceLROnPlateau.",
    code: `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
for epoch in range(epochs):
    train(); validate()
    scheduler.step()`,
    importance: "medium",
    followUps: ["Which scheduler for a fixed 10-epoch budget?", "Why does warmup help transformers?"],
    tags: ["scheduler", "cosine", "onecycle"],
  },
  {
    id: "iq-data-1",
    category: "practical ML",
    topic: "data",
    question: "Walk me through writing a custom Dataset and DataLoader.",
    shortAnswer:
      "Subclass Dataset with __len__ and __getitem__ returning one (input, target) pair, then wrap it in a DataLoader with a batch size, shuffling and workers.",
    detailedAnswer:
      "__getitem__ handles a SINGLE sample; the DataLoader's sampler generates indices, workers call __getitem__ in parallel, and collate_fn stacks the results into a batch. Put per-sample work (file reads, decoding, augmentation) inside __getitem__ so it parallelises across workers. Use num_workers > 0 and pin_memory=True to overlap loading with GPU compute, shuffle=True only for training, and a custom collate_fn when samples have different shapes.",
    code: `class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

loader = DataLoader(ds, batch_size=32, shuffle=True,
                    num_workers=4, pin_memory=True)`,
    importance: "high",
    followUps: ["How do you batch variable-length sequences?", "What does num_workers actually do?"],
    tags: ["Dataset", "DataLoader", "collate"],
  },
  {
    id: "iq-data-2",
    category: "practical ML",
    topic: "data",
    question: "Why do we train on mini-batches?",
    shortAnswer:
      "They balance three things: full-batch gradients are accurate but memory-bound and give one update per epoch; single samples are noisy and underuse the GPU. Mini-batches give frequent, reasonably accurate updates with good hardware utilisation — and the noise acts as a regulariser.",
    detailedAnswer:
      "GPUs are throughput devices: a batch of 64 costs little more than a batch of 1 in wall-clock time, so batching is nearly free parallelism. Statistically, the gradient noise from sampling helps escape sharp minima and is associated with better generalisation, which is why very large batches sometimes need a warmup or a scaled learning rate to match small-batch accuracy. Batch size also interacts with BatchNorm, which needs enough samples for a meaningful variance estimate.",
    code: `DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)`,
    importance: "high",
    followUps: ["How do batch size and learning rate interact?", "What if the batch does not fit in memory?"],
    tags: ["mini-batch", "sgd", "generalisation"],
  },
  {
    id: "iq-data-3",
    category: "practical ML",
    topic: "data",
    question: "What do num_workers and pin_memory do?",
    shortAnswer:
      "num_workers spawns subprocesses that prefetch and preprocess batches in parallel with GPU compute. pin_memory allocates page-locked host memory so host→device copies are faster and can be asynchronous.",
    detailedAnswer:
      "With num_workers=0 everything happens in the main process, so the GPU sits idle while the next batch is decoded. A good starting point is 4-8 workers, tuned by watching GPU utilisation. pin_memory=True combined with .to(device, non_blocking=True) lets the copy overlap with computation. persistent_workers=True avoids respawning workers every epoch. Too many workers can exhaust RAM or thrash the CPU, so it is worth measuring.",
    code: `DataLoader(ds, batch_size=64, num_workers=8,
           pin_memory=True, persistent_workers=True)
X = X.to(device, non_blocking=True)`,
    importance: "medium",
    followUps: ["How do you tell if you are data-bound?", "What can go wrong with many workers?"],
    tags: ["num_workers", "pin_memory", "throughput"],
  },
  {
    id: "iq-gpu-1",
    category: "GPU",
    topic: "gpu",
    question: "What happens if the model is on the GPU but the input tensors are on the CPU?",
    shortAnswer:
      "PyTorch raises 'Expected all tensors to be on the same device'. The fix is to move the batch as well — remembering that tensor .to() returns a new tensor and must be reassigned.",
    detailedAnswer:
      "The asymmetry catches everyone: nn.Module.to() moves parameters in place, while Tensor.to() is out-of-place. So `model.to(device)` works but `X.to(device)` on its own does nothing. Also watch for tensors created inside forward() — build them with torch.zeros_like(x) or pass device=x.device rather than relying on the default device. In multi-GPU code the same error appears as cuda:0 vs cuda:1.",
    code: `model.to(device)                      # in place
X, y = X.to(device), y.to(device)     # must reassign`,
    importance: "high",
    followUps: ["Why is model.to() in place but tensor.to() not?", "How do you create a tensor on the same device as x?"],
    tags: ["device", "mismatch", "to"],
  },
  {
    id: "iq-gpu-2",
    category: "GPU",
    topic: "gpu",
    question: "How do you write device-agnostic PyTorch code?",
    shortAnswer:
      "Resolve a single `device` at the top from torch.cuda.is_available() (and torch.backends.mps.is_available() on Apple Silicon), then send the model and every batch to it with .to(device).",
    detailedAnswer:
      "Never hardcode .cuda(). Create tensors with device=... or the *_like constructors so they land in the right place, and pass map_location=device when loading checkpoints so a GPU checkpoint can be restored on a CPU machine. This makes the same script run on a laptop, a CI runner and an A100 without edits.",
    code: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
state = torch.load("model.pth", map_location=device)`,
    importance: "high",
    followUps: ["What is MPS?", "Why does map_location matter?"],
    tags: ["device", "portability", "cuda", "mps"],
  },
  {
    id: "iq-save-1",
    category: "practical ML",
    topic: "saving",
    question: "How do you save and load a model, and why prefer state_dict?",
    shortAnswer:
      "Save model.state_dict() and load it into a freshly constructed model. Saving the whole object pickles the class path, so any refactor breaks old checkpoints.",
    detailedAnswer:
      "A state_dict is just an OrderedDict of named tensors — portable, inspectable, and independent of your file layout. torch.save(model) uses pickle and stores a reference to the module path, so moving or renaming the class makes the checkpoint unloadable. For resuming training, save the optimizer state (Adam's moments), the scheduler state and the epoch too; otherwise the loss spikes on resume. Call model.eval() after loading for inference.",
    code: `torch.save({"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch}, "ckpt.pth")

ckpt = torch.load("ckpt.pth", map_location=device)
model.load_state_dict(ckpt["model"])`,
    importance: "high",
    followUps: ["What else goes in a resumable checkpoint?", "What does strict=False do in load_state_dict?"],
    tags: ["save", "state_dict", "checkpoint"],
  },
];
