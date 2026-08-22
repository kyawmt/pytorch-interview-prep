import type { MiniProject } from "@/lib/types";

export const fullPipeline: MiniProject = {
  slug: "full-pipeline",
  title: "Complete Training Pipeline",
  tagline: "Dataset → DataLoader → Model → Loss → Optimizer → Train → Validate → Save. From memory.",
  difficulty: "hard",
  estimatedMinutes: 40,
  topics: ["data", "nn", "losses", "optimizers", "training", "evaluation", "gpu", "saving"],
  outline: [
    "Dataset",
    "DataLoader",
    "Device",
    "Model",
    "Loss + Optimizer",
    "Training loop",
    "Validation",
    "Checkpoint",
  ],
  overview:
    "The final project: reconstruct an entire training script without a starter. This is deliberately close to what a live-coding interview asks for.",
  steps: [
    {
      id: "fp-1",
      title: "Custom Dataset",
      brief: "Write a Dataset class wrapping tensors X and y.",
      kind: "code",
      requiredPatterns: [
        "class\\s\\w+\\(Dataset\\)",
        "def\\s__init__\\(self",
        "def\\s__len__\\(self\\)",
        "def\\s__getitem__\\(self,\\w+\\)",
      ],
      hint: "__init__, __len__, __getitem__.",
      solution: `class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]`,
      explanation: "__getitem__ returns ONE sample; the DataLoader batches them.",
    },
    {
      id: "fp-2",
      title: "DataLoaders",
      brief:
        "Create a training loader (batch 32, shuffled) and a validation loader (batch 64, not shuffled).",
      kind: "code",
      requiredPatterns: ["DataLoader", "batch_size=32", "shuffle=True", "batch_size=64"],
      hint: "Two DataLoader calls; only training shuffles.",
      solution: `train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)`,
      explanation:
        "Validation can use a larger batch because no activations are stored for backward. Shuffling it would change nothing about the metric, but keeping it fixed makes per-batch logs reproducible.",
    },
    {
      id: "fp-3",
      title: "Device and model",
      brief: "Resolve the device and move the model onto it (two lines).",
      kind: "code",
      requiredPatterns: ["torch\\.cuda\\.is_available\\(\\)", "\\.to\\(device\\)"],
      hint: "One device variable, then model.to(device).",
      solution: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)`,
      explanation:
        "Move the model BEFORE constructing the optimizer, so the optimizer's state is created for the parameters in their final location.",
    },
    {
      id: "fp-4",
      title: "Loss and optimizer",
      brief: "Create a CrossEntropyLoss and an AdamW optimizer with lr=1e-3 and weight_decay=0.01.",
      kind: "code",
      requiredPatterns: ["CrossEntropyLoss\\(\\)", "AdamW", "model\\.parameters\\(\\)", "weight_decay"],
      hint: "AdamW takes weight_decay as a keyword argument.",
      solution: `loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)`,
      explanation: "AdamW decouples weight decay from the adaptive scaling, which is why it is preferred over Adam.",
    },
    {
      id: "fp-5",
      title: "Training epoch",
      brief:
        "Write a training epoch that moves batches to the device, runs the optimisation step and accumulates the loss.",
      kind: "code",
      requiredPatterns: [
        "model\\.train\\(\\)",
        "\\.to\\(device\\)",
        "zero_grad\\(\\)",
        "backward\\(\\)",
        "\\.step\\(\\)",
        "item\\(\\)",
      ],
      hint: "train() → move → zero_grad → forward → loss → backward → step → log with .item().",
      solution: `model.train()
running = 0.0
for X, y in train_loader:
    X, y = X.to(device), y.to(device)

    optimizer.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()

    running += loss.item() * X.size(0)
train_loss = running / len(train_loader.dataset)`,
      explanation:
        "Using .item() is what keeps this loop from leaking memory: adding the loss tensor itself would retain every batch's graph.",
    },
    {
      id: "fp-6",
      title: "Validation epoch",
      brief: "Write the validation pass that computes accuracy.",
      kind: "code",
      requiredPatterns: [
        "model\\.eval\\(\\)",
        "torch\\.(no_grad|inference_mode)",
        "argmax",
        "==y",
      ],
      hint: "eval() + no_grad() + argmax comparison.",
      solution: `model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
val_acc = correct / total`,
      explanation: "Both eval() and no_grad() are required — one changes layer behaviour, the other stops graph construction.",
    },
    {
      id: "fp-7",
      title: "Save the best checkpoint",
      brief:
        "When val_acc beats `best_acc`, save a checkpoint containing the model AND optimizer state plus the epoch.",
      kind: "code",
      requiredPatterns: ["torch\\.save", "model\\.state_dict\\(\\)", "optimizer\\.state_dict\\(\\)"],
      hint: "torch.save of a dict.",
      solution: `if val_acc > best_acc:
    best_acc = val_acc
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_acc": val_acc,
    }, "best.pth")`,
      explanation:
        "Saving state_dicts (not the objects) keeps checkpoints loadable after refactors. Including the optimizer state avoids a loss spike when you resume.",
    },
    {
      id: "fp-8",
      title: "Reload for inference",
      brief: "Load `best.pth` into a fresh model and prepare it for inference (three lines).",
      kind: "code",
      requiredPatterns: ["torch\\.load", "load_state_dict", "model\\.eval\\(\\)"],
      hint: "Construct, load_state_dict, eval.",
      solution: `model = MyModel().to(device)
model.load_state_dict(torch.load("best.pth", map_location=device)["model"])
model.eval()`,
      explanation:
        "The architecture must be recreated first — a state_dict is only tensors. map_location lets a GPU checkpoint load on a CPU-only machine. Forgetting eval() leaves dropout active during inference.",
    },
    {
      id: "fp-9",
      title: "What is missing?",
      brief: "The pipeline works. Which addition would most improve a real run?",
      kind: "multiple-choice",
      options: [
        "Calling torch.cuda.empty_cache() every batch",
        "An LR scheduler plus early stopping on the validation metric",
        "Increasing the number of epochs to 1000",
        "Switching every tensor to float64",
      ],
      correctOption: 1,
      hint: "Think about what changes the final quality rather than the memory profile.",
      solution: `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# after each epoch: scheduler.step(); stop if val has not improved for N epochs`,
      explanation:
        "Decaying the learning rate reliably improves the final result, and early stopping avoids both wasted compute and overfitting. empty_cache() per batch only adds synchronisation, and float64 halves throughput for no benefit.",
    },
  ],
};
