import type { Exercise } from "@/lib/types";

/**
 * "Write from memory" tasks. These are validated purely by required patterns —
 * variable names are free, only the structure must be right.
 */
export const MEMORY_DRILLS: Exercise[] = [
  {
    id: "md-training-loop",
    title: "The training loop",
    topic: "training",
    level: 5,
    difficulty: "medium",
    importance: "high",
    type: "memory",
    question:
      "Write a complete training loop over a dataloader: training mode, the batch loop, and one full optimisation step.",
    requiredPatterns: [
      "\\.train\\(\\)",
      "for.+in.+:",
      "zero_grad\\(\\)",
      "\\w+=\\w+\\(\\w+\\)",
      "\\.backward\\(\\)",
      "\\.step\\(\\)",
    ],
    hint: "train() → for batch → zero_grad → forward → loss → backward → step.",
    solution: `model.train()
for X, y in train_loader:
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()`,
    explanation:
      "Names are free: `outputs = network(inputs)` is as valid as `pred = model(X)`. What is graded is that every step is present and in the right order.",
    interviewNote: "🔥 The single most valuable thing to be able to type without thinking.",
    tags: ["training loop", "memory"],
  },
  {
    id: "md-eval-loop",
    title: "The evaluation loop",
    topic: "evaluation",
    level: 5,
    difficulty: "medium",
    importance: "high",
    type: "memory",
    question:
      "Write an evaluation loop that computes accuracy: mode switch, no-grad context, forward pass, argmax comparison.",
    requiredPatterns: [
      "\\.eval\\(\\)",
      "torch\\.(no_grad|inference_mode)\\(\\)",
      "for.+in.+:",
      "argmax",
      "==",
    ],
    hint: "eval() and no_grad() do different jobs — you need both.",
    solution: `model.eval()
correct = 0
with torch.no_grad():
    for X, y in val_loader:
        pred = model(X)
        correct += (pred.argmax(dim=1) == y).sum().item()`,
    explanation:
      "eval() changes dropout and BatchNorm behaviour; no_grad() stops the graph being built. Missing either one is the most common validation bug.",
    tags: ["evaluation", "memory"],
  },
  {
    id: "md-dataset",
    title: "A custom Dataset",
    topic: "data",
    level: 6,
    difficulty: "medium",
    importance: "high",
    type: "memory",
    question: "Write a complete custom Dataset class wrapping two tensors.",
    requiredPatterns: [
      "class\\s\\w+\\(Dataset\\)",
      "def\\s__init__\\(self",
      "def\\s__len__\\(self\\)",
      "def\\s__getitem__\\(self,\\w+\\)",
      "return",
    ],
    hint: "Three methods, and __getitem__ returns a single sample.",
    solution: `class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]`,
    explanation:
      "__len__ tells the sampler how many indices exist; __getitem__ maps one index to one (input, target) pair.",
    tags: ["Dataset", "memory"],
  },
  {
    id: "md-model",
    title: "An MLP with nn.Module",
    topic: "nn",
    level: 4,
    difficulty: "medium",
    importance: "high",
    type: "memory",
    question:
      "Write an nn.Module subclass for a two-layer MLP: Linear → ReLU → Linear, with a forward method.",
    requiredPatterns: [
      "class\\s\\w+\\(nn\\.Module\\)",
      "super\\(\\)\\.__init__\\(\\)",
      "nn\\.Linear",
      "def\\sforward\\(self",
      "return",
    ],
    hint: "super().__init__() first, layers as attributes, then forward.",
    solution: `class MLP(nn.Module):
    def __init__(self, in_features, hidden, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))`,
    explanation:
      "No activation on the output — CrossEntropyLoss expects raw logits. Layers assigned as attributes are registered automatically.",
    tags: ["nn.Module", "memory"],
  },
  {
    id: "md-setup",
    title: "Model, loss, optimizer, device",
    topic: "training",
    level: 5,
    difficulty: "medium",
    importance: "high",
    type: "memory",
    question:
      "Write the four setup lines of a training script: resolve the device, move the model, create the loss, create the optimizer.",
    requiredPatterns: [
      "torch\\.cuda\\.is_available\\(\\)",
      "\\.to\\(device\\)",
      "Loss\\(\\)",
      "(Adam|AdamW|SGD)\\(",
      "parameters\\(\\)",
    ],
    hint: "device → model.to(device) → loss_fn → optimizer(model.parameters(), lr=...).",
    solution: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)`,
    explanation:
      "Order matters: move the model before constructing the optimizer, so its state is created for the parameters in their final location.",
    tags: ["setup", "memory"],
  },
  {
    id: "md-checkpoint",
    title: "Save and reload a model",
    topic: "saving",
    level: 7,
    difficulty: "medium",
    importance: "high",
    type: "memory",
    question: "Write the code to save a model's weights and reload them for inference.",
    requiredPatterns: [
      "torch\\.save",
      "state_dict\\(\\)",
      "torch\\.load",
      "load_state_dict",
      "\\.eval\\(\\)",
    ],
    hint: "Save the state_dict; rebuild, load, then switch to eval mode.",
    solution: `torch.save(model.state_dict(), "model.pth")

model = MyModel()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()`,
    explanation:
      "A state_dict is only tensors, so the architecture must be recreated first. eval() is required before inference or dropout stays active.",
    tags: ["saving", "memory"],
  },
  {
    id: "md-attention",
    title: "Scaled dot-product attention",
    topic: "transformers",
    level: 8,
    difficulty: "hard",
    importance: "high",
    type: "memory",
    question:
      "Write scaled dot-product attention from Q, K and V: scores, scaling, softmax, and the weighted sum.",
    requiredPatterns: ["(@|matmul)", "transpose", "(0\\.5|sqrt)", "softmax", "dim=-1"],
    hint: "softmax(QKᵀ / √d) V, with the softmax over the key axis.",
    solution: `scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
attn = torch.softmax(scores, dim=-1)
out = attn @ V`,
    explanation:
      "The √d_k scaling stops the dot products growing with dimension and saturating softmax. dim=-1 makes each query's weights sum to 1 across the keys.",
    tags: ["attention", "memory"],
  },
  {
    id: "md-amp",
    title: "A mixed-precision training step",
    topic: "performance",
    level: 7,
    difficulty: "hard",
    importance: "medium",
    type: "memory",
    question: "Write one AMP training step using autocast and a GradScaler.",
    requiredPatterns: ["zero_grad\\(\\)", "autocast", "scaler\\.scale", "backward\\(\\)", "scaler\\.step", "scaler\\.update"],
    hint: "Autocast wraps only the forward pass and loss; backward and step go through the scaler.",
    solution: `optimizer.zero_grad()
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = loss_fn(model(X), y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()`,
    explanation:
      "The scaler multiplies the loss so small float16 gradients do not underflow, then unscales them before the optimizer step.",
    tags: ["amp", "memory"],
  },
];
