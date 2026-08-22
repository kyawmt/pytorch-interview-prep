import type { MiniProject } from "@/lib/types";

export const mnistClassifier: MiniProject = {
  slug: "mnist-classifier",
  title: "MNIST Classifier",
  tagline: "Flatten → Linear → ReLU → Linear → 10 logits, with a real train/eval split.",
  difficulty: "medium",
  estimatedMinutes: 25,
  topics: ["nn", "losses", "training", "evaluation", "data"],
  outline: ["DataLoader", "MLP", "CrossEntropyLoss", "Adam", "Train", "Evaluate", "argmax"],
  overview:
    "The canonical first classifier. The interesting parts are the input shape (28×28 images must be flattened), the fact that the model outputs raw logits, and evaluating with argmax under eval() + no_grad().",
  steps: [
    {
      id: "mn-1",
      title: "Batch the data",
      brief: "Create a training DataLoader over `train_ds` with batch size 64 and shuffling.",
      kind: "code",
      acceptedAnswers: [
        "train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)",
        "train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)",
      ],
      requiredPatterns: ["DataLoader", "batch_size=64", "shuffle=True"],
      hint: "DataLoader(dataset, batch_size=..., shuffle=...).",
      solution: "train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)",
      explanation: "Each batch will be (64, 1, 28, 28) images and (64,) int64 labels.",
    },
    {
      id: "mn-2",
      title: "Input size",
      brief: "MNIST images arrive as (B, 1, 28, 28). What in_features does the first Linear layer need?",
      kind: "multiple-choice",
      options: ["28", "128", "784", "256"],
      correctOption: 2,
      hint: "1 × 28 × 28.",
      solution: "nn.Flatten()  # (B, 1, 28, 28) -> (B, 784)",
      explanation: "Flattening everything after the batch dimension gives 784 features per image.",
    },
    {
      id: "mn-3",
      title: "Build the model",
      brief:
        "Create a Sequential model: Flatten, Linear(784 → 128), ReLU, Linear(128 → 10).",
      kind: "code",
      requiredPatterns: [
        "nn\\.Sequential",
        "nn\\.Flatten\\(\\)",
        "nn\\.Linear\\(784,128\\)",
        "nn\\.ReLU",
        "nn\\.Linear\\(128,10\\)",
      ],
      hint: "nn.Flatten() defaults to start_dim=1, so the batch dimension survives.",
      solution: `model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)`,
      explanation:
        "The output is (B, 10) raw logits — one score per digit. No softmax: CrossEntropyLoss applies log_softmax itself.",
    },
    {
      id: "mn-4",
      title: "Loss and optimizer",
      brief: "Create the multi-class loss and an Adam optimizer with lr=1e-3 (two lines).",
      kind: "code",
      requiredPatterns: ["CrossEntropyLoss\\(\\)", "Adam", "model\\.parameters\\(\\)"],
      hint: "CrossEntropyLoss for 10 mutually exclusive classes.",
      solution: `loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)`,
      explanation:
        "CrossEntropyLoss takes (B, 10) float logits and (B,) int64 labels. The MNIST labels are already int64.",
    },
    {
      id: "mn-5",
      title: "Train one epoch",
      brief: "Write the training epoch: set the mode, iterate batches, and run one optimisation step each.",
      kind: "code",
      requiredPatterns: [
        "model\\.train\\(\\)",
        "for.+in.+:",
        "zero_grad\\(\\)",
        "model\\(X",
        "backward\\(\\)",
        "\\.step\\(\\)",
      ],
      hint: "model.train() first, then the five lines per batch.",
      solution: `model.train()
for X, y in train_loader:
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()`,
      explanation:
        "model.train() matters as soon as you add dropout or BatchNorm — build the habit now rather than debugging it later.",
    },
    {
      id: "mn-6",
      title: "Predicted digits",
      brief: "Convert `logits` of shape (B, 10) into predicted digit labels of shape (B,).",
      kind: "code",
      acceptedAnswers: [
        "preds = logits.argmax(dim=1)",
        "preds = logits.argmax(1)",
        "preds = torch.argmax(logits, dim=1)",
        "preds = logits.argmax(dim=-1)",
      ],
      requiredPatterns: ["argmax"],
      hint: "The index of the largest logit is the predicted class.",
      solution: "preds = logits.argmax(dim=1)",
      explanation: "No softmax needed — it is monotonic, so it cannot change which entry is largest.",
    },
    {
      id: "mn-7",
      title: "Evaluate",
      brief:
        "Write the evaluation loop that counts correct predictions: switch mode, disable the graph, iterate, compare.",
      kind: "code",
      context: "correct, total = 0, 0",
      requiredPatterns: [
        "model\\.eval\\(\\)",
        "torch\\.(no_grad|inference_mode)",
        "argmax",
        "==y",
      ],
      hint: "eval() AND no_grad() — they do different things.",
      solution: `model.eval()
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
print(correct / total)`,
      explanation:
        "eval() disables dropout/BatchNorm updates; no_grad() stops graph construction. A simple MLP like this reaches roughly 97% on MNIST.",
    },
    {
      id: "mn-8",
      title: "Why no softmax in the model?",
      brief: "The model returns raw logits. Why not end with nn.Softmax(dim=1)?",
      kind: "multiple-choice",
      options: [
        "Softmax is too slow to compute",
        "CrossEntropyLoss applies log_softmax internally, so adding softmax applies it twice and weakens gradients",
        "Softmax would change the argmax",
        "Softmax does not work on (B, 10) tensors",
      ],
      correctOption: 1,
      hint: "CrossEntropyLoss = log_softmax + NLLLoss.",
      solution: "logits = model(X)\nloss = loss_fn(logits, y)   # no softmax",
      explanation:
        "Double-softmaxing does not raise an error, it just trains worse — which is why it survives in real codebases for a long time.",
    },
  ],
};
