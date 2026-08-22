import type { MiniProject } from "@/lib/types";

export const binaryClassifier: MiniProject = {
  slug: "binary-classifier",
  title: "Binary Classification",
  tagline: "Logits, BCEWithLogitsLoss, thresholding — and why sigmoid does not belong in the model.",
  difficulty: "easy",
  estimatedMinutes: 20,
  topics: ["nn", "losses", "training", "evaluation"],
  outline: ["MLP", "Logits", "BCEWithLogitsLoss", "Adam", "Training", "Threshold"],
  overview:
    "Build a small MLP for binary classification. The theme of this project is the logits/probabilities distinction, which is one of the most commonly probed ideas in interviews.",
  steps: [
    {
      id: "bc-1",
      title: "Build the network",
      brief:
        "Create a Sequential model: Linear(20 → 64), ReLU, Linear(64 → 1). Assign it to `model`.",
      kind: "code",
      requiredPatterns: ["nn\\.Sequential", "nn\\.Linear\\(20,64\\)", "nn\\.ReLU", "nn\\.Linear\\(64,1\\)"],
      hint: "No activation after the last layer.",
      solution: `model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)`,
      explanation:
        "The output is a single raw score per sample — a logit. Adding a sigmoid here would be a mistake, because the loss will apply it internally.",
    },
    {
      id: "bc-2",
      title: "What is a logit?",
      brief: "Which statement about the model's output is correct?",
      kind: "multiple-choice",
      options: [
        "It is a probability between 0 and 1",
        "It is an unbounded raw score; sigmoid maps it to a probability",
        "It is the predicted class (0 or 1)",
        "It is the loss value",
      ],
      correctOption: 1,
      hint: "There is no activation on the final layer.",
      solution: "prob = torch.sigmoid(logit)   # only at inference",
      explanation:
        "A logit is log-odds and can be any real number. Negative → probability below 0.5, positive → above.",
    },
    {
      id: "bc-3",
      title: "Choose the loss",
      brief: "Create the loss function that takes raw logits for binary classification.",
      kind: "code",
      acceptedAnswers: [
        "loss_fn = nn.BCEWithLogitsLoss()",
        "criterion = nn.BCEWithLogitsLoss()",
        "loss_fn = torch.nn.BCEWithLogitsLoss()",
      ],
      requiredPatterns: ["BCEWithLogitsLoss\\(\\)"],
      hint: "It fuses sigmoid into the loss.",
      solution: "loss_fn = nn.BCEWithLogitsLoss()",
      explanation:
        "The fused version uses the log-sum-exp trick and stays stable for large logits, where separate sigmoid + BCELoss can produce inf or NaN.",
    },
    {
      id: "bc-4",
      title: "Fix the target dtype and shape",
      brief:
        "`y` is an int64 tensor of shape (N,). Convert it to what BCEWithLogitsLoss expects, given logits of shape (N, 1).",
      kind: "code",
      acceptedAnswers: [
        "y = y.float().unsqueeze(1)",
        "y = y.unsqueeze(1).float()",
        "y = y.float().view(-1, 1)",
        "y = y.float().reshape(-1, 1)",
        "y = y.view(-1, 1).float()",
      ],
      requiredPatterns: ["float\\(\\)", "(unsqueeze\\(1\\)|view\\(-1,1\\)|reshape\\(-1,1\\))"],
      hint: "Float dtype AND the same shape as the logits.",
      solution: "y = y.float().unsqueeze(1)",
      explanation:
        "BCE compares element-wise, so shapes must match exactly — (N,) against (N, 1) would broadcast to (N, N). Note the contrast with CrossEntropyLoss, which wants int64 of shape (N,).",
    },
    {
      id: "bc-5",
      title: "Create the optimizer",
      brief: "Use Adam with a learning rate of 1e-3.",
      kind: "code",
      acceptedAnswers: [
        "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)",
        "optimizer = optim.Adam(model.parameters(), lr=1e-3)",
        "optimizer = torch.optim.Adam(model.parameters(), lr=0.001)",
        "optimizer = optim.Adam(model.parameters(), lr=0.001)",
      ],
      requiredPatterns: ["Adam", "model\\.parameters\\(\\)", "lr="],
      hint: "1e-3 is Adam's default.",
      solution: "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)",
      explanation: "Adam adapts a per-parameter step size, which usually converges faster than SGD without tuning.",
    },
    {
      id: "bc-6",
      title: "Training step",
      brief: "Write the body of the batch loop: one full optimisation step.",
      kind: "code",
      context: "model.train()\nfor X, y in train_loader:",
      requiredPatterns: ["zero_grad\\(\\)", "model\\(X\\)", "backward\\(\\)", "\\.step\\(\\)"],
      hint: "The same five lines as always.",
      solution: `model.train()
for X, y in train_loader:
    optimizer.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()`,
      explanation: "Naming the output `logits` rather than `pred` documents that it has not been through a sigmoid.",
    },
    {
      id: "bc-7",
      title: "Convert logits to predictions",
      brief:
        "At inference, turn `logits` into hard 0/1 predictions using a threshold of 0.5 on the probability.",
      kind: "code",
      acceptedAnswers: [
        "preds = (torch.sigmoid(logits) > 0.5).float()",
        "preds = (torch.sigmoid(logits) >= 0.5).float()",
        "preds = (logits > 0).float()",
        "preds = (torch.sigmoid(logits) > 0.5).long()",
        "preds = (logits > 0).long()",
      ],
      requiredPatterns: ["(sigmoid|>0)"],
      hint: "sigmoid(logit) > 0.5 is equivalent to logit > 0.",
      solution: "preds = (torch.sigmoid(logits) > 0.5).float()",
      explanation:
        "Because sigmoid is monotonic and sigmoid(0) = 0.5, thresholding the logit at 0 gives identical predictions and skips the sigmoid entirely.",
    },
    {
      id: "bc-8",
      title: "Tuning the threshold",
      brief: "Your positive class is rare and recall matters more than precision. What do you change?",
      kind: "multiple-choice",
      options: [
        "Raise the threshold above 0.5",
        "Lower the threshold below 0.5, and/or use pos_weight in BCEWithLogitsLoss",
        "Switch to CrossEntropyLoss",
        "Remove the sigmoid",
      ],
      correctOption: 1,
      hint: "A lower bar means more positives are predicted.",
      solution: "loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))",
      explanation:
        "The 0.5 threshold is a choice, not a law. Lowering it trades precision for recall; pos_weight makes the loss itself penalise missed positives more heavily during training.",
    },
  ],
};
