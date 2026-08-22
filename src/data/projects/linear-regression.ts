import type { MiniProject } from "@/lib/types";

export const linearRegression: MiniProject = {
  slug: "linear-regression",
  title: "Linear Regression",
  tagline: "The smallest complete PyTorch program: tensors, a Linear layer, MSE, SGD and a loop.",
  difficulty: "easy",
  estimatedMinutes: 15,
  topics: ["tensors", "nn", "losses", "optimizers", "training"],
  outline: ["Data", "nn.Linear", "MSELoss", "SGD", "Training loop", "Inference"],
  overview:
    "Fit y = 3x + 2 from noisy samples. Every piece here reappears in every model you will ever write, so the goal is fluency rather than accuracy.",
  steps: [
    {
      id: "lr-1",
      title: "Create the training data",
      brief:
        "Build the input tensor `X` with shape (100, 1) of random values, so it can feed a Linear layer directly.",
      kind: "code",
      context: "# target relationship: y = 3x + 2 + noise",
      acceptedAnswers: ["X = torch.randn(100, 1)", "X = torch.rand(100, 1)", "X = torch.randn((100, 1))"],
      requiredPatterns: ["torch\\.rand", "100,1"],
      hint: "nn.Linear expects (batch, features) — even with a single feature.",
      solution: "X = torch.randn(100, 1)",
      explanation:
        "The trailing dimension of 1 is the feature count. A shape of (100,) would fail in Linear, because the last dimension must equal in_features.",
    },
    {
      id: "lr-2",
      title: "Generate the targets",
      brief: "Compute `y` as 3·X + 2 plus a little Gaussian noise.",
      kind: "code",
      acceptedAnswers: [
        "y = 3 * X + 2 + 0.1 * torch.randn(100, 1)",
        "y = 3 * X + 2 + torch.randn(100, 1) * 0.1",
        "y = 3 * X + 2 + 0.1 * torch.randn_like(X)",
        "y = 3 * X + 2 + torch.randn_like(X) * 0.1",
      ],
      requiredPatterns: ["3\\*X", "\\+2", "randn"],
      hint: "Broadcasting handles the scalars; use randn_like(X) to match the shape automatically.",
      solution: "y = 3 * X + 2 + 0.1 * torch.randn_like(X)",
      explanation:
        "y has the same (100, 1) shape as X, which matters: MSELoss between (100,) and (100, 1) would broadcast into a (100, 100) matrix and silently train on nonsense.",
    },
    {
      id: "lr-3",
      title: "Define the model",
      brief: "Create a single Linear layer mapping 1 input feature to 1 output.",
      kind: "code",
      acceptedAnswers: ["model = nn.Linear(1, 1)", "model = torch.nn.Linear(1, 1)"],
      requiredPatterns: ["nn\\.Linear\\(1,1\\)"],
      hint: "nn.Linear(in_features, out_features).",
      solution: "model = nn.Linear(1, 1)",
      explanation:
        "This layer has exactly two parameters — a weight and a bias — which should converge to about 3 and 2.",
    },
    {
      id: "lr-4",
      title: "Choose the loss",
      brief: "Create the loss function appropriate for regression.",
      kind: "code",
      acceptedAnswers: ["loss_fn = nn.MSELoss()", "criterion = nn.MSELoss()", "loss_fn = torch.nn.MSELoss()"],
      requiredPatterns: ["MSELoss\\(\\)"],
      hint: "Mean squared error.",
      solution: "loss_fn = nn.MSELoss()",
      explanation: "MSE penalises large errors quadratically, which is the standard default for continuous targets.",
    },
    {
      id: "lr-5",
      title: "Create the optimizer",
      brief: "Use SGD over the model's parameters with a learning rate of 0.01.",
      kind: "code",
      acceptedAnswers: [
        "optimizer = torch.optim.SGD(model.parameters(), lr=0.01)",
        "optimizer = optim.SGD(model.parameters(), lr=0.01)",
      ],
      requiredPatterns: ["SGD", "model\\.parameters\\(\\)", "lr=0\\.01"],
      hint: "The first argument is always the parameters.",
      solution: "optimizer = torch.optim.SGD(model.parameters(), lr=0.01)",
      explanation:
        "The optimizer holds references to the parameter tensors and reads their .grad after backward().",
    },
    {
      id: "lr-6",
      title: "Write the training loop body",
      brief:
        "Inside `for epoch in range(200):`, write the five lines of one optimisation step over the full batch (X, y).",
      kind: "code",
      context: "for epoch in range(200):",
      requiredPatterns: [
        "zero_grad\\(\\)",
        "\\w+=model\\(X\\)",
        "loss=\\w+\\(",
        "loss\\.backward\\(\\)",
        "\\.step\\(\\)",
      ],
      hint: "zero_grad → forward → loss → backward → step.",
      solution: `for epoch in range(200):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()`,
      explanation:
        "This is full-batch gradient descent — with 100 samples there is no need for a DataLoader. The structure is identical to a mini-batch loop.",
    },
    {
      id: "lr-7",
      title: "Inspect what was learned",
      brief: "Print the learned weight and bias to check they are close to 3 and 2.",
      kind: "code",
      acceptedAnswers: [
        "print(model.weight, model.bias)",
        "print(model.weight.item(), model.bias.item())",
        "print(model.weight.data, model.bias.data)",
      ],
      requiredPatterns: ["model\\.weight", "model\\.bias"],
      hint: "nn.Linear exposes .weight and .bias.",
      solution: "print(model.weight.item(), model.bias.item())",
      explanation:
        "Both are nn.Parameter tensors with requires_grad=True. .item() extracts the Python float from a single-element tensor.",
    },
    {
      id: "lr-8",
      title: "Predict on new data",
      brief: "Run inference on `X_new` without building a computational graph.",
      kind: "code",
      context: "X_new = torch.tensor([[5.0]])",
      requiredPatterns: ["torch\\.(no_grad|inference_mode)", "model\\(X_new\\)"],
      hint: "Wrap the forward pass in the no-gradient context manager.",
      solution: `with torch.no_grad():
    pred = model(X_new)
print(pred)   # ~17.0`,
      explanation:
        "3·5 + 2 = 17. no_grad() avoids building a graph you will never backpropagate through. (This model has no dropout or BatchNorm, so model.eval() changes nothing here — but calling it is a good habit.)",
    },
  ],
};
