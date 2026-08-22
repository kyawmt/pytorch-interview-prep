import type { MiniProject } from "@/lib/types";

export const cnnClassifier: MiniProject = {
  slug: "cnn-classifier",
  title: "CNN Image Classifier",
  tagline: "Conv → ReLU → Pool blocks, and the shape arithmetic interviewers love to ask for.",
  difficulty: "medium",
  estimatedMinutes: 30,
  topics: ["nn", "shapes", "training"],
  outline: ["Conv block 1", "Conv block 2", "Shape maths", "Flatten", "Classifier head", "Forward"],
  overview:
    "Build a small CNN for 32×32 RGB images (CIFAR-10 style). Most of the work is tracking shapes — get the flattened size wrong and nothing runs.",
  steps: [
    {
      id: "cnn-1",
      title: "First convolution",
      brief:
        "Create the first conv layer: RGB input, 32 output channels, 3×3 kernel, padding that preserves the spatial size.",
      kind: "code",
      acceptedAnswers: [
        "self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)",
        "conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)",
        "self.conv1 = nn.Conv2d(3, 32, 3, padding=1)",
        "conv1 = nn.Conv2d(3, 32, 3, padding=1)",
      ],
      requiredPatterns: ["nn\\.Conv2d\\(3,32", "padding=1"],
      hint: "padding = kernel_size // 2 preserves H and W.",
      solution: "self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)",
      explanation: "(32 + 2·1 - 3)/1 + 1 = 32, so the output is (B, 32, 32, 32).",
    },
    {
      id: "cnn-2",
      title: "After the first pool",
      brief: "Input is (B, 3, 32, 32). After conv1 then MaxPool2d(2), what is the shape?",
      kind: "shape",
      acceptedAnswers: ["(B, 32, 16, 16)", "(1, 32, 16, 16)"],
      hint: "The padded 3×3 conv preserves H/W; MaxPool2d(2) halves them.",
      solution: "torch.Size([B, 32, 16, 16])",
      explanation: "Channels come from out_channels=32; 32 → 16 spatially because pooling with kernel 2 and stride 2 halves each dimension.",
    },
    {
      id: "cnn-3",
      title: "Second convolution",
      brief: "Create the second conv layer: 32 input channels, 64 output channels, 3×3, padding 1.",
      kind: "code",
      acceptedAnswers: [
        "self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)",
        "conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)",
        "self.conv2 = nn.Conv2d(32, 64, 3, padding=1)",
        "conv2 = nn.Conv2d(32, 64, 3, padding=1)",
      ],
      requiredPatterns: ["nn\\.Conv2d\\(32,64", "padding=1"],
      hint: "in_channels must match the previous layer's out_channels.",
      solution: "self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)",
      explanation:
        "Mismatching in_channels is the most common CNN error: 'expected input to have 32 channels but got 3 channels'.",
    },
    {
      id: "cnn-4",
      title: "Flattened size",
      brief:
        "After conv2 + MaxPool2d(2) the tensor is (B, 64, 8, 8). What in_features does the classifier Linear need?",
      kind: "multiple-choice",
      options: ["64", "512", "4096", "8"],
      correctOption: 2,
      hint: "64 × 8 × 8.",
      solution: "nn.Linear(64 * 8 * 8, 10)   # 4096 -> 10",
      explanation:
        "Everything after the batch dimension is flattened into one vector. Writing it as `64 * 8 * 8` keeps the arithmetic legible.",
    },
    {
      id: "cnn-5",
      title: "Classifier head",
      brief: "Create the Linear layer mapping the flattened features to 10 classes.",
      kind: "code",
      acceptedAnswers: [
        "self.fc = nn.Linear(64 * 8 * 8, 10)",
        "self.fc = nn.Linear(4096, 10)",
        "fc = nn.Linear(64 * 8 * 8, 10)",
        "fc = nn.Linear(4096, 10)",
      ],
      requiredPatterns: ["nn\\.Linear\\((64\\*8\\*8|4096),10\\)"],
      hint: "Either the product or the literal 4096.",
      solution: "self.fc = nn.Linear(64 * 8 * 8, 10)",
      explanation: "Raw logits again — the softmax lives inside CrossEntropyLoss.",
    },
    {
      id: "cnn-6",
      title: "Flatten before the head",
      brief: "Inside forward(), flatten `x` while keeping the batch dimension.",
      kind: "code",
      acceptedAnswers: [
        "x = x.flatten(1)",
        "x = x.view(x.size(0), -1)",
        "x = x.reshape(x.size(0), -1)",
        "x = x.flatten(start_dim=1)",
        "x = x.view(x.shape[0], -1)",
        "x = x.reshape(x.shape[0], -1)",
      ],
      requiredPatterns: ["(flatten\\(1|-1)"],
      hint: "flatten(1), or reshape to (batch, -1).",
      solution: "x = x.flatten(1)",
      explanation:
        "A bare x.flatten() would merge the batch too, producing one enormous vector and a confusing shape error at the Linear layer.",
    },
    {
      id: "cnn-7",
      title: "Complete forward()",
      brief:
        "Write the full forward pass: conv1 → relu → pool → conv2 → relu → pool → flatten → fc.",
      kind: "code",
      context: "def forward(self, x):",
      requiredPatterns: [
        "conv1",
        "(relu|F\\.relu)",
        "pool",
        "conv2",
        "(flatten|view|reshape)",
        "fc",
        "return",
      ],
      hint: "Follow the outline; self.pool can be reused for both stages.",
      solution: `def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.flatten(1)
    return self.fc(x)`,
      explanation:
        "Pooling has no parameters, so one nn.MaxPool2d(2) instance can be reused. The same is true of a stateless activation — but a layer WITH parameters must never be reused unless you intend weight sharing.",
    },
    {
      id: "cnn-8",
      title: "Why convolutions for images?",
      brief: "What is the main advantage of Conv2d over a fully connected layer on images?",
      kind: "multiple-choice",
      options: [
        "It always trains faster",
        "Weight sharing and locality: far fewer parameters, and features are detected anywhere in the image",
        "It removes the need for an activation function",
        "It works with any input dtype",
      ],
      correctOption: 1,
      hint: "Compare parameter counts for a 224×224 image.",
      solution: "# Conv2d(3, 64, 3): 1,792 params — independent of image size\n# Linear(150528, 64): 9.6M params",
      explanation:
        "A kernel is applied at every position, so the parameter count is independent of resolution and a feature learned in one corner transfers to the whole image (translation equivariance).",
    },
  ],
};
