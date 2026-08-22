import type { MiniProject } from "@/lib/types";

export const textClassifier: MiniProject = {
  slug: "text-classifier",
  title: "Text Classification",
  tagline: "Token ids → embeddings → pooled representation → classifier.",
  difficulty: "medium",
  estimatedMinutes: 25,
  topics: ["nn", "transformers", "data", "shapes"],
  outline: ["Token ids", "nn.Embedding", "Mean pooling", "Linear head", "Padding", "Loss"],
  overview:
    "The simplest NLP model that actually works: average the word embeddings of a sentence and classify the result. It introduces embeddings, pooling and padding — the vocabulary you need before transformers make sense.",
  steps: [
    {
      id: "tc-1",
      title: "Embedding layer",
      brief: "Create an embedding layer for a 20,000-token vocabulary with 128-dimensional vectors.",
      kind: "code",
      acceptedAnswers: [
        "self.emb = nn.Embedding(20000, 128)",
        "emb = nn.Embedding(20000, 128)",
        "self.embedding = nn.Embedding(20000, 128)",
        "embedding = nn.Embedding(20000, 128)",
      ],
      requiredPatterns: ["nn\\.Embedding\\(20000,128\\)"],
      hint: "nn.Embedding(num_embeddings, embedding_dim).",
      solution: "self.emb = nn.Embedding(20000, 128)",
      explanation: "A (20000, 128) learned lookup table. Feeding it (B, T) ids returns (B, T, 128).",
    },
    {
      id: "tc-2",
      title: "Embedding input dtype",
      brief: "What dtype must the token id tensor have?",
      kind: "multiple-choice",
      options: ["torch.float32", "torch.int64 (long)", "torch.bool", "Any numeric dtype works"],
      correctOption: 1,
      hint: "The ids index rows of a matrix.",
      solution: "ids = ids.long()",
      explanation:
        "Float ids raise 'Expected tensor for argument #1 indices to have scalar type Long'. The same requirement applies to CrossEntropyLoss targets.",
    },
    {
      id: "tc-3",
      title: "Embedding output shape",
      brief: "`ids` has shape (32, 100). What is the shape of `self.emb(ids)`?",
      kind: "shape",
      acceptedAnswers: ["(32, 100, 128)"],
      hint: "Each id is replaced by a 128-dim vector.",
      solution: "torch.Size([32, 100, 128])",
      explanation: "(batch, seq_len) → (batch, seq_len, embedding_dim): the canonical NLP tensor layout.",
    },
    {
      id: "tc-4",
      title: "Mean pooling",
      brief:
        "Reduce the (B, T, C) embeddings to one (B, C) vector per sentence by averaging over the tokens.",
      kind: "code",
      acceptedAnswers: [
        "pooled = x.mean(dim=1)",
        "pooled = x.mean(1)",
        "pooled = torch.mean(x, dim=1)",
        "x = x.mean(dim=1)",
        "x = x.mean(1)",
      ],
      requiredPatterns: ["mean\\((dim=)?1\\)"],
      hint: "dim=1 is the sequence axis, and it should disappear.",
      solution: "pooled = x.mean(dim=1)",
      explanation:
        "Averaging discards word order — surprisingly effective for topic classification, and hopeless for anything needing syntax. That limitation is exactly what attention fixes.",
    },
    {
      id: "tc-5",
      title: "Handle padding",
      brief: "Sentences are padded to equal length with id 0. What is the problem with a plain mean?",
      kind: "multiple-choice",
      options: [
        "There is no problem — padding vectors are zero",
        "Padding tokens have a learned embedding too, so they contribute to the average and dilute short sentences",
        "The mean will raise a shape error",
        "Padding makes the model slower but not wrong",
      ],
      correctOption: 1,
      hint: "Is index 0's embedding necessarily zero?",
      solution: `emb = nn.Embedding(20000, 128, padding_idx=0)
mask = (ids != 0).unsqueeze(-1)
pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1)`,
      explanation:
        "padding_idx=0 pins that row to zero and excludes it from gradients, but the DIVISOR is still the padded length. A masked mean divides by the real token count.",
    },
    {
      id: "tc-6",
      title: "Classifier head",
      brief: "Map the pooled 128-dim representation to 4 class logits.",
      kind: "code",
      acceptedAnswers: [
        "self.fc = nn.Linear(128, 4)",
        "fc = nn.Linear(128, 4)",
        "self.classifier = nn.Linear(128, 4)",
      ],
      requiredPatterns: ["nn\\.Linear\\(128,4\\)"],
      hint: "in_features is the embedding dimension.",
      solution: "self.fc = nn.Linear(128, 4)",
      explanation: "Output (B, 4) logits, ready for CrossEntropyLoss.",
    },
    {
      id: "tc-7",
      title: "Complete forward()",
      brief: "Write the forward pass: embed, mean-pool over the sequence, classify.",
      kind: "code",
      context: "def forward(self, ids):",
      requiredPatterns: ["emb", "mean", "(fc|classifier)", "return"],
      hint: "Three lines.",
      solution: `def forward(self, ids):
    x = self.emb(ids)        # (B, T, C)
    x = x.mean(dim=1)        # (B, C)
    return self.fc(x)        # (B, num_classes)`,
      explanation:
        "Tracking the shape in a comment at each line is a habit worth keeping — it makes shape bugs visible while writing rather than at runtime.",
    },
    {
      id: "tc-8",
      title: "Batching variable-length text",
      brief: "Different sentences have different lengths and the DataLoader fails to stack them. Fix?",
      kind: "multiple-choice",
      options: [
        "Set batch_size=1",
        "Pass a custom collate_fn that pads each batch to its longest sequence",
        "Truncate every sentence to one token",
        "Convert the sentences to NumPy first",
      ],
      correctOption: 1,
      hint: "The default collate calls torch.stack, which needs identical shapes.",
      solution: `def collate(batch):
    seqs, labels = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, torch.stack(labels)`,
      explanation:
        "Padding per batch rather than to a global maximum wastes far less compute. Sorting similar lengths into the same batch (bucketing) reduces padding further.",
    },
  ],
};
