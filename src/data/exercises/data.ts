import type { Exercise } from "@/lib/types";

/** Level 6 — Data pipeline. */
export const dataExercises: Exercise[] = [
  {
    id: "ex-data-1",
    title: "Dataset methods",
    topic: "data",
    level: 6,
    difficulty: "easy",
    importance: "high",
    type: "multiple-choice",
    question: "Which two methods must a map-style `Dataset` implement?",
    options: [
      "__init__ and forward",
      "__len__ and __getitem__",
      "__iter__ and __next__",
      "load and batch",
    ],
    correctOption: 1,
    hint: "One reports the size, one returns a single sample.",
    solution: "def __len__(self): ...\ndef __getitem__(self, idx): ...",
    explanation:
      "__len__ tells the sampler how many indices exist; __getitem__ maps one index to one (input, target) pair. The DataLoader handles batching.",
    tags: ["Dataset", "protocol"],
  },
  {
    id: "ex-data-2",
    title: "Implement __len__",
    topic: "data",
    level: 6,
    difficulty: "easy",
    importance: "high",
    type: "code",
    question: "Complete `__len__` for a dataset storing features in `self.X`.",
    context: `class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        ________`,
    acceptedAnswers: ["return len(self.X)", "return self.X.shape[0]", "return self.X.size(0)", "return len(self.y)"],
    requiredPatterns: ["return"],
    hint: "Return the number of samples.",
    solution: "return len(self.X)",
    explanation: "The DataLoader uses this to generate indices in [0, len(dataset)).",
    tags: ["Dataset", "__len__"],
  },
  {
    id: "ex-data-3",
    title: "Write a custom Dataset",
    topic: "data",
    level: 6,
    difficulty: "hard",
    importance: "high",
    type: "memory",
    question: "Write a complete custom Dataset class that wraps tensors `X` and `y`.",
    starterCode: "class MyDataset(Dataset):\n",
    requiredPatterns: [
      "class\\s\\w+\\(Dataset\\)",
      "def\\s__init__\\(self",
      "def\\s__len__\\(self\\)",
      "def\\s__getitem__\\(self,\\w+\\)",
      "return",
    ],
    hint: "Three methods: __init__, __len__, __getitem__.",
    solution: `class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]`,
    explanation:
      "__getitem__ returns ONE sample as a tuple; the DataLoader stacks them into batches. Put expensive per-sample work (decoding, augmentation) here so worker processes can parallelise it.",
    interviewNote: "🔥 A very common 'write this on the whiteboard' request.",
    tags: ["Dataset", "memory", "custom"],
  },
  {
    id: "ex-data-4",
    title: "Create a DataLoader",
    topic: "data",
    level: 6,
    difficulty: "easy",
    importance: "high",
    type: "code",
    question: "Create a DataLoader over `train_ds` with batch size 32 and shuffling enabled.",
    acceptedAnswers: [
      "train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)",
      "loader = DataLoader(train_ds, batch_size=32, shuffle=True)",
      "train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)",
      "DataLoader(train_ds, batch_size=32, shuffle=True)",
    ],
    requiredPatterns: ["DataLoader", "batch_size=32", "shuffle=True"],
    hint: "DataLoader(dataset, batch_size=..., shuffle=...).",
    solution: "train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)",
    explanation:
      "shuffle=True reshuffles every epoch, which decorrelates consecutive batches. Use shuffle=False for validation so metrics are comparable across runs.",
    tags: ["DataLoader", "batch_size", "shuffle"],
  },
  {
    id: "ex-data-5",
    title: "Validation loader",
    topic: "data",
    level: 6,
    difficulty: "easy",
    importance: "medium",
    type: "multiple-choice",
    question: "Should the validation DataLoader use shuffle=True?",
    options: [
      "Yes, always shuffle",
      "No — shuffling only helps optimisation; validation metrics should be deterministic and comparable",
      "Yes, but only with a fixed seed",
      "It changes the computed accuracy",
    ],
    correctOption: 1,
    hint: "No gradient steps happen there.",
    solution: "val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)",
    explanation:
      "Shuffling exists to decorrelate gradient updates. Validation computes an average over the whole set, so the order does not change the number — but keeping it fixed makes per-batch logs and any sample-level inspection reproducible.",
    tags: ["DataLoader", "shuffle", "validation"],
  },
  {
    id: "ex-data-6",
    title: "Iterate a DataLoader",
    topic: "data",
    level: 6,
    difficulty: "easy",
    importance: "high",
    type: "code",
    question: "Write the `for` statement that iterates batches of features and labels from `train_loader`.",
    acceptedAnswers: [
      "for X, y in train_loader:",
      "for X, y in train_loader:\npass",
      "for inputs, targets in train_loader:",
      "for batch_X, batch_y in train_loader:",
    ],
    requiredPatterns: ["for\\s\\w+,\\w+\\sin\\s\\w+:"],
    hint: "Each iteration yields a tuple that you can unpack.",
    solution: "for X, y in train_loader:",
    explanation:
      "The loader yields whatever __getitem__ returns, stacked into batches: (B, ...) features and (B, ...) targets.",
    tags: ["DataLoader", "iterate"],
  },
  {
    id: "ex-data-7",
    title: "Speed up loading",
    topic: "data",
    level: 6,
    difficulty: "medium",
    importance: "medium",
    type: "multiple-choice",
    question: "GPU utilisation sits at 25% and the GPU is waiting for data. Which change helps most?",
    options: [
      "Reduce the learning rate",
      "Set num_workers > 0 and pin_memory=True on the DataLoader",
      "Switch from Adam to SGD",
      "Call torch.cuda.empty_cache() every batch",
    ],
    correctOption: 1,
    hint: "The bottleneck is the input pipeline, not the model.",
    solution: "DataLoader(ds, batch_size=64, num_workers=8, pin_memory=True, persistent_workers=True)",
    explanation:
      "num_workers>0 loads batches in parallel subprocesses so preprocessing overlaps with GPU compute; pin_memory=True enables faster (and async) host→device copies. Always identify whether you are data-bound or compute-bound first.",
    tags: ["num_workers", "pin_memory", "throughput"],
  },
  {
    id: "ex-data-8",
    title: "Variable-length sequences",
    topic: "data",
    level: 6,
    difficulty: "hard",
    importance: "medium",
    type: "multiple-choice",
    question: "Your text samples have different lengths and the default DataLoader raises a stack error. What is the standard fix?",
    options: [
      "Set batch_size=1",
      "Provide a custom collate_fn that pads the sequences to the longest in the batch",
      "Convert the sequences to NumPy first",
      "Use shuffle=False",
    ],
    correctOption: 1,
    hint: "The default collate calls torch.stack, which needs identical shapes.",
    solution: `def collate(batch):
    seqs, labels = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, torch.stack(labels)`,
    explanation:
      "Padding per batch (rather than to a global maximum) wastes far less compute. Carry a padding mask alongside so attention and losses can ignore the pad positions.",
    tags: ["collate_fn", "padding", "nlp"],
  },
  {
    id: "ex-data-9",
    title: "Batches per epoch",
    topic: "data",
    level: 6,
    difficulty: "medium",
    importance: "medium",
    type: "fill-blank",
    question:
      "A dataset has 50,000 samples and batch_size=64 with drop_last=False. How many iterations does one epoch take? (a number)",
    acceptedAnswers: ["782"],
    hint: "ceil(50000 / 64).",
    solution: "math.ceil(50000 / 64) == 782",
    explanation:
      "781 full batches of 64 (49,984 samples) plus one final batch of 16. With drop_last=True it would be 781. len(loader) returns this number of batches; len(loader.dataset) returns 50,000.",
    tags: ["epoch", "batch", "len"],
  },
  {
    id: "ex-data-10",
    title: "Why mini-batches?",
    topic: "data",
    level: 6,
    difficulty: "medium",
    importance: "high",
    type: "multiple-choice",
    question: "Why train on mini-batches instead of the whole dataset at once?",
    options: [
      "Mini-batches give exactly the same gradient but use less memory",
      "They trade gradient accuracy for memory and speed, parallelise well on GPUs, and their gradient noise acts as a regulariser",
      "Because PyTorch cannot process more than 512 samples at once",
      "To make the learning rate irrelevant",
    ],
    correctOption: 1,
    hint: "Think memory, hardware utilisation and the effect of noise.",
    solution: "DataLoader(ds, batch_size=64, shuffle=True)",
    explanation:
      "Full-batch gradients are exact but memory-hungry and give one update per epoch. Single samples underuse the GPU and are very noisy. Mini-batches sit in the middle — and the noise itself helps escape sharp minima.",
    tags: ["mini-batch", "concept", "sgd"],
  },
  {
    id: "ex-data-11",
    title: "Split a dataset",
    topic: "data",
    level: 6,
    difficulty: "medium",
    importance: "low",
    type: "code",
    question: "Split `ds` (1000 samples) into 800 training and 200 validation samples.",
    acceptedAnswers: [
      "train_ds, val_ds = random_split(ds, [800, 200])",
      "train_ds, val_ds = torch.utils.data.random_split(ds, [800, 200])",
      "train_ds, val_ds = random_split(ds, [0.8, 0.2])",
    ],
    requiredPatterns: ["random_split"],
    hint: "torch.utils.data.random_split(dataset, lengths).",
    solution: "train_ds, val_ds = random_split(ds, [800, 200])",
    explanation:
      "Pass generator=torch.Generator().manual_seed(42) to make the split reproducible. Fit any normalisation statistics on the training split only.",
    tags: ["random_split", "validation"],
  },
  {
    id: "ex-data-12",
    title: "Peek at one batch",
    topic: "data",
    level: 6,
    difficulty: "medium",
    importance: "low",
    type: "code",
    question: "Grab a single batch from `train_loader` for a shape sanity check.",
    acceptedAnswers: [
      "X, y = next(iter(train_loader))",
      "batch = next(iter(train_loader))",
      "X, y = next(iter(train_loader))\n",
    ],
    requiredPatterns: ["next\\(iter\\(train_loader\\)\\)"],
    hint: "Wrap the loader in iter() and take the first item.",
    solution: "X, y = next(iter(train_loader))",
    explanation:
      "The fastest way to verify shapes and dtypes before launching a long run. Note it spins up the workers, so it is not free in a tight loop.",
    tags: ["DataLoader", "debug", "next"],
  },
];
