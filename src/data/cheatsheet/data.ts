import type { CheatSheetEntry } from "@/lib/types";

export const dataEntries: CheatSheetEntry[] = [
  {
    id: "cs-dataset",
    topic: "data",
    section: "Dataset",
    title: "Custom Dataset",
    description:
      "Subclass Dataset and implement __len__ and __getitem__. __getitem__ returns ONE sample; the DataLoader handles batching.",
    syntax: "class MyDataset(Dataset): __len__ / __getitem__",
    example: `from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

ds = MyDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
print(len(ds), ds[0][0].shape)`,
    result: "100 torch.Size([10])",
    interviewNote:
      "🔥 Nearly guaranteed to come up. Say the contract out loud: __len__ tells the sampler how many indices exist, __getitem__ maps one index to one (input, target) pair. Do expensive per-sample work (decode, augment) inside __getitem__ so worker processes parallelise it.",
    importance: "high",
    tags: ["Dataset", "__getitem__", "__len__", "custom", "map-style"],
    relatedExercises: ["ex-data-1", "ex-data-2", "ex-data-3"],
  },
  {
    id: "cs-dataloader",
    topic: "data",
    section: "DataLoader",
    title: "DataLoader",
    description:
      "Wraps a Dataset and yields batches. Handles shuffling, batching, parallel loading and (optionally) pinned memory.",
    syntax: "DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)",
    example: `train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
X, y = next(iter(train_loader))
print(X.shape, y.shape, len(train_loader))`,
    interviewNote:
      "shuffle=True for training, False for validation and test (you want reproducible, comparable metrics). len(loader) is the number of BATCHES, len(dataset) the number of samples.",
    importance: "high",
    tags: ["DataLoader", "batch_size", "shuffle", "iterate", "batches"],
    relatedExercises: ["ex-data-4", "ex-data-5", "ex-data-6"],
  },
  {
    id: "cs-dataloader-args",
    topic: "data",
    section: "DataLoader",
    title: "num_workers · pin_memory · drop_last",
    description:
      "The knobs that separate a GPU-starved pipeline from a saturated one.",
    syntax: "num_workers=N  ·  pin_memory=True  ·  drop_last=True  ·  persistent_workers=True",
    example: `loader = DataLoader(ds, batch_size=64,
                    num_workers=8,          # subprocesses that prefetch
                    pin_memory=True,        # page-locked host memory
                    persistent_workers=True,
                    drop_last=True)         # keep every batch the same size`,
    interviewNote:
      "num_workers>0 spawns subprocesses, so data loading overlaps with GPU compute. pin_memory=True enables faster (and async, with non_blocking=True) host→device copies. drop_last avoids a ragged final batch, which matters for BatchNorm with batch size 1.",
    importance: "medium",
    tags: ["num_workers", "pin_memory", "drop_last", "throughput", "prefetch"],
    relatedExercises: ["ex-data-7"],
  },
  {
    id: "cs-collate",
    topic: "data",
    section: "DataLoader",
    title: "collate_fn",
    description:
      "Turns a list of samples into a batch. The default stacks tensors, which requires every sample to have the same shape — so variable-length text needs a custom one.",
    syntax: "DataLoader(ds, collate_fn=my_collate)",
    example: `from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    seqs, labels = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, torch.stack(labels)

loader = DataLoader(ds, batch_size=8, collate_fn=collate)`,
    interviewNote:
      "This is the standard answer to \"how do you batch sentences of different lengths?\": pad in collate_fn and carry an attention/padding mask alongside.",
    importance: "medium",
    tags: ["collate_fn", "padding", "variable length", "nlp", "pad_sequence"],
    relatedExercises: ["ex-data-8"],
  },
  {
    id: "cs-random-split",
    topic: "data",
    section: "Dataset",
    title: "random_split & TensorDataset",
    description:
      "TensorDataset wraps tensors you already have in memory; random_split carves a dataset into train/val portions.",
    syntax: "TensorDataset(X, y)  ·  random_split(ds, [n_train, n_val])",
    example: `from torch.utils.data import TensorDataset, random_split

ds = TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))
train_ds, val_ds = random_split(
    ds, [800, 200],
    generator=torch.Generator().manual_seed(42),
)
print(len(train_ds), len(val_ds))`,
    result: "800 200",
    interviewNote:
      "Pass an explicit generator so the split is reproducible. Fit normalisation statistics on the TRAIN split only — computing them before splitting leaks validation information.",
    importance: "medium",
    tags: ["TensorDataset", "random_split", "train test split", "leakage"],
  },
  {
    id: "cs-batching-concepts",
    topic: "data",
    section: "Concepts",
    title: "Batch, epoch, iteration",
    description:
      "One batch = one forward/backward/step. One epoch = one pass over the dataset = len(dataset) / batch_size iterations.",
    syntax: "steps_per_epoch = ceil(len(dataset) / batch_size)",
    example: `# 50,000 samples, batch_size=64
# -> 782 iterations per epoch (781 full + 1 partial)
# -> with drop_last=True: 781 iterations`,
    interviewNote:
      "Why mini-batches at all? Full-batch gradients are expensive and memory-bound; single samples give noisy gradients and poor hardware utilisation. Mini-batches trade off gradient quality, memory and parallelism — and the noise itself acts as a regulariser.",
    importance: "high",
    tags: ["batch", "epoch", "iteration", "mini-batch", "concept"],
    relatedExercises: ["ex-data-9", "ex-data-10"],
  },
];
