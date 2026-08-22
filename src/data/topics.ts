import type { Topic, TopicId } from "@/lib/types";

/**
 * Canonical topic list. Cheat-sheet entries, exercises and interview questions
 * all key off these ids, which is what makes cross-linking (and the learning
 * path) possible without a database.
 */
export const TOPICS: Topic[] = [
  {
    id: "tensors",
    title: "Tensor Basics",
    summary: "Creating tensors, dtypes, devices and the properties you read constantly.",
    importance: "high",
    pathOrder: 1,
    icon: "▦",
  },
  {
    id: "shapes",
    title: "Tensor Manipulation",
    summary: "reshape, view, permute, squeeze, indexing, cat vs stack.",
    importance: "high",
    pathOrder: 2,
    icon: "⤢",
  },
  {
    id: "broadcasting",
    title: "Broadcasting",
    summary: "The alignment rules behind silent shape bugs.",
    importance: "high",
    pathOrder: 3,
    icon: "⇔",
  },
  {
    id: "math",
    title: "Math & Matrix Ops",
    summary: "Reductions, argmax, matmul, mm, bmm and when each applies.",
    importance: "medium",
    pathOrder: 4,
    icon: "∑",
  },
  {
    id: "autograd",
    title: "Autograd",
    summary: "requires_grad, backward, .grad, no_grad, detach, the graph.",
    importance: "high",
    pathOrder: 5,
    icon: "∂",
  },
  {
    id: "nn",
    title: "nn.Module & Layers",
    summary: "Model classes, Linear, Conv2d, normalization, dropout, Sequential.",
    importance: "high",
    pathOrder: 6,
    icon: "◫",
  },
  {
    id: "losses",
    title: "Loss Functions",
    summary: "MSE, L1, CrossEntropy, BCE, BCEWithLogits — and logits vs probabilities.",
    importance: "high",
    pathOrder: 7,
    icon: "◈",
  },
  {
    id: "optimizers",
    title: "Optimizers & Schedulers",
    summary: "SGD, Adam, AdamW, learning rate, weight decay, LR schedules.",
    importance: "high",
    pathOrder: 8,
    icon: "↘",
  },
  {
    id: "training",
    title: "Training Loop",
    summary: "The five lines every interviewer asks you to write from memory.",
    importance: "high",
    pathOrder: 9,
    icon: "↻",
  },
  {
    id: "evaluation",
    title: "Evaluation & Inference",
    summary: "model.eval(), torch.no_grad(), inference_mode, metrics.",
    importance: "high",
    pathOrder: 10,
    icon: "◉",
  },
  {
    id: "data",
    title: "Dataset & DataLoader",
    summary: "Custom datasets, batching, shuffling, workers, pinned memory.",
    importance: "high",
    pathOrder: 11,
    icon: "▤",
  },
  {
    id: "gpu",
    title: "Devices & GPU",
    summary: "cuda / mps / cpu, .to(device), device-mismatch errors, memory.",
    importance: "high",
    pathOrder: 12,
    icon: "⚙",
  },
  {
    id: "saving",
    title: "Saving & Loading",
    summary: "state_dict, checkpoints, resuming training, reproducibility.",
    importance: "medium",
    pathOrder: 13,
    icon: "⤓",
  },
  {
    id: "performance",
    title: "Performance & AMP",
    summary: "Mixed precision, gradient accumulation, clipping, throughput.",
    importance: "medium",
    pathOrder: 14,
    icon: "⚡",
  },
  {
    id: "transformers",
    title: "Transformer Blocks",
    summary: "Embeddings, Q/K/V, MultiheadAttention, LayerNorm, residuals.",
    importance: "high",
    pathOrder: 15,
    icon: "◇",
  },
  {
    id: "debugging",
    title: "Debugging",
    summary: "Device mismatch, shape errors, NaN loss, OOM, dtype problems.",
    importance: "high",
    pathOrder: 16,
    icon: "⚑",
  },
  {
    id: "interop",
    title: "NumPy Interop",
    summary: "from_numpy, .numpy(), shared memory, and what PyTorch adds.",
    importance: "low",
    pathOrder: 17,
    icon: "⇄",
  },
];

export const TOPIC_MAP: Record<TopicId, Topic> = Object.fromEntries(
  TOPICS.map((topic) => [topic.id, topic]),
) as Record<TopicId, Topic>;

export function topicTitle(id: TopicId | string): string {
  return TOPIC_MAP[id as TopicId]?.title ?? id;
}

/** Progressive levels used by the Practice page (see prompt §29). */
export const LEVELS: { level: number; title: string; topics: TopicId[]; blurb: string }[] = [
  { level: 1, title: "Tensor Basics", topics: ["tensors", "interop"], blurb: "Create tensors, read their properties, control dtype." },
  { level: 2, title: "Tensor Manipulation", topics: ["shapes", "broadcasting", "math"], blurb: "Reshape, permute, index, combine and reduce." },
  { level: 3, title: "Autograd", topics: ["autograd"], blurb: "Gradients, the graph, no_grad and detach." },
  { level: 4, title: "Neural Networks", topics: ["nn", "losses"], blurb: "Modules, layers, activations and loss selection." },
  { level: 5, title: "Training", topics: ["training", "optimizers", "evaluation"], blurb: "The loop, optimizers, eval mode." },
  { level: 6, title: "Data Pipeline", topics: ["data"], blurb: "Datasets, DataLoaders, batching and collation." },
  { level: 7, title: "GPU & Performance", topics: ["gpu", "saving", "performance"], blurb: "Devices, checkpoints, mixed precision." },
  { level: 8, title: "Interview Challenge", topics: ["debugging", "transformers"], blurb: "Debugging scenarios and transformer building blocks." },
];

/** The recommended path shown on /progress. */
export const LEARNING_PATH: TopicId[] = [
  "tensors",
  "shapes",
  "autograd",
  "nn",
  "losses",
  "optimizers",
  "training",
  "data",
  "gpu",
  "saving",
  "transformers",
  "debugging",
];
