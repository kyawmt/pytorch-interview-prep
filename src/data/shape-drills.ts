import type { Exercise } from "@/lib/types";

/**
 * Tensor Shape Playground: shape-reasoning drills generated from a fixed table
 * of (starting shape, operation, result) triples. Kept separate from the main
 * exercise bank so the playground can be reshuffled endlessly without
 * polluting progress statistics for the graded exercises.
 */
interface ShapeDrill {
  id: string;
  start: string;
  operation: string;
  result: string;
  why: string;
  difficulty: "easy" | "medium" | "hard";
}

export const SHAPE_DRILLS: ShapeDrill[] = [
  { id: "sd-1", start: "(32, 3, 64, 64)", operation: "x.flatten(1)", result: "(32, 12288)", why: "3 × 64 × 64 = 12288; the batch dimension is preserved.", difficulty: "easy" },
  { id: "sd-2", start: "(8, 100, 512)", operation: "x.transpose(1, 2)", result: "(8, 512, 100)", why: "transpose swaps exactly the two named axes.", difficulty: "easy" },
  { id: "sd-3", start: "(32, 224, 224, 3)", operation: "x.permute(0, 3, 1, 2)", result: "(32, 3, 224, 224)", why: "Output dim i takes source dim dims[i]: NHWC → NCHW.", difficulty: "medium" },
  { id: "sd-4", start: "(4, 1, 6)", operation: "x.squeeze()", result: "(4, 6)", why: "squeeze with no argument removes every size-1 dimension.", difficulty: "easy" },
  { id: "sd-5", start: "(3, 224, 224)", operation: "x.unsqueeze(0)", result: "(1, 3, 224, 224)", why: "unsqueeze inserts a size-1 dimension at the given index.", difficulty: "easy" },
  { id: "sd-6", start: "(6, 4)", operation: "x.reshape(-1)", result: "(24,)", why: "-1 is inferred from the element count: 6 × 4 = 24.", difficulty: "easy" },
  { id: "sd-7", start: "(2, 3, 4)", operation: "x.reshape(2, -1)", result: "(2, 12)", why: "The batch dimension is fixed; the rest collapses to 3 × 4 = 12.", difficulty: "easy" },
  { id: "sd-8", start: "(32, 10)", operation: "x.mean(dim=1)", result: "(32,)", why: "The named dimension disappears in a reduction.", difficulty: "easy" },
  { id: "sd-9", start: "(32, 10)", operation: "x.sum(dim=1, keepdim=True)", result: "(32, 1)", why: "keepdim=True retains the reduced axis with size 1.", difficulty: "medium" },
  { id: "sd-10", start: "(8, 3, 32, 32)", operation: "x.mean(dim=(2, 3))", result: "(8, 3)", why: "Both spatial dimensions are reduced — global average pooling.", difficulty: "medium" },
  { id: "sd-11", start: "(4, 10)", operation: "x.argmax(dim=1)", result: "(4,)", why: "argmax reduces the class axis to a single index per row.", difficulty: "easy" },
  { id: "sd-12", start: "(2, 3) and (2, 3)", operation: "torch.cat([a, b], dim=0)", result: "(4, 3)", why: "cat sums the sizes along the concatenation axis.", difficulty: "easy" },
  { id: "sd-13", start: "(2, 3) and (2, 3)", operation: "torch.stack([a, b])", result: "(2, 2, 3)", why: "stack inserts a new leading dimension of size 2.", difficulty: "medium" },
  { id: "sd-14", start: "(4, 3, 8) and (4, 5, 8)", operation: "torch.cat([a, b], dim=1)", result: "(4, 8, 8)", why: "Only dim 1 changes: 3 + 5 = 8.", difficulty: "medium" },
  { id: "sd-15", start: "(3, 4) and (4,)", operation: "a + b", result: "(3, 4)", why: "b right-aligns to (1, 4) and repeats across rows.", difficulty: "easy" },
  { id: "sd-16", start: "(5, 1) and (1, 7)", operation: "a * b", result: "(5, 7)", why: "Both size-1 dimensions expand — an outer product.", difficulty: "medium" },
  { id: "sd-17", start: "(32,) and (32, 1)", operation: "a - b", result: "(32, 32)", why: "(1, 32) against (32, 1) broadcasts to (32, 32) — the classic silent loss bug.", difficulty: "hard" },
  { id: "sd-18", start: "(32, 128)", operation: "x @ torch.randn(128, 10)", result: "(32, 10)", why: "The inner dimensions contract: (n, k) @ (k, m) → (n, m).", difficulty: "easy" },
  { id: "sd-19", start: "(8, 100, 64)", operation: "torch.bmm(x, x.transpose(1, 2))", result: "(8, 100, 100)", why: "Batched (100, 64) @ (64, 100) — the QKᵀ step of attention.", difficulty: "medium" },
  { id: "sd-20", start: "(8, 100, 512)", operation: "nn.Linear(512, 10)(x)", result: "(8, 100, 10)", why: "Linear maps only the last dimension; leading dimensions are batch.", difficulty: "medium" },
  { id: "sd-21", start: "(8, 3, 32, 32)", operation: "nn.Conv2d(3, 16, 3, padding=1)(x)", result: "(8, 16, 32, 32)", why: "(32 + 2 - 3)/1 + 1 = 32; channels become out_channels.", difficulty: "medium" },
  { id: "sd-22", start: "(8, 3, 32, 32)", operation: "nn.Conv2d(3, 16, 3)(x)", result: "(8, 16, 30, 30)", why: "Without padding: (32 - 3)/1 + 1 = 30.", difficulty: "medium" },
  { id: "sd-23", start: "(8, 16, 32, 32)", operation: "nn.MaxPool2d(2)(x)", result: "(8, 16, 16, 16)", why: "Kernel 2 with stride 2 halves both spatial dimensions.", difficulty: "easy" },
  { id: "sd-24", start: "(8, 3, 32, 32)", operation: "nn.Conv2d(3, 16, 3, stride=2, padding=1)(x)", result: "(8, 16, 16, 16)", why: "(32 + 2 - 3)/2 + 1 = 16 (integer division).", difficulty: "hard" },
  { id: "sd-25", start: "(8, 16, 7, 7)", operation: "nn.AdaptiveAvgPool2d((1, 1))(x)", result: "(8, 16, 1, 1)", why: "Adaptive pooling forces the requested output size whatever the input.", difficulty: "medium" },
  { id: "sd-26", start: "(8, 128)", operation: "nn.Embedding(1000, 64)(x.long())", result: "(8, 128, 64)", why: "Each id becomes a 64-dim vector, appending a dimension.", difficulty: "medium" },
  { id: "sd-27", start: "(16, 1, 28, 28)", operation: "nn.Flatten()(x)", result: "(16, 784)", why: "nn.Flatten defaults to start_dim=1: 1 × 28 × 28 = 784.", difficulty: "easy" },
  { id: "sd-28", start: "(8, 100, 192)", operation: "x.chunk(3, dim=-1)[0]", result: "(8, 100, 64)", why: "chunk splits the last dimension into three equal parts — the fused QKV trick.", difficulty: "hard" },
  { id: "sd-29", start: "(2, 3, 4)", operation: "x.permute(2, 0, 1)", result: "(4, 2, 3)", why: "Output dims take source dims 2, 0, 1 in that order.", difficulty: "hard" },
  { id: "sd-30", start: "(32, 10)", operation: "x[:, 0]", result: "(32,)", why: "An integer index removes that dimension; a slice would keep it.", difficulty: "easy" },
  { id: "sd-31", start: "(8, 3, 32, 32)", operation: "x[..., 0]", result: "(8, 3, 32)", why: "Ellipsis covers the leading dimensions; the last is indexed away.", difficulty: "medium" },
  { id: "sd-32", start: "(1, 3, 1, 5)", operation: "x.squeeze(2)", result: "(1, 3, 5)", why: "Only dimension 2 is removed; dimension 0 stays because it was not named.", difficulty: "hard" },
];

/** Converted to the standard Exercise shape so the playground can reuse the runner. */
export const SHAPE_DRILL_EXERCISES: Exercise[] = SHAPE_DRILLS.map((drill) => ({
  id: drill.id,
  title: drill.operation,
  topic: "shapes",
  level: 2,
  difficulty: drill.difficulty,
  importance: "high",
  type: "shape",
  question: "What is the resulting shape?",
  context: `x.shape == ${drill.start}\n${drill.operation}`,
  acceptedAnswers: [drill.result],
  hint: "Work through the dimensions one at a time, left to right.",
  solution: `torch.Size([${drill.result.replace(/[()]/g, "")}])`,
  explanation: drill.why,
  tags: ["shape", "playground"],
}));
