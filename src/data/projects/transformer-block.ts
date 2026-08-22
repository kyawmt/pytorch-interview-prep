import type { MiniProject } from "@/lib/types";

export const transformerBlock: MiniProject = {
  slug: "transformer-block",
  title: "Transformer Building Blocks",
  tagline: "Q/K/V, scaled dot-product attention, residuals and LayerNorm — the interview subset.",
  difficulty: "hard",
  estimatedMinutes: 35,
  topics: ["transformers", "nn", "shapes", "math"],
  outline: ["Embeddings", "Positions", "Q/K/V", "Attention", "Multi-head", "Residual + Norm", "FFN"],
  overview:
    "Assemble a transformer block piece by piece. The goal is not a full LLM — it is being able to explain and write the parts an interviewer will ask about.",
  steps: [
    {
      id: "tb-1",
      title: "Token embeddings",
      brief: "Create the token embedding for a 50,000-token vocabulary and d_model = 512.",
      kind: "code",
      acceptedAnswers: [
        "self.tok_emb = nn.Embedding(50000, 512)",
        "tok_emb = nn.Embedding(50000, 512)",
        "self.emb = nn.Embedding(50000, 512)",
      ],
      requiredPatterns: ["nn\\.Embedding\\(50000,512\\)"],
      hint: "Vocabulary size first, model dimension second.",
      solution: "self.tok_emb = nn.Embedding(50000, 512)",
      explanation: "Turns (B, T) ids into (B, T, 512) hidden states — the residual stream the whole block operates on.",
    },
    {
      id: "tb-2",
      title: "Positional information",
      brief:
        "Attention is permutation-invariant. Add learned positional embeddings to the token embeddings (assume `self.pos_emb` exists and T is the sequence length).",
      kind: "code",
      context: "x = self.tok_emb(idx)      # (B, T, C)",
      acceptedAnswers: [
        "x = x + self.pos_emb(torch.arange(T, device=idx.device))",
        "x = x + self.pos_emb(torch.arange(T))",
        "x = x + self.pos_emb(torch.arange(T, device=x.device))",
      ],
      requiredPatterns: ["pos_emb", "arange"],
      hint: "Build the position indices with torch.arange and let broadcasting handle the batch.",
      solution: "x = x + self.pos_emb(torch.arange(T, device=idx.device))",
      explanation:
        "(T, C) positions broadcast against (B, T, C) tokens. Passing device= keeps it working on GPU — creating the arange on the CPU is a classic device-mismatch bug.",
    },
    {
      id: "tb-3",
      title: "Project to Q, K, V",
      brief: "Create the three linear projections from d_model=512 to head_dim=64 (three lines).",
      kind: "code",
      requiredPatterns: ["nn\\.Linear\\(512,64\\)"],
      hint: "Each is an nn.Linear(512, 64).",
      solution: `self.q = nn.Linear(512, 64)
self.k = nn.Linear(512, 64)
self.v = nn.Linear(512, 64)`,
      explanation:
        "Queries ask, keys advertise, values carry the content. Production code fuses these into one Linear(512, 192) and splits with chunk(3, dim=-1).",
    },
    {
      id: "tb-4",
      title: "Attention scores",
      brief:
        "Compute the scaled scores from Q (B, T, 64) and K (B, T, 64), including the 1/√d_k scaling.",
      kind: "code",
      acceptedAnswers: [
        "scores = Q @ K.transpose(-2, -1) / (64 ** 0.5)",
        "scores = Q @ K.transpose(-2, -1) / math.sqrt(64)",
        "scores = (Q @ K.transpose(-2, -1)) / (64 ** 0.5)",
        "scores = Q @ K.transpose(-1, -2) / (64 ** 0.5)",
      ],
      requiredPatterns: ["(@|matmul)", "transpose\\(-", "(0\\.5|sqrt)"],
      hint: "Q @ Kᵀ, then divide by the square root of the head dimension.",
      solution: "scores = Q @ K.transpose(-2, -1) / (64 ** 0.5)",
      explanation:
        "The result is (B, T, T): how much each query attends to each key. Without the scaling, the dot products' variance grows with d_k, softmax saturates and gradients vanish.",
    },
    {
      id: "tb-5",
      title: "Attention weights",
      brief: "Turn `scores` (B, T, T) into attention weights that sum to 1 over the keys.",
      kind: "code",
      acceptedAnswers: [
        "attn = torch.softmax(scores, dim=-1)",
        "attn = F.softmax(scores, dim=-1)",
        "attn = torch.softmax(scores, dim=2)",
        "attn = scores.softmax(dim=-1)",
      ],
      requiredPatterns: ["softmax", "dim=(-1|2)"],
      hint: "The KEY axis is the last one.",
      solution: "attn = torch.softmax(scores, dim=-1)",
      explanation:
        "dim=-1 makes each query's row a probability distribution over the keys. Softmaxing over dim=-2 would normalise the wrong way and quietly break the model.",
    },
    {
      id: "tb-6",
      title: "Attention output",
      brief: "Apply the weights to V (B, T, 64).",
      kind: "code",
      acceptedAnswers: ["out = attn @ V", "out = torch.matmul(attn, V)", "out = attn.matmul(V)"],
      requiredPatterns: ["attn(@|\\.matmul\\(|,)"],
      hint: "(B, T, T) @ (B, T, 64).",
      solution: "out = attn @ V",
      explanation: "Each output row is a weighted average of value vectors — the shape returns to (B, T, 64).",
    },
    {
      id: "tb-7",
      title: "Multi-head split",
      brief: "With d_model = 512 and 8 heads, what dimension does each head operate on?",
      kind: "multiple-choice",
      options: ["512", "64", "8", "4096"],
      correctOption: 1,
      hint: "d_model // num_heads.",
      solution: "nn.MultiheadAttention(512, 8, batch_first=True)   # head_dim = 64",
      explanation:
        "The heads run in parallel on 64-dim subspaces and are concatenated back to 512. Total compute is roughly the same as one 512-dim head, but different heads can specialise.",
    },
    {
      id: "tb-8",
      title: "Residual + norm",
      brief:
        "Write the pre-norm attention sublayer: normalise, attend, and add the residual (`self.ln1`, `self.attn`).",
      kind: "code",
      acceptedAnswers: [
        "h = self.ln1(x)\nx = x + self.attn(h, h, h)[0]",
        "x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]",
      ],
      requiredPatterns: ["ln1", "attn", "x=x\\+"],
      hint: "x = x + sublayer(norm(x)).",
      solution: `h = self.ln1(x)
x = x + self.attn(h, h, h)[0]`,
      explanation:
        "The residual keeps a clean gradient path through depth; pre-norm (norm on the sublayer INPUT) is far more stable at depth than the original post-norm design. Note the [0] — MultiheadAttention returns (output, weights).",
    },
    {
      id: "tb-9",
      title: "Feed-forward network",
      brief: "Build the position-wise FFN for d_model=512 with the conventional 4× expansion.",
      kind: "code",
      requiredPatterns: ["nn\\.Sequential", "nn\\.Linear\\(512,2048\\)", "(GELU|ReLU)", "nn\\.Linear\\(2048,512\\)"],
      hint: "Linear up, activation, Linear back down.",
      solution: `self.ff = nn.Sequential(
    nn.Linear(512, 2048),
    nn.GELU(),
    nn.Linear(2048, 512),
)`,
      explanation:
        "Attention mixes information across positions; the FFN transforms each position independently. The 4× expansion is convention, and GELU is the modern activation choice.",
    },
  ],
};
