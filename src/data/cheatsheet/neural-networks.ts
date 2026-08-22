import type { CheatSheetEntry } from "@/lib/types";

export const nnEntries: CheatSheetEntry[] = [
  {
    id: "cs-module",
    topic: "nn",
    section: "nn.Module",
    title: "Defining a model",
    description:
      "Subclass nn.Module, register layers in __init__, describe the data flow in forward(). PyTorch discovers parameters by walking the attributes you assign.",
    syntax: "class Net(nn.Module): def __init__(self) / def forward(self, x)",
    example: `class NeuralNetwork(nn.Module):
    def __init__(self, in_features, hidden, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = NeuralNetwork(784, 128, 10)
print(model(torch.randn(4, 784)).shape)`,
    result: "torch.Size([4, 10])",
    interviewNote:
      "super().__init__() must be first — it initialises the internal parameter/buffer registries, and without it assigning layers raises an AttributeError.",
    importance: "high",
    tags: ["nn.Module", "forward", "super", "model", "class"],
    relatedExercises: ["ex-nn-1", "ex-nn-2", "ex-nn-3"],
  },
  {
    id: "cs-call-model",
    topic: "nn",
    section: "nn.Module",
    title: "Call the model, don't call forward()",
    description:
      "model(x) runs __call__, which fires hooks and module bookkeeping before delegating to forward(). model.forward(x) skips all of that.",
    syntax: "pred = model(X)   # not model.forward(X)",
    example: `pred = model(X)          # correct
# pred = model.forward(X)  # skips hooks — avoid`,
    interviewNote:
      "A small detail interviewers use to check whether you have actually read PyTorch source. Hooks (used by profilers, quantisation, feature extraction) only run through __call__.",
    importance: "medium",
    tags: ["forward", "__call__", "hooks", "model"],
  },
  {
    id: "cs-parameters",
    topic: "nn",
    section: "nn.Module",
    title: "parameters() · named_parameters() · nn.Parameter",
    description:
      "parameters() yields every learnable tensor, recursively. Wrapping a tensor in nn.Parameter is what registers it (and sets requires_grad=True).",
    syntax: "model.parameters()  ·  model.named_parameters()  ·  nn.Parameter(torch.zeros(4))",
    example: `total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total, trainable)
for name, p in model.named_parameters():
    print(name, tuple(p.shape))`,
    interviewNote:
      "A plain tensor attribute is NOT a parameter — it will not be trained, moved by .to(device) or saved in the state_dict. Use nn.Parameter for learnables and register_buffer for non-learnable state such as running stats or a positional-encoding table.",
    importance: "high",
    tags: ["parameters", "nn.Parameter", "buffer", "trainable", "count"],
    relatedExercises: ["ex-nn-4", "ex-nn-5"],
  },
  {
    id: "cs-sequential",
    topic: "nn",
    section: "nn.Module",
    title: "nn.Sequential",
    description:
      "Chains modules in order. Great for straight-line stacks, useless when forward() needs branching, skip connections or multiple inputs.",
    syntax: "nn.Sequential(layer1, layer2, ...)",
    example: `model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10),
)
print(model(torch.randn(4, 1, 28, 28)).shape)`,
    result: "torch.Size([4, 10])",
    interviewNote:
      "Use nn.ModuleList when you need a list of layers you index yourself, and nn.ModuleDict for named branches. A plain Python list of layers is invisible to parameters().",
    importance: "high",
    tags: ["Sequential", "ModuleList", "stack", "mlp"],
    relatedExercises: ["ex-nn-6"],
  },
  {
    id: "cs-linear",
    topic: "nn",
    section: "Layers",
    title: "nn.Linear",
    description:
      "Applies y = xWᵀ + b to the LAST dimension. Every other dimension is treated as batch.",
    syntax: "nn.Linear(in_features, out_features, bias=True)",
    example: `fc = nn.Linear(128, 10)
print(fc(torch.randn(32, 128)).shape)      # tabular batch
print(fc(torch.randn(8, 100, 128)).shape)  # (B, T, C) — works too
print(fc.weight.shape, fc.bias.shape)`,
    result: `torch.Size([32, 10])
torch.Size([8, 100, 10])
torch.Size([10, 128]) torch.Size([10])`,
    interviewNote:
      "Note the weight is stored as (out_features, in_features) — the transpose of what the formula suggests. Because Linear only touches the last dim, it applies position-wise in transformers without any reshaping.",
    importance: "high",
    tags: ["Linear", "dense", "fc", "weight shape", "last dimension"],
    relatedExercises: ["ex-nn-7", "ex-nn-8", "ex-shape-16"],
  },
  {
    id: "cs-relu",
    topic: "nn",
    section: "Activations",
    title: "ReLU · LeakyReLU · GELU",
    description:
      "ReLU zeroes negatives — cheap and the CNN default. LeakyReLU keeps a small negative slope to avoid dead units. GELU is smooth and is the transformer standard.",
    syntax: "nn.ReLU()  ·  nn.LeakyReLU(0.01)  ·  nn.GELU()  ·  F.relu(x)",
    example: `x = torch.tensor([-2.0, 0.0, 2.0])
print(F.relu(x))
print(F.leaky_relu(x, 0.1))
print(F.gelu(x))`,
    result: `tensor([0., 0., 2.])
tensor([-0.2000,  0.0000,  2.0000])
tensor([-0.0455,  0.0000,  1.9545])`,
    interviewNote:
      "Dying ReLU: a unit stuck in the negative region gets zero gradient forever, so it never recovers. LeakyReLU/GELU keep a non-zero gradient there. GELU is used by BERT, GPT and most modern LLMs.",
    importance: "high",
    tags: ["relu", "gelu", "leaky", "activation", "dying relu"],
    relatedExercises: ["ex-nn-9"],
  },
  {
    id: "cs-sigmoid-tanh",
    topic: "nn",
    section: "Activations",
    title: "Sigmoid · Tanh · Softmax",
    description:
      "Sigmoid squashes to (0, 1) for independent probabilities. Tanh maps to (-1, 1) and is zero-centred. Softmax produces a distribution over one axis.",
    syntax: "torch.sigmoid(x)  ·  torch.tanh(x)  ·  torch.softmax(x, dim=1)",
    example: `logits = torch.randn(4, 1)
probs = torch.sigmoid(logits)        # binary
multi = torch.softmax(torch.randn(4, 3), dim=1)
print(probs.shape, multi.sum(dim=1))`,
    result: "torch.Size([4, 1]) tensor([1., 1., 1.])",
    interviewNote:
      "Both saturate: far from 0 their gradient is ~0, which is the classic vanishing-gradient story for deep sigmoid/tanh stacks. Sigmoid = multi-LABEL (independent), softmax = multi-CLASS (mutually exclusive).",
    importance: "high",
    tags: ["sigmoid", "tanh", "softmax", "vanishing gradient", "probability"],
    relatedExercises: ["ex-nn-10"],
  },
  {
    id: "cs-dropout",
    topic: "nn",
    section: "Regularisation",
    title: "nn.Dropout",
    description:
      "During training, zeroes each activation with probability p and scales the rest by 1/(1-p). During eval it is a no-op.",
    syntax: "nn.Dropout(p=0.5)",
    example: `drop = nn.Dropout(0.5)
drop.train()
print(drop(torch.ones(6)))
drop.eval()
print(drop(torch.ones(6)))`,
    result: `tensor([0., 2., 2., 0., 2., 0.])
tensor([1., 1., 1., 1., 1., 1.])`,
    interviewNote:
      "The 1/(1-p) scaling at training time (\"inverted dropout\") is what makes inference need no rescaling. This behaviour switch is exactly why forgetting model.eval() makes validation results noisy and worse.",
    importance: "high",
    tags: ["dropout", "regularisation", "train", "eval", "inverted dropout"],
    relatedExercises: ["ex-nn-11", "ex-eval-1"],
  },
  {
    id: "cs-batchnorm",
    topic: "nn",
    section: "Regularisation",
    title: "nn.BatchNorm1d / BatchNorm2d",
    description:
      "Normalises each feature/channel using the mean and variance of the current BATCH, then applies a learnable scale and shift. Keeps running statistics for eval.",
    syntax: "nn.BatchNorm1d(num_features)  ·  nn.BatchNorm2d(num_channels)",
    example: `bn = nn.BatchNorm2d(16)
x = torch.randn(8, 16, 32, 32)
print(bn(x).shape)
print(bn.running_mean.shape)`,
    result: `torch.Size([8, 16, 32, 32])
torch.Size([16])`,
    interviewNote:
      "Batch-dependent: it behaves differently in train (batch stats) and eval (running stats), and degrades badly with very small batches. Bias in the preceding conv/linear is redundant when BatchNorm follows.",
    importance: "high",
    tags: ["batchnorm", "normalisation", "running stats", "train eval"],
    relatedExercises: ["ex-nn-12"],
  },
  {
    id: "cs-layernorm",
    topic: "nn",
    section: "Regularisation",
    title: "nn.LayerNorm",
    description:
      "Normalises across the FEATURE dimension of each sample independently. No batch statistics, no running buffers, identical in train and eval.",
    syntax: "nn.LayerNorm(normalized_shape)",
    example: `ln = nn.LayerNorm(512)
x = torch.randn(8, 100, 512)   # (B, T, C)
print(ln(x).shape)`,
    result: "torch.Size([8, 100, 512])",
    interviewNote:
      "Why transformers use LayerNorm: sequence lengths vary, batches are small relative to model size, and inference is often batch-size 1 — all things BatchNorm handles badly. LayerNorm is per-token, so none of that matters.",
    importance: "high",
    tags: ["layernorm", "transformer", "normalisation", "batchnorm comparison"],
    relatedExercises: ["ex-nn-13", "ex-tf-3"],
  },
  {
    id: "cs-embedding",
    topic: "nn",
    section: "Layers",
    title: "nn.Embedding",
    description:
      "A lookup table: maps integer ids to dense vectors. Mathematically a one-hot matmul, implemented as an indexing op.",
    syntax: "nn.Embedding(num_embeddings, embedding_dim, padding_idx=None)",
    example: `emb = nn.Embedding(10000, 256)
ids = torch.randint(0, 10000, (8, 100))   # (B, T) int64
print(emb(ids).shape)                     # (B, T, C)`,
    result: "torch.Size([8, 100, 256])",
    interviewNote:
      "Input MUST be int64 — float ids raise \"Expected tensor for argument #1 'indices' to have scalar type Long\". padding_idx keeps the pad vector fixed at zero and excluded from gradients.",
    importance: "high",
    tags: ["embedding", "lookup", "nlp", "int64", "padding_idx"],
    relatedExercises: ["ex-nn-14", "ex-tf-1"],
  },
  {
    id: "cs-conv2d",
    topic: "nn",
    section: "Layers",
    title: "nn.Conv2d",
    description:
      "Slides learnable kernels over a (B, C, H, W) input. in_channels must match the incoming channel count; out_channels is how many filters you learn.",
    syntax: "nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)",
    example: `conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
x = torch.randn(8, 3, 32, 32)
print(conv(x).shape)
print(conv.weight.shape)   # (out_c, in_c, kH, kW)`,
    result: `torch.Size([8, 16, 32, 32])
torch.Size([16, 3, 3, 3])`,
    interviewNote:
      "Parameter count = out_c × in_c × kH × kW + out_c. Conv layers are translation-equivariant and share weights across positions — that is the whole reason they beat MLPs on images.",
    importance: "high",
    tags: ["conv2d", "cnn", "channels", "kernel", "padding"],
    relatedExercises: ["ex-nn-15", "ex-shape-14"],
  },
  {
    id: "cs-conv1d",
    topic: "nn",
    section: "Layers",
    title: "nn.Conv1d",
    description:
      "The 1-D analogue for sequences and time series. Expects (B, C, L) — channels before length, which trips people up after working with (B, T, C) transformers.",
    syntax: "nn.Conv1d(in_channels, out_channels, kernel_size)",
    example: `conv = nn.Conv1d(64, 128, kernel_size=3, padding=1)
x = torch.randn(8, 64, 100)    # (B, C, L)
print(conv(x).shape)`,
    result: "torch.Size([8, 128, 100])",
    interviewNote:
      "Going from a transformer layout to Conv1d needs x.transpose(1, 2). Getting this backwards convolves over the feature axis and quietly ruins the model.",
    importance: "medium",
    tags: ["conv1d", "sequence", "transpose", "channels first"],
  },
  {
    id: "cs-pooling",
    topic: "nn",
    section: "Layers",
    title: "MaxPool2d & AdaptiveAvgPool2d",
    description:
      "MaxPool2d(2) halves the spatial dimensions. AdaptiveAvgPool2d((1, 1)) reduces any H×W to a fixed size, making the head input-resolution independent.",
    syntax: "nn.MaxPool2d(kernel_size)  ·  nn.AdaptiveAvgPool2d(output_size)",
    example: `x = torch.randn(8, 16, 32, 32)
print(nn.MaxPool2d(2)(x).shape)
print(nn.AdaptiveAvgPool2d((1, 1))(x).shape)`,
    result: `torch.Size([8, 16, 16, 16])
torch.Size([8, 16, 1, 1])`,
    interviewNote:
      "Adaptive pooling is why ResNet accepts arbitrary input sizes: it guarantees the flattened vector fed to the final Linear is always the same length. Pooling layers have no parameters.",
    importance: "medium",
    tags: ["maxpool", "adaptive", "pooling", "resnet", "global pooling"],
  },
  {
    id: "cs-rnn",
    topic: "nn",
    section: "Layers",
    title: "nn.LSTM / nn.GRU",
    description:
      "Recurrent layers that return (output, hidden). By default they expect (T, B, C); pass batch_first=True to use the (B, T, C) layout everything else uses.",
    syntax: "nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)",
    example: `lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True)
x = torch.randn(8, 100, 64)
out, (h, c) = lstm(x)
print(out.shape, h.shape)`,
    result: "torch.Size([8, 100, 128]) torch.Size([2, 8, 128])",
    interviewNote:
      "out holds every timestep, h only the last one per layer. GRU is the same idea with 2 gates instead of 3 — fewer parameters, usually comparable quality. Forgetting batch_first is a top-5 source of silent RNN bugs.",
    importance: "medium",
    tags: ["lstm", "gru", "rnn", "batch_first", "hidden state"],
  },
  {
    id: "cs-mha",
    topic: "transformers",
    section: "Attention",
    title: "nn.MultiheadAttention",
    description:
      "Runs scaled dot-product attention with several heads in parallel. Returns the attended output and (optionally) the attention weights.",
    syntax: "nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)",
    example: `mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
x = torch.randn(8, 100, 512)
out, weights = mha(x, x, x)          # self-attention: q = k = v
print(out.shape, weights.shape)`,
    result: "torch.Size([8, 100, 512]) torch.Size([8, 100, 100])",
    interviewNote:
      "embed_dim must be divisible by num_heads: each head works on embed_dim // num_heads channels. Passing the same tensor three times is self-attention; different tensors for query vs key/value is cross-attention.",
    importance: "high",
    tags: ["attention", "multihead", "transformer", "self-attention", "qkv"],
    relatedExercises: ["ex-tf-2", "ex-tf-4"],
  },
  {
    id: "cs-attention-math",
    topic: "transformers",
    section: "Attention",
    title: "Scaled dot-product attention by hand",
    description:
      "softmax(QKᵀ / √d) V. The scaling keeps the dot products small enough that softmax does not saturate.",
    syntax: "torch.softmax(Q @ K.transpose(-2, -1) / d ** 0.5, dim=-1) @ V",
    example: `Q = K = V = torch.randn(8, 100, 64)
scores = Q @ K.transpose(-2, -1) / (64 ** 0.5)   # (B, T, T)
attn = torch.softmax(scores, dim=-1)
out = attn @ V
print(scores.shape, out.shape)`,
    result: "torch.Size([8, 100, 100]) torch.Size([8, 100, 64])",
    interviewNote:
      "Without the √d scaling the variance of the dot products grows with d, softmax saturates and gradients vanish. Note dim=-1: each query row must sum to 1 across keys. F.scaled_dot_product_attention is the fused, memory-efficient version.",
    importance: "high",
    tags: ["attention", "qkv", "softmax", "scaling", "transformer"],
    relatedExercises: ["ex-tf-5", "ex-tf-6"],
  },
  {
    id: "cs-transformer-block",
    topic: "transformers",
    section: "Transformer Blocks",
    title: "Anatomy of a transformer block",
    description:
      "Attention and a feed-forward network, each wrapped in a residual connection and a LayerNorm. Pre-norm (norm before the sublayer) is the modern default because it trains more stably.",
    syntax: "x = x + attn(norm(x))  ·  x = x + ff(norm(x))",
    example: `class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h)[0]
        return x + self.ff(self.ln2(x))`,
    interviewNote:
      "Know the three details: the residual stream keeps gradients flowing through depth, the FFN expands by 4x by convention, and pre-norm (vs the original post-norm) is why modern LLMs train without a warmup-heavy schedule.",
    importance: "high",
    tags: ["transformer", "residual", "layernorm", "ffn", "pre-norm"],
    relatedExercises: ["ex-tf-7"],
  },
  {
    id: "cs-positional",
    topic: "transformers",
    section: "Transformer Blocks",
    title: "Positional information",
    description:
      "Attention is permutation-invariant, so position must be injected. The simplest approach is a learned nn.Embedding over positions, added to the token embeddings.",
    syntax: "tok_emb(idx) + pos_emb(torch.arange(T, device=idx.device))",
    example: `T = 100
tok = nn.Embedding(10000, 512)
pos = nn.Embedding(T, 512)
idx = torch.randint(0, 10000, (8, T))
x = tok(idx) + pos(torch.arange(T, device=idx.device))
print(x.shape)`,
    result: "torch.Size([8, 100, 512])",
    interviewNote:
      "Note the broadcast: (T, C) positions add to (B, T, C) tokens. Sinusoidal encodings extrapolate to unseen lengths; learned ones are simpler; RoPE (rotary) is what most current LLMs use.",
    importance: "medium",
    tags: ["positional", "embedding", "arange", "rope", "transformer"],
  },
  {
    id: "cs-weight-init",
    topic: "nn",
    section: "nn.Module",
    title: "Weight initialisation",
    description:
      "Layers ship with sensible defaults (Kaiming-uniform for Linear/Conv). Override with torch.nn.init when a paper specifies something else.",
    syntax: "nn.init.kaiming_normal_(w, nonlinearity='relu')  ·  model.apply(fn)",
    example: `def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)

model.apply(init_weights)`,
    interviewNote:
      "Why not zeros? Identical weights make every unit compute the same thing and receive the same gradient — the network never breaks symmetry. Kaiming for ReLU nets, Xavier for tanh/sigmoid.",
    importance: "medium",
    tags: ["init", "kaiming", "xavier", "symmetry", "apply"],
  },
];
