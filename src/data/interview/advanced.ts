import type { InterviewQuestion } from "@/lib/types";

export const advancedQuestions: InterviewQuestion[] = [
  {
    id: "iq-debug-1",
    category: "debugging",
    topic: "debugging",
    question: "How would you debug a NaN loss?",
    shortAnswer:
      "Work through a checklist: check the inputs and targets for NaN/inf, lower the learning rate, clip gradients, look for log/sqrt/division by zero, and switch to the fused *WithLogits losses. torch.autograd.set_detect_anomaly(True) pinpoints the op.",
    detailedAnswer:
      "First establish WHEN it happens. Immediately → the data or the loss formulation (unnormalised inputs, a NaN in the dataset, log(0)). After a few hundred steps → divergence from too high a learning rate, or exploding gradients in an RNN/transformer without clipping. In mixed precision → float16 overflow, so check the GradScaler is present or switch to bfloat16. Anomaly detection is slow but tells you exactly which backward op produced the NaN. A useful guard is `assert torch.isfinite(loss)` in the loop so you fail fast rather than training on garbage.",
    code: `torch.autograd.set_detect_anomaly(True)
assert torch.isfinite(loss), "NaN loss"
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`,
    importance: "high",
    followUps: ["What causes exploding gradients?", "How does AMP change this?"],
    tags: ["nan", "debug", "clipping", "stability"],
  },
  {
    id: "iq-debug-2",
    category: "debugging",
    topic: "debugging",
    question: "Your loss is not decreasing at all. How do you diagnose it?",
    shortAnswer:
      "Try to overfit a single batch. If the model cannot drive that loss to ~0, the bug is in the code (missing backward/step, parameters not in the optimizer, everything frozen). If it can, the problem is data, learning rate or capacity.",
    detailedAnswer:
      "The single-batch overfit test is the fastest discriminator between 'broken plumbing' and 'bad hyperparameters'. Plumbing checks: is optimizer.step() called; were the parameters passed to the optimizer BEFORE any layer was replaced; is anything left with requires_grad=False; are layers hidden in a plain Python list (invisible to parameters()); is the loss actually a function of the model output. If the plumbing is fine, check label alignment, input normalisation, and run an LR range test.",
    code: `X, y = next(iter(loader))
for _ in range(200):
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()
print(loss.item())   # should approach 0`,
    importance: "high",
    followUps: ["What if it overfits one batch but not the dataset?", "How do you check gradients are flowing?"],
    tags: ["debug", "sanity check", "training"],
  },
  {
    id: "iq-debug-3",
    category: "debugging",
    topic: "gpu",
    question: "How do you deal with CUDA out of memory?",
    shortAnswer:
      "If it fails on batch 1 the model or batch is too big — reduce the batch size, enable AMP, use gradient accumulation or checkpointing. If it fails after N batches, you have a leak: something is retaining graphs, usually accumulating loss tensors without .item().",
    detailedAnswer:
      "That timing distinction is the whole diagnosis. For genuine capacity limits: smaller batch + gradient accumulation to keep the effective batch size, autocast to halve activation memory, torch.utils.checkpoint to trade compute for memory, and a smaller model or shorter sequences. For leaks: use loss.item() when logging, detach anything you store, do not keep a Python list of output tensors, and detach RNN hidden states between chunks. torch.cuda.empty_cache() only returns cached blocks to the driver — it does not fix a leak.",
    code: `running_loss += loss.item()          # not: += loss
print(torch.cuda.max_memory_allocated() / 1e9)`,
    importance: "high",
    followUps: ["What is gradient checkpointing?", "Why does empty_cache() rarely help?"],
    tags: ["oom", "memory", "amp", "leak"],
  },
  {
    id: "iq-debug-4",
    category: "debugging",
    topic: "debugging",
    question: "What causes exploding gradients and how do you fix them?",
    shortAnswer:
      "Repeated multiplication by weights with magnitude > 1 through depth or time makes gradients grow exponentially. Fix with gradient clipping, a lower learning rate, normalisation layers and better initialisation.",
    detailedAnswer:
      "It is most common in RNNs, where the same weight matrix is applied at every timestep, and in deep networks without normalisation. Symptoms are a loss that spikes to inf/NaN and gradient norms that jump by orders of magnitude. clip_grad_norm_ rescales all gradients together so their combined L2 norm is bounded, which preserves direction — prefer it over clip_grad_value_, which clips element-wise and distorts the direction. Residual connections and LayerNorm are the architectural fixes.",
    code: `loss.backward()
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()`,
    importance: "medium",
    followUps: ["How is this different from vanishing gradients?", "clip_grad_norm_ vs clip_grad_value_?"],
    tags: ["exploding", "clipping", "rnn"],
  },
  {
    id: "iq-debug-5",
    category: "debugging",
    topic: "debugging",
    question: "What causes vanishing gradients?",
    shortAnswer:
      "Gradients shrink toward zero as they propagate back through many layers, typically because saturating activations (sigmoid/tanh) have near-zero derivative or weights are too small. Early layers stop learning.",
    detailedAnswer:
      "The chain rule multiplies many small factors, so the product decays exponentially with depth. Sigmoid's maximum derivative is 0.25, so a 10-layer sigmoid stack scales the gradient by at most 0.25^10. The modern fixes are all architectural: non-saturating activations (ReLU/GELU), residual connections that give gradients a direct path, normalisation layers keeping activations well-scaled, and Kaiming initialisation. LSTM/GRU gates were the RNN-era answer to the same problem.",
    code: `x = x + sublayer(norm(x))   # residual keeps the gradient path open`,
    importance: "medium",
    followUps: ["How do residual connections help exactly?", "Why did LSTMs help over vanilla RNNs?"],
    tags: ["vanishing", "residual", "activation"],
  },
  {
    id: "iq-debug-6",
    category: "debugging",
    topic: "debugging",
    question: "Training accuracy is high but validation accuracy is poor. What do you do?",
    shortAnswer:
      "That is overfitting: add regularisation (dropout, weight decay, augmentation), get more data, reduce capacity, or use early stopping. First rule out a data problem such as a distribution shift or leakage.",
    detailedAnswer:
      "Order of attack: (1) verify the split is sound — no duplicates across splits, no leakage, same preprocessing; (2) add data or augmentation, which is almost always the highest-leverage fix; (3) regularise: dropout, weight decay, label smoothing, early stopping on validation loss; (4) shrink the model or use a pretrained backbone. Plot both curves: a widening gap is classic overfitting, while both curves being bad is underfitting and needs the opposite treatment.",
    code: `nn.Dropout(0.3)
torch.optim.AdamW(model.parameters(), weight_decay=0.01)
nn.CrossEntropyLoss(label_smoothing=0.1)`,
    importance: "medium",
    followUps: ["How do you implement early stopping?", "What if BOTH accuracies are low?"],
    tags: ["overfitting", "regularisation", "validation"],
  },
  {
    id: "iq-debug-7",
    category: "debugging",
    topic: "debugging",
    question: "You get 'expected scalar type Long but found Float'. What is going on?",
    shortAnswer:
      "A function that needs integer indices received floats — almost always CrossEntropyLoss targets or nn.Embedding input ids. Cast with .long().",
    detailedAnswer:
      "Class labels and token ids are indices used to select a row, so they must be int64. The mirror-image error comes from BCEWithLogitsLoss, which requires FLOAT targets of the same shape as the logits — passing long there raises the opposite complaint. Whenever a dtype error appears, print the dtype of every tensor involved; it is a five-second diagnosis.",
    code: `loss = loss_fn(logits, y.long())    # CrossEntropyLoss
loss = bce(logits, y.float())       # BCEWithLogitsLoss`,
    importance: "medium",
    followUps: ["What dtype do BCE targets need?", "Why must embedding ids be int64?"],
    tags: ["dtype", "crossentropy", "embedding"],
  },
  {
    id: "iq-perf-1",
    category: "performance",
    topic: "performance",
    question: "How would you speed up a slow training run?",
    shortAnswer:
      "Profile first to find whether you are data-bound or GPU-bound. Then: vectorise Python loops, raise the batch size, enable AMP, tune the DataLoader (workers, pinned memory), remove per-batch synchronisation, and consider torch.compile.",
    detailedAnswer:
      "Low GPU utilisation means the input pipeline is the bottleneck: add workers, pin memory, cache decoded data, move augmentation to the GPU. High utilisation but slow means the model: AMP for tensor cores, a larger batch for better kernel efficiency, cudnn.benchmark=True when input shapes are fixed, torch.compile for kernel fusion, and channels_last memory format for CNNs. Per-batch .item() or print() forces a device sync and can dominate the profile of a small model.",
    code: `with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = loss_fn(model(X), y)
torch.backends.cudnn.benchmark = True
model = torch.compile(model)`,
    importance: "medium",
    followUps: ["How do you tell if you are data-bound?", "What does torch.compile actually do?"],
    tags: ["performance", "amp", "profiling", "compile"],
  },
  {
    id: "iq-perf-2",
    category: "performance",
    topic: "performance",
    question: "Explain mixed precision training.",
    shortAnswer:
      "Run most operations in 16-bit for speed and memory while keeping master weights and numerically sensitive operations in float32. torch.autocast picks the precision per operation.",
    detailedAnswer:
      "Matmuls and convolutions run in 16-bit on tensor cores, giving large speedups; reductions, softmax and losses stay in float32 to avoid precision loss. With float16 you also need a GradScaler, because small gradients underflow to zero — the scaler multiplies the loss before backward and unscales before the step. bfloat16 has float32's exponent range, so it rarely underflows and typically needs no scaler; it is the default for LLM training on Ampere and newer. Autocast wraps only the forward pass and loss — never the backward call or the optimizer step.",
    code: `scaler = torch.amp.GradScaler("cuda")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = loss_fn(model(X), y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()`,
    importance: "medium",
    followUps: ["Why does bfloat16 not need a scaler?", "What can go numerically wrong?"],
    tags: ["amp", "float16", "bfloat16", "GradScaler"],
  },
  {
    id: "iq-perf-3",
    category: "performance",
    topic: "performance",
    question: "What is gradient accumulation and when do you use it?",
    shortAnswer:
      "Running several small forward/backward passes before a single optimizer.step(), so the gradients sum into the equivalent of one large batch. It lets you use a large effective batch size on limited memory.",
    detailedAnswer:
      "Divide each micro-batch loss by the number of accumulation steps so the accumulated gradient equals the average over the full batch — omit that and you effectively multiply your learning rate. It is standard in LLM fine-tuning where a batch of 1-2 sequences is all that fits. Caveat: BatchNorm still sees only the micro-batch, so its statistics do not match a true large batch (another reason transformers use LayerNorm).",
    code: `for i, (X, y) in enumerate(loader):
    loss = loss_fn(model(X), y) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()`,
    importance: "medium",
    followUps: ["Why divide the loss?", "How does it interact with BatchNorm?"],
    tags: ["accumulation", "batch size", "memory"],
  },
  {
    id: "iq-tf-1",
    category: "transformers",
    topic: "transformers",
    question: "Explain scaled dot-product attention.",
    shortAnswer:
      "softmax(QKᵀ/√d)V. Queries and keys produce a similarity matrix, the scaling keeps softmax out of saturation, and the resulting weights average the value vectors.",
    detailedAnswer:
      "Each token emits a query, a key and a value via learned linear projections. QKᵀ gives a (seq × seq) score matrix — how much each position should attend to every other. Dividing by √d_k matters because the dot product's variance grows with dimension; without it softmax saturates into a near one-hot distribution and gradients vanish. Softmax over the KEY axis (dim=-1) makes each query's weights sum to 1, and multiplying by V produces the weighted average. Multi-head attention runs this in several lower-dimensional subspaces in parallel and concatenates the results.",
    code: `scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
attn = torch.softmax(scores, dim=-1)
out = attn @ V`,
    importance: "high",
    followUps: ["Why softmax over dim=-1?", "What is the memory cost for long sequences?"],
    tags: ["attention", "qkv", "softmax", "scaling"],
  },
  {
    id: "iq-tf-2",
    category: "transformers",
    topic: "transformers",
    question: "What are the components of a transformer block?",
    shortAnswer:
      "Multi-head self-attention and a position-wise feed-forward network, each wrapped in a residual connection and a LayerNorm. Modern implementations put the norm before each sublayer (pre-norm).",
    detailedAnswer:
      "Attention mixes information ACROSS positions; the FFN (usually expanding 4× with GELU in between) transforms each position independently. Residual connections give gradients a direct path through depth and let each block make an incremental edit to a shared 'residual stream'. Pre-norm keeps that stream unnormalised and trains far more stably at depth than the original post-norm design — it is why modern LLMs need less warmup. Positional information must be injected separately, since attention itself is permutation-invariant.",
    code: `h = self.ln1(x)
x = x + self.attn(h, h, h)[0]
x = x + self.ff(self.ln2(x))`,
    importance: "high",
    followUps: ["Why 4× expansion in the FFN?", "Pre-norm vs post-norm?"],
    tags: ["transformer", "residual", "ffn", "layernorm"],
  },
  {
    id: "iq-tf-3",
    category: "transformers",
    topic: "transformers",
    question: "Why do transformers need positional encodings?",
    shortAnswer:
      "Self-attention is permutation-invariant — shuffling the tokens produces the same set of outputs. Position must be added explicitly, either as learned embeddings, sinusoids, or rotary embeddings applied to Q and K.",
    detailedAnswer:
      "Attention computes weights from content alone, so without positional information 'dog bites man' and 'man bites dog' are indistinguishable. Learned absolute embeddings are simplest but cannot extrapolate past the trained context length. Sinusoidal encodings extrapolate in principle. RoPE (rotary) rotates Q and K by a position-dependent angle, so the attention score depends on RELATIVE position — it generalises better to longer contexts and is what most current LLMs use.",
    code: `x = tok_emb(idx) + pos_emb(torch.arange(T, device=idx.device))`,
    importance: "medium",
    followUps: ["What is RoPE?", "Why do learned encodings limit context length?"],
    tags: ["positional", "rope", "transformer"],
  },
  {
    id: "iq-tf-4",
    category: "transformers",
    topic: "transformers",
    question: "How does nn.MultiheadAttention split the embedding across heads?",
    shortAnswer:
      "embed_dim is divided evenly by num_heads, so each head attends in an embed_dim // num_heads subspace. The head outputs are concatenated back to embed_dim and passed through an output projection.",
    detailedAnswer:
      "With embed_dim=512 and num_heads=8 each head works on 64 channels. The point is representational: different heads can specialise (syntax, coreference, positional patterns) while the total compute stays roughly the same as a single 512-dim head. embed_dim must be divisible by num_heads or the constructor raises. Remember batch_first=True if you use the (B, T, C) layout, and that the module returns a tuple (output, attention_weights).",
    code: `mha = nn.MultiheadAttention(512, 8, batch_first=True)
out, weights = mha(x, x, x)   # self-attention`,
    importance: "medium",
    followUps: ["What is cross-attention?", "Why return the weights at all?"],
    tags: ["multihead", "attention", "heads"],
  },
  {
    id: "iq-tf-5",
    category: "transformers",
    topic: "transformers",
    question: "What does nn.Embedding do, and why must its input be int64?",
    shortAnswer:
      "It is a learned lookup table mapping token ids to dense vectors. The ids index rows of the weight matrix, and indexing requires integers — int64 specifically.",
    detailedAnswer:
      "Mathematically it is equivalent to multiplying a one-hot vector by the weight matrix, but implemented as a gather, which is far cheaper. Passing float ids raises 'Expected tensor for argument #1 indices to have scalar type Long'. padding_idx pins one row to zero and excludes it from gradients, which is how padded positions are handled. Output shape is input shape plus an embedding dimension: (B, T) → (B, T, C).",
    code: `emb = nn.Embedding(10000, 512, padding_idx=0)
emb(torch.randint(0, 10000, (8, 128))).shape   # (8, 128, 512)`,
    importance: "medium",
    followUps: ["What does padding_idx do?", "How is this different from a Linear on one-hot input?"],
    tags: ["embedding", "int64", "nlp"],
  },
  {
    id: "iq-general-1",
    category: "practical ML",
    topic: "training",
    question: "How do you know your model is training correctly in the first five minutes?",
    shortAnswer:
      "Check that the initial loss matches the theoretical value for random predictions, overfit a single batch to near-zero loss, and confirm gradient norms are finite and non-zero.",
    detailedAnswer:
      "For 10-class cross-entropy, an untrained model should start near ln(10) ≈ 2.30; a very different value means the labels, the output shape or the loss are wrong. Then overfit one batch — if that fails, the plumbing is broken, not the hyperparameters. Print the gradient norm for a couple of layers to confirm gradients reach the earliest layers. These three checks catch the large majority of setup bugs before you waste a GPU-hour.",
    code: `print(loss.item())   # ~2.30 for 10 balanced classes at init
total = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
print(total)         # finite and > 0`,
    importance: "medium",
    followUps: ["What is the expected initial loss for binary classification?", "What would a much LOWER initial loss suggest?"],
    tags: ["sanity check", "debug", "initial loss"],
  },
  {
    id: "iq-general-2",
    category: "practical ML",
    topic: "evaluation",
    question: "How would you deploy a PyTorch model for inference?",
    shortAnswer:
      "Load the state_dict into the architecture, call model.eval(), run under torch.inference_mode(), batch requests where possible, and export to TorchScript or ONNX if the serving environment needs it.",
    detailedAnswer:
      "The correctness essentials are eval() (dropout off, BatchNorm on running statistics) and inference_mode() (no graph, lower overhead). For throughput: batch incoming requests, use half or bfloat16 precision where accuracy allows, pin the model to a device once at startup, and warm it up with a dummy forward so lazy CUDA initialisation and any compilation happen before the first real request. torch.jit.script/trace or an ONNX export removes the Python dependency; quantisation cuts latency further on CPU.",
    code: `model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device).eval()
with torch.inference_mode():
    logits = model(batch)`,
    importance: "medium",
    followUps: ["TorchScript vs ONNX?", "What does a warmup pass fix?"],
    tags: ["deployment", "inference", "eval", "onnx"],
  },
  {
    id: "iq-general-3",
    category: "practical ML",
    topic: "training",
    question: "How do you make a PyTorch experiment reproducible?",
    shortAnswer:
      "Seed Python, NumPy and torch (including CUDA), seed the DataLoader workers, and optionally enable deterministic algorithms. Full bit-exactness also requires disabling cuDNN autotuning.",
    detailedAnswer:
      "Seeding alone is not enough: cuDNN may pick different algorithms per run, some CUDA kernels use non-deterministic atomics, and each DataLoader worker has its own RNG. torch.use_deterministic_algorithms(True) forces deterministic kernels (raising when none exists) and cudnn.benchmark=False stops autotuning — both cost performance, so most teams accept run-to-run variance and instead report a mean over several seeds. Also log the config, the git commit and the library versions; that reproduces far more failures than the RNG does.",
    code: `torch.manual_seed(42); torch.cuda.manual_seed_all(42)
np.random.seed(42); random.seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False`,
    importance: "medium",
    followUps: ["Why is seeding not sufficient?", "What is the performance cost?"],
    tags: ["reproducibility", "seed", "determinism"],
  },
  {
    id: "iq-general-4",
    category: "practical ML",
    topic: "nn",
    question: "How do you fine-tune a pretrained model in PyTorch?",
    shortAnswer:
      "Replace the head with one matching your class count, optionally freeze the backbone by setting requires_grad=False, and build the optimizer AFTER the architecture is final — usually with a lower learning rate for pretrained layers.",
    detailedAnswer:
      "The common failure is constructing the optimizer before replacing the head: it then holds references to the old parameters and the new head never updates. A typical recipe is to freeze the backbone and train the head for a few epochs, then unfreeze and continue with a small learning rate (discriminative learning rates via parameter groups). Keep the pretrained model's normalisation statistics for preprocessing, and remember frozen BatchNorm layers should usually stay in eval mode so their running statistics are not overwritten by your smaller dataset.",
    code: `for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(512, num_classes)      # new head, requires_grad=True
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-3)`,
    importance: "medium",
    followUps: ["What are discriminative learning rates?", "Why keep frozen BatchNorm in eval mode?"],
    tags: ["fine-tuning", "freeze", "transfer learning"],
  },
  {
    id: "iq-general-5",
    category: "practical ML",
    topic: "data",
    question: "What is data leakage and how do you avoid it?",
    shortAnswer:
      "Information from the validation or test set influencing training — e.g. scaling with statistics computed over the whole dataset, duplicate rows across splits, or a feature derived from the target. Split first, then fit every transform on the training portion only.",
    detailedAnswer:
      "Leakage produces validation numbers that look excellent and collapse in production. Common forms: normalisation or imputation fitted before splitting; random splits of time-series data (the model sees the future); near-duplicate images or augmented copies of the same source landing in both splits; group leakage where the same patient/user appears in both. Defences: split by group or by time, fit all preprocessing inside the training fold, and be suspicious of any result that seems too good.",
    code: `train_ds, val_ds = random_split(ds, [800, 200])
mean, std = compute_stats(train_ds)   # TRAIN only`,
    importance: "medium",
    followUps: ["How do you split time-series data?", "What is group leakage?"],
    tags: ["leakage", "validation", "splitting"],
  },
  {
    id: "iq-general-6",
    category: "PyTorch fundamentals",
    topic: "nn",
    question: "Why do you call model(x) instead of model.forward(x)?",
    shortAnswer:
      "model(x) invokes __call__, which runs registered forward pre/post hooks and module bookkeeping before delegating to forward(). Calling forward() directly skips all of that.",
    detailedAnswer:
      "Hooks are how profilers, feature extractors, quantisation and some distributed wrappers work — they attach to __call__, not to forward. Bypassing it usually appears to work in a simple script and then silently breaks when someone adds a hook or wraps the model. It is a small detail, but a good signal that you have read the framework rather than only copied examples.",
    code: `pred = model(X)          # correct
# pred = model.forward(X)  # skips hooks`,
    importance: "low",
    followUps: ["What is a forward hook used for?"],
    tags: ["forward", "__call__", "hooks"],
  },
  {
    id: "iq-general-7",
    category: "PyTorch fundamentals",
    topic: "gpu",
    question: "How would you train on multiple GPUs?",
    shortAnswer:
      "Use DistributedDataParallel: one process per GPU, each with its own copy of the model and a DistributedSampler, with gradients all-reduced during backward. DataParallel is the old single-process approach and is not recommended.",
    detailedAnswer:
      "DDP scales near-linearly because gradient communication overlaps with the backward pass and there is no single-process bottleneck. Each rank sees a distinct shard of the data via DistributedSampler (remember sampler.set_epoch(epoch) so shuffling differs each epoch). The effective batch size becomes per_gpu_batch × num_gpus, so the learning rate usually needs scaling. Only rank 0 should write checkpoints and logs. For models too large for one GPU, you move to sharding approaches such as FSDP or ZeRO.",
    code: `model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler, batch_size=32)`,
    importance: "low",
    followUps: ["Why is DataParallel discouraged?", "What is FSDP for?"],
    tags: ["ddp", "multi-gpu", "distributed"],
  },
  {
    id: "iq-general-8",
    category: "tensors",
    topic: "shapes",
    question: "What does .contiguous() do and when do you need it?",
    shortAnswer:
      "It returns a tensor whose memory layout matches its shape, copying if necessary. You need it before view() on a tensor produced by transpose/permute.",
    detailedAnswer:
      "PyTorch tensors are a flat buffer plus strides. transpose and permute only rewrite strides, so the logical order no longer matches memory order. view() requires them to match, hence the familiar `.transpose(1, 2).contiguous().view(B, T, C)` in attention implementations. reshape() hides this by copying for you. Contiguity also affects performance: some kernels are faster on contiguous input, which is why channels_last is a deliberate memory-format choice for CNNs.",
    code: `y = x.permute(1, 0, 2)
y.is_contiguous()               # False
y.contiguous().view(...)        # ok`,
    importance: "medium",
    followUps: ["Why is a transposed tensor not contiguous?", "What is channels_last?"],
    tags: ["contiguous", "stride", "view"],
  },
];
