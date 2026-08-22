import type { InterviewQuestion } from "@/lib/types";

export const fundamentalQuestions: InterviewQuestion[] = [
  {
    id: "iq-tensor-1",
    category: "PyTorch fundamentals",
    topic: "interop",
    question: "What is a PyTorch tensor, and how does it differ from a NumPy array?",
    shortAnswer:
      "A tensor is an n-dimensional array like ndarray, but it can live on a GPU, it records operations for automatic differentiation, and it plugs directly into nn.Module and the optimizers.",
    detailedAnswer:
      "The API is deliberately close to NumPy — same indexing, broadcasting and reduction semantics — so the mental model transfers. The three additions are what make it a deep-learning framework: (1) tensors can be allocated on accelerators and every op dispatches to the right kernel; (2) when requires_grad is set, operations are recorded in a dynamic graph so gradients can be computed by reverse-mode autodiff; (3) tensors are the currency of nn.Module, so the same object flows through layers, losses and optimizers. On CPU the two even share memory: torch.from_numpy and .numpy() are views, not copies.",
    code: `t = torch.from_numpy(arr)      # shares memory (CPU)
arr = t.detach().cpu().numpy() # back to NumPy`,
    importance: "medium",
    followUps: ["Why does .numpy() fail on a GPU tensor?", "What dtype does from_numpy give you?"],
    tags: ["tensor", "numpy", "gpu", "autograd"],
  },
  {
    id: "iq-tensor-2",
    category: "tensors",
    topic: "shapes",
    question: "What is the difference between reshape() and view()?",
    shortAnswer:
      "view() returns a zero-copy reinterpretation and requires the tensor to be contiguous; reshape() does the same when it can, and silently falls back to a copy when it cannot.",
    detailedAnswer:
      "A tensor is a flat buffer plus a shape and strides. view() only rewrites the shape/stride metadata, so it needs the existing memory order to be compatible — after a transpose or permute it is not, and view() raises. reshape() tries view() first and copies otherwise, so it always succeeds. Practical guidance: use reshape() by default; use view() when you want an error rather than a hidden copy in a performance-critical path. When you know you need the copy, .contiguous().view(...) makes the intent explicit.",
    code: `x = torch.randn(4, 6)
x.view(2, 12)          # ok
y = x.t()              # non-contiguous
# y.view(4, 6)         # RuntimeError
y.reshape(4, 6)        # ok (copies)
y.contiguous().view(4, 6)   # explicit`,
    importance: "high",
    followUps: ["What does contiguous() actually do?", "Why is a transposed tensor not contiguous?"],
    tags: ["reshape", "view", "contiguous", "memory"],
  },
  {
    id: "iq-tensor-3",
    category: "tensors",
    topic: "shapes",
    question: "What is the difference between torch.cat() and torch.stack()?",
    shortAnswer:
      "cat joins tensors along an existing dimension and keeps the rank; stack creates a new dimension and increases the rank by one.",
    detailedAnswer:
      "torch.cat requires the inputs to match on every dimension except the concatenation dimension, and the output's size along that dimension is the sum of the inputs'. torch.stack requires the shapes to be identical and inserts a brand-new axis. Rule of thumb: collecting per-batch predictions over an epoch and wanting one long tensor → cat. Turning a list of individual samples into a batch → stack (this is exactly what the default DataLoader collate does).",
    code: `a = torch.randn(2, 3); b = torch.randn(2, 3)
torch.cat([a, b], dim=0).shape    # (4, 3)
torch.cat([a, b], dim=1).shape    # (2, 6)
torch.stack([a, b], dim=0).shape  # (2, 2, 3)`,
    importance: "high",
    followUps: ["What does the default collate_fn use?", "How would you undo a stack?"],
    tags: ["cat", "stack", "shapes"],
  },
  {
    id: "iq-tensor-4",
    category: "tensors",
    topic: "broadcasting",
    question: "Explain broadcasting. What are the rules?",
    shortAnswer:
      "Shapes are aligned from the right; two dimensions are compatible if they are equal or one of them is 1, and missing leading dimensions count as 1. The size-1 dimensions are then virtually repeated.",
    detailedAnswer:
      "Broadcasting lets you combine tensors of different shapes without materialising copies — internally the size-1 dimension gets a stride of 0. It is what makes `x + bias` work for a (B, F) batch and an (F,) bias. The danger is that it makes some SHAPE MISTAKES succeed instead of raising: subtracting a (32, 1) target from a (32,) prediction gives a (32, 32) matrix, and the loss still returns a number. When a loss looks wrong, print both shapes first.",
    code: `(3, 4) + (4,)      -> (3, 4)
(32, 10) + (10,)   -> (32, 10)
(5, 1) * (1, 7)    -> (5, 7)
(3, 4) + (3,)      -> RuntimeError`,
    importance: "high",
    followUps: ["How do you add a per-row vector to a matrix?", "Give an example where broadcasting hides a bug."],
    tags: ["broadcasting", "shapes", "bug"],
  },
  {
    id: "iq-tensor-5",
    category: "tensors",
    topic: "shapes",
    question: "What are the standard tensor shapes for tabular, image, and text models?",
    shortAnswer:
      "Tabular: (batch, features). Images: (batch, channels, height, width) — PyTorch is channels-first. Text/transformers: (batch, seq_len, hidden). Classification logits: (batch, num_classes), with labels of shape (batch,).",
    detailedAnswer:
      "Knowing the conventions makes error messages readable. PyTorch's NCHW layout differs from TensorFlow/PIL/OpenCV's NHWC, so image pipelines need permute(0, 3, 1, 2). RNNs default to (seq, batch, feature) unless you pass batch_first=True — modern code always does. Attention scores add a head axis: (batch, heads, seq, seq). Class labels for CrossEntropyLoss are 1-D int64 indices, not one-hot.",
    code: `(32, 128)             # tabular
(32, 3, 224, 224)     # images (NCHW)
(8, 128, 768)         # transformer hidden states
(32, 10)              # logits
(32,)                 # int64 labels`,
    importance: "high",
    tags: ["shapes", "NCHW", "conventions"],
  },
  {
    id: "iq-autograd-1",
    category: "autograd",
    topic: "autograd",
    question: "What does requires_grad=True do?",
    shortAnswer:
      "It marks a tensor as one to differentiate with respect to, so every operation involving it is recorded in the computational graph and .backward() can fill its .grad.",
    detailedAnswer:
      "Nothing is computed when you set the flag — it only enables recording. Each resulting tensor gets a grad_fn pointing back at the operation that produced it, forming a DAG. Parameters created by nn.Module have it set automatically (nn.Parameter). You set it on inputs only in special cases: adversarial examples, input optimisation, or feature visualisation. Setting it to False on a submodule's parameters is how you freeze a backbone for fine-tuning.",
    code: `x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
print(y.grad_fn)   # <PowBackward0>
y.backward()
print(x.grad)      # tensor(4.)`,
    importance: "high",
    followUps: ["Can you set requires_grad on an int tensor?", "How do you freeze part of a model?"],
    tags: ["requires_grad", "autograd", "graph"],
  },
  {
    id: "iq-autograd-2",
    category: "autograd",
    topic: "autograd",
    question: "What is a computational graph in PyTorch?",
    shortAnswer:
      "A dynamic DAG built during the forward pass: leaves are inputs and parameters, nodes are operations, and each node knows how to compute its local gradient. backward() traverses it in reverse applying the chain rule.",
    detailedAnswer:
      "PyTorch is 'define-by-run': the graph is constructed fresh on every forward pass, which is why ordinary Python control flow (if, for, recursion) works inside forward() and why debugging with a print or a breakpoint works. The trade-off versus a static graph is that the framework cannot optimise across iterations ahead of time — torch.compile in PyTorch 2.x closes much of that gap. The graph's buffers are freed after backward() to save memory, which is why calling backward twice raises unless you pass retain_graph=True.",
    code: `x = torch.randn(3, requires_grad=True)
y = (x * 2).sum()
y.backward()
# y.backward()   # RuntimeError: graph freed`,
    importance: "high",
    followUps: ["Why is calling backward() twice an error?", "What does define-by-run buy you?"],
    tags: ["graph", "dynamic", "define-by-run"],
  },
  {
    id: "iq-autograd-3",
    category: "autograd",
    topic: "autograd",
    question: "Why does PyTorch accumulate gradients by default?",
    shortAnswer:
      "Because backward() adds into .grad rather than overwriting it, you can accumulate gradients across several forward passes or several losses before stepping — which is what makes gradient accumulation and multi-task losses trivial.",
    detailedAnswer:
      "Accumulation is the more general primitive: summing then stepping is exactly what a single large batch would do, so splitting a batch into micro-batches to fit in memory needs no special support. It also makes multi-loss setups natural — call backward on each loss and the gradients add. The cost is that you must clear the accumulator yourself once per step with optimizer.zero_grad(). Forgetting it means every step uses the sum of all previous gradients, so the effective step size grows and training diverges.",
    code: `x = torch.tensor(2.0, requires_grad=True)
(x ** 2).backward()   # x.grad = 4
(x ** 2).backward()   # x.grad = 8, not 4`,
    importance: "high",
    followUps: ["What happens if you forget zero_grad()?", "How do you implement gradient accumulation?"],
    tags: ["accumulate", "zero_grad", "gradient accumulation"],
  },
  {
    id: "iq-autograd-4",
    category: "autograd",
    topic: "autograd",
    question: "What does .backward() do, and where are the gradients stored?",
    shortAnswer:
      "It walks the graph backwards applying the chain rule and accumulates dLoss/dParam into the .grad attribute of every leaf tensor with requires_grad=True — i.e. the model parameters.",
    detailedAnswer:
      "backward() must be called on a scalar, because reverse-mode autodiff needs a starting gradient (implicitly 1.0). For a non-scalar you must pass an explicit gradient tensor, which is why losses are always reduced with mean() or sum(). Gradients for intermediate (non-leaf) tensors are computed but discarded unless you call retain_grad(). The only link between backward() and optimizer.step() is the parameter objects themselves: backward writes p.grad, step reads it.",
    code: `loss.backward()
for p in model.parameters():
    print(p.grad.shape)
optimizer.step()`,
    importance: "high",
    followUps: ["Why must backward() be called on a scalar?", "Why is y.grad None for an intermediate y?"],
    tags: ["backward", "grad", "leaf"],
  },
  {
    id: "iq-autograd-5",
    category: "autograd",
    topic: "autograd",
    question: "What is torch.no_grad() and when do you use it?",
    shortAnswer:
      "A context manager that stops autograd from recording operations. Use it for validation and inference: computation still runs, but no graph is built, which saves memory and time.",
    detailedAnswer:
      "Inside the block, results have requires_grad=False and no grad_fn, so the activations that would be needed for a backward pass are never retained — often a large fraction of training memory. torch.inference_mode() is a newer, slightly faster variant that additionally skips version-counter bookkeeping; its one restriction is that tensors created inside it cannot later participate in autograd. Note that no_grad() has nothing to do with dropout or BatchNorm — that is model.eval().",
    code: `model.eval()
with torch.no_grad():
    for X, y in val_loader:
        pred = model(X)`,
    importance: "high",
    followUps: ["How is it different from model.eval()?", "What is inference_mode()?"],
    tags: ["no_grad", "inference_mode", "memory"],
  },
  {
    id: "iq-autograd-6",
    category: "autograd",
    topic: "autograd",
    question: "What does .detach() do, and how is it different from no_grad()?",
    shortAnswer:
      "detach() returns a tensor sharing the same data but cut out of the graph. It acts on one tensor; no_grad() prevents graph construction for a whole block.",
    detailedAnswer:
      "Use detach() when you want a value to stop propagating gradients: logging metrics, computing a target from a frozen network, or truncated BPTT where you detach the hidden state between chunks so the graph does not grow without bound. Note detach() shares storage, so an in-place change to one is visible in the other — use .detach().clone() if you need independence. no_grad() is broader and also avoids the memory cost of building the graph in the first place, which detaching after the fact cannot recover.",
    code: `logged = loss.detach()                # keep the value, drop the graph
arr = pred.detach().cpu().numpy()     # hand off to NumPy
hidden = hidden.detach()              # truncated BPTT`,
    importance: "high",
    followUps: ["Why does .numpy() fail without detach?", "What is truncated BPTT?"],
    tags: ["detach", "no_grad", "bptt"],
  },
  {
    id: "iq-nn-1",
    category: "neural networks",
    topic: "nn",
    question: "Walk me through defining a model with nn.Module.",
    shortAnswer:
      "Subclass nn.Module, call super().__init__() first, create the layers as attributes in __init__, and describe the data flow in forward(). Then call the model itself, not forward().",
    detailedAnswer:
      "Assigning a Module or nn.Parameter as an attribute registers it in the parent's internal dictionaries, which is what makes parameters(), state_dict(), .to(device) and train()/eval() work recursively. super().__init__() must run first because it creates those dictionaries. A plain Python list of layers is NOT registered — use nn.ModuleList or nn.Sequential. You implement forward() but invoke the module via __call__ so that hooks run.",
    code: `class Net(nn.Module):
    def __init__(self, in_f, hidden, out_f):
        super().__init__()
        self.fc1 = nn.Linear(in_f, hidden)
        self.fc2 = nn.Linear(hidden, out_f)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))`,
    importance: "high",
    followUps: ["What happens if you forget super().__init__()?", "Why nn.ModuleList over a Python list?"],
    tags: ["nn.Module", "forward", "parameters"],
  },
  {
    id: "iq-nn-2",
    category: "neural networks",
    topic: "nn",
    question: "What is the difference between model.parameters() and model.state_dict()?",
    shortAnswer:
      "parameters() yields only the learnable tensors — what the optimizer needs. state_dict() is a name→tensor mapping that also includes buffers such as BatchNorm's running statistics — what a checkpoint needs.",
    detailedAnswer:
      "Buffers are registered with register_buffer: non-learnable state that must move with the model and be saved, e.g. running_mean/running_var/num_batches_tracked, or a fixed positional-encoding table. If you saved only the parameters, a restored model would produce different eval-mode predictions because the BatchNorm statistics would be reset. state_dict() is also just an OrderedDict of tensors, which makes it inspectable and portable across refactors.",
    code: `sum(p.numel() for p in model.parameters())
list(model.state_dict().keys())   # includes running_mean, running_var`,
    importance: "medium",
    followUps: ["What is a buffer?", "Why save state_dict rather than the model?"],
    tags: ["parameters", "state_dict", "buffers"],
  },
  {
    id: "iq-nn-3",
    category: "neural networks",
    topic: "nn",
    question: "What is dropout and why must it behave differently at inference?",
    shortAnswer:
      "During training it zeroes each activation with probability p and scales the survivors by 1/(1-p); at inference it is disabled entirely, so predictions are deterministic.",
    detailedAnswer:
      "Dropout prevents co-adaptation: units cannot rely on a specific partner being present, so the network learns redundant representations. The 1/(1-p) scaling at training time ('inverted dropout') keeps the expected activation constant, which is what allows inference to simply skip dropout with no rescaling. model.eval() is what flips the switch — forgetting it makes validation stochastic and pessimistic.",
    code: `drop = nn.Dropout(0.5)
drop.train(); drop(torch.ones(6))   # some zeros, others 2.0
drop.eval();  drop(torch.ones(6))   # all 1.0`,
    importance: "high",
    followUps: ["Why the 1/(1-p) scaling?", "Where in the network would you place dropout?"],
    tags: ["dropout", "regularisation", "eval"],
  },
  {
    id: "iq-nn-4",
    category: "neural networks",
    topic: "nn",
    question: "What is batch normalisation, and what does it do differently in eval mode?",
    shortAnswer:
      "It normalises each channel using the current batch's mean and variance, then applies a learnable scale and shift. In eval mode it uses running statistics collected during training instead.",
    detailedAnswer:
      "BatchNorm stabilises and speeds up training by keeping activation distributions well-conditioned, which permits higher learning rates and reduces sensitivity to initialisation. Because it depends on the batch, it degrades with very small batches and cannot use batch statistics at inference (a prediction would depend on which other samples share the batch, and batch size 1 has no variance). Hence the running_mean/running_var buffers. A bias in the preceding conv/linear is redundant since BatchNorm subtracts the mean anyway.",
    code: `bn = nn.BatchNorm2d(16)
bn.running_mean.shape   # (16,)`,
    importance: "high",
    followUps: ["Why does BatchNorm fail with batch size 1 in training?", "When would you use GroupNorm?"],
    tags: ["batchnorm", "eval", "running stats"],
  },
  {
    id: "iq-nn-5",
    category: "neural networks",
    topic: "nn",
    question: "What is layer normalisation, and why do transformers prefer it over BatchNorm?",
    shortAnswer:
      "LayerNorm normalises across the feature dimension of each sample independently. Transformers use it because it does not depend on the batch, so it works with variable sequence lengths, small batches and batch-size-1 inference.",
    detailedAnswer:
      "BatchNorm's statistics are computed over the batch axis. In NLP, sequences are padded to different lengths, so batch statistics mix real tokens with padding; batches are also small relative to model size, and autoregressive inference runs one sequence at a time. LayerNorm sidesteps all of it by normalising each token's own feature vector, and behaves identically in train and eval — no running buffers at all. Modern LLMs typically use pre-norm (LayerNorm before each sublayer) and often RMSNorm, a cheaper variant that skips mean subtraction.",
    code: `ln = nn.LayerNorm(768)
x = torch.randn(8, 128, 768)   # (B, T, C)
ln(x).shape                     # (8, 128, 768)`,
    importance: "high",
    followUps: ["What is pre-norm vs post-norm?", "What is RMSNorm?"],
    tags: ["layernorm", "batchnorm", "transformer"],
  },
  {
    id: "iq-nn-6",
    category: "neural networks",
    topic: "nn",
    question: "Compare ReLU, LeakyReLU, GELU, sigmoid and tanh. Where is each used?",
    shortAnswer:
      "ReLU is the cheap CNN default; LeakyReLU avoids dead units; GELU is smooth and standard in transformers; sigmoid produces independent probabilities at the output; tanh is zero-centred and shows up in RNN gates.",
    detailedAnswer:
      "ReLU is fast and non-saturating for positive inputs, but a unit stuck negative gets zero gradient forever ('dying ReLU') — LeakyReLU's small negative slope and GELU/ELU's smooth curves fix that. Sigmoid and tanh saturate at both ends, so deep stacks of them suffer vanishing gradients; that is why they survive mainly as output activations (sigmoid) or inside gating mechanisms (tanh/sigmoid in LSTM). Softmax is an output activation only — it normalises across classes.",
    code: `F.relu(x)  ·  F.leaky_relu(x, 0.01)  ·  F.gelu(x)
torch.sigmoid(x)  ·  torch.tanh(x)  ·  torch.softmax(x, dim=-1)`,
    importance: "medium",
    followUps: ["What is the dying ReLU problem?", "Why does sigmoid cause vanishing gradients?"],
    tags: ["activation", "relu", "gelu", "vanishing"],
  },
  {
    id: "iq-nn-7",
    category: "neural networks",
    topic: "nn",
    question: "Why can't you initialise all weights to zero?",
    shortAnswer:
      "Every unit in a layer would compute the same output and receive the same gradient, so they would stay identical forever — the network never breaks symmetry and behaves like a single unit per layer.",
    detailedAnswer:
      "Random initialisation breaks the symmetry, but the SCALE matters too: too small and the signal (and gradient) vanishes through depth; too large and it explodes. Kaiming/He initialisation accounts for ReLU zeroing half the activations and is the PyTorch default for Linear and Conv; Xavier/Glorot targets tanh/sigmoid. Biases are usually initialised to zero, which is fine because the weights already break symmetry.",
    code: `nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
nn.init.zeros_(m.bias)`,
    importance: "medium",
    followUps: ["Kaiming vs Xavier — when do you use which?", "Why is zero bias fine?"],
    tags: ["initialisation", "symmetry", "kaiming"],
  },
];
