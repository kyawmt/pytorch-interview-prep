import type { CommonMistake } from "@/lib/types";

/**
 * Realistic PyTorch mistakes, phrased the way they actually appear:
 * the symptom you observe first, then the fix.
 */
export const MISTAKES: CommonMistake[] = [
  {
    id: "mk-zero-grad",
    title: "Forgetting optimizer.zero_grad()",
    topic: "training",
    importance: "high",
    wrong: `for X, y in loader:
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()`,
    right: `for X, y in loader:
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()`,
    symptom: "Loss climbs after a few dozen steps, or turns into NaN. Nothing raises.",
    explanation:
      "backward() ACCUMULATES into .grad rather than overwriting it. Without zero_grad each step applies the sum of every gradient so far, so the effective step size grows every iteration until training diverges.",
    tags: ["zero_grad", "accumulate"],
  },
  {
    id: "mk-softmax-ce",
    title: "Softmax before CrossEntropyLoss",
    topic: "losses",
    importance: "high",
    wrong: `logits = model(X)
probs = torch.softmax(logits, dim=1)
loss = nn.CrossEntropyLoss()(probs, y)`,
    right: `logits = model(X)
loss = nn.CrossEntropyLoss()(logits, y)`,
    symptom: "Training works but converges slowly and plateaus at a worse accuracy.",
    explanation:
      "CrossEntropyLoss = log_softmax + NLLLoss. Applying softmax first applies it twice: the distribution is flattened toward uniform, gradients shrink, and you lose the numerically stable fused implementation.",
    tags: ["softmax", "crossentropy", "logits"],
  },
  {
    id: "mk-device",
    title: "Model on GPU, data on CPU",
    topic: "gpu",
    importance: "high",
    wrong: `model.to(device)

for X, y in loader:
    pred = model(X)`,
    right: `model.to(device)

for X, y in loader:
    X, y = X.to(device), y.to(device)
    pred = model(X)`,
    symptom: "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!",
    explanation:
      "Moving the model does not move the data. Every tensor that touches the model — inputs and targets — must be on the same device.",
    tags: ["device", "cuda", "mismatch"],
  },
  {
    id: "mk-to-inplace",
    title: "X.to(device) without reassigning",
    topic: "gpu",
    importance: "high",
    wrong: `X.to(device)
y.to(device)
pred = model(X)`,
    right: `X = X.to(device)
y = y.to(device)
pred = model(X)`,
    symptom: "The same device error as above, even though .to(device) is clearly in the code.",
    explanation:
      "Tensor.to() is out-of-place and returns a new tensor. nn.Module.to() is the exception that mutates in place — mixing up the two conventions is extremely common. The same applies to .float(), .reshape() and friends.",
    tags: ["to", "in place", "device"],
  },
  {
    id: "mk-eval",
    title: "Forgetting model.eval()",
    topic: "evaluation",
    importance: "high",
    wrong: `with torch.no_grad():
    for X, y in val_loader:
        pred = model(X)`,
    right: `model.eval()
with torch.no_grad():
    for X, y in val_loader:
        pred = model(X)`,
    symptom: "Validation accuracy is lower than training and changes between runs on the same data.",
    explanation:
      "Without eval(), dropout stays active and BatchNorm keeps updating its running statistics from the validation batches. Your metric becomes stochastic AND your model is quietly contaminated by validation data.",
    tags: ["eval", "dropout", "batchnorm"],
  },
  {
    id: "mk-train-back",
    title: "Forgetting model.train() after validating",
    topic: "evaluation",
    importance: "high",
    wrong: `for epoch in range(epochs):
    for X, y in train_loader:
        ...
    model.eval()
    validate()`,
    right: `for epoch in range(epochs):
    model.train()
    for X, y in train_loader:
        ...
    model.eval()
    validate()`,
    symptom: "The model overfits noticeably more from epoch 2 onwards.",
    explanation:
      "eval() is sticky. Every epoch after the first trains with dropout disabled and BatchNorm frozen, which removes your regularisation without any error.",
    tags: ["train", "eval", "mode"],
  },
  {
    id: "mk-no-grad",
    title: "Forgetting torch.no_grad() at inference",
    topic: "evaluation",
    importance: "high",
    wrong: `model.eval()
for X, y in val_loader:
    pred = model(X)`,
    right: `model.eval()
with torch.no_grad():
    for X, y in val_loader:
        pred = model(X)`,
    symptom: "Validation uses far more memory than training and can OOM on a large batch.",
    explanation:
      "eval() does not stop autograd. Without no_grad(), PyTorch stores every activation for a backward pass that never happens. torch.inference_mode() is the faster modern alternative.",
    tags: ["no_grad", "memory", "inference"],
  },
  {
    id: "mk-loss-accum",
    title: "Accumulating loss tensors instead of floats",
    topic: "training",
    importance: "high",
    wrong: `running_loss = 0
for X, y in loader:
    loss = loss_fn(model(X), y)
    running_loss += loss`,
    right: `running_loss = 0.0
for X, y in loader:
    loss = loss_fn(model(X), y)
    running_loss += loss.item()`,
    symptom: "CUDA out of memory after a few hundred batches, though batch 1 was fine.",
    explanation:
      "Adding the loss TENSOR keeps its computational graph alive, and the chain grows with every batch. .item() extracts a plain Python float and releases the graph.",
    tags: ["item", "memory leak", "oom"],
  },
  {
    id: "mk-target-dtype",
    title: "Wrong target dtype for CrossEntropyLoss",
    topic: "losses",
    importance: "high",
    wrong: `y = torch.tensor([0.0, 1.0, 2.0])
loss = nn.CrossEntropyLoss()(logits, y)`,
    right: `y = torch.tensor([0, 1, 2])          # int64
loss = nn.CrossEntropyLoss()(logits, y)`,
    symptom: "RuntimeError: expected scalar type Long but found Float",
    explanation:
      "Class labels are indices, so they must be int64. The mirror image: BCEWithLogitsLoss requires FLOAT targets shaped exactly like the logits.",
    tags: ["dtype", "int64", "crossentropy"],
  },
  {
    id: "mk-loss-shape",
    title: "Shape mismatch that broadcasting hides",
    topic: "broadcasting",
    importance: "high",
    wrong: `pred = model(X).squeeze()      # (32,)
loss = nn.MSELoss()(pred, y)  # y is (32, 1)`,
    right: `pred = model(X)               # (32, 1)
loss = nn.MSELoss()(pred, y)  # y is (32, 1)`,
    symptom: "Training runs, the loss decreases a little, and the model never becomes useful.",
    explanation:
      "(32,) and (32, 1) broadcast to (32, 32), so MSE averages 1024 pairwise differences instead of 32 errors. No error is raised. Assert pred.shape == y.shape when a loss looks suspicious.",
    tags: ["broadcasting", "mse", "shape"],
  },
  {
    id: "mk-flatten",
    title: "Flattening away the batch dimension",
    topic: "shapes",
    importance: "high",
    wrong: `x = conv_out.flatten()          # (B*C*H*W,)
x = self.fc(x)`,
    right: `x = conv_out.flatten(1)         # (B, C*H*W)
x = self.fc(x)`,
    symptom: "mat1 and mat2 shapes cannot be multiplied — with one enormous dimension in the message.",
    explanation:
      "flatten() with no argument merges every dimension including the batch. nn.Flatten() defaults to start_dim=1 precisely to avoid this.",
    tags: ["flatten", "batch", "cnn"],
  },
  {
    id: "mk-view-permute",
    title: "Using reshape where permute was meant",
    topic: "shapes",
    importance: "high",
    wrong: `x = x.reshape(B, 3, 224, 224)   # from (B, 224, 224, 3)`,
    right: `x = x.permute(0, 3, 1, 2)       # from (B, 224, 224, 3)`,
    symptom: "No error at all. The model simply never learns anything useful from the images.",
    explanation:
      "reshape reinterprets the flat buffer; permute moves axes and preserves which value belongs to which pixel and channel. Both produce (B, 3, 224, 224), but reshape scrambles the image.",
    tags: ["permute", "reshape", "NCHW"],
  },
  {
    id: "mk-module-list",
    title: "Layers in a plain Python list",
    topic: "nn",
    importance: "medium",
    wrong: `self.layers = [nn.Linear(10, 10) for _ in range(3)]`,
    right: `self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])`,
    symptom: "The model runs but those layers never train, and .to(device) leaves them on the CPU.",
    explanation:
      "nn.Module only registers submodules assigned directly as attributes or held in a Module container. A plain list is invisible to parameters(), state_dict() and .to().",
    tags: ["ModuleList", "parameters", "registration"],
  },
  {
    id: "mk-super",
    title: "Forgetting super().__init__()",
    topic: "nn",
    importance: "medium",
    wrong: `class Net(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(10, 1)`,
    right: `class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)`,
    symptom: "AttributeError: cannot assign module before Module.__init__() call",
    explanation:
      "nn.Module.__init__ creates the internal _parameters and _modules dictionaries. Assigning a submodule before they exist fails immediately.",
    tags: ["super", "nn.Module"],
  },
  {
    id: "mk-optimizer-order",
    title: "Building the optimizer before changing the model",
    topic: "optimizers",
    importance: "medium",
    wrong: `optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.fc = nn.Linear(512, 10)   # new head`,
    right: `model.fc = nn.Linear(512, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)`,
    symptom: "The new head never learns; the loss barely moves.",
    explanation:
      "The optimizer captured references to the OLD parameter objects. Replacing a submodule creates new ones that the optimizer knows nothing about. Always construct the optimizer once the architecture and device are final.",
    tags: ["optimizer", "fine-tuning", "parameters"],
  },
  {
    id: "mk-bce",
    title: "Sigmoid then BCELoss",
    topic: "losses",
    importance: "medium",
    wrong: `probs = torch.sigmoid(logits)
loss = nn.BCELoss()(probs, y)`,
    right: `loss = nn.BCEWithLogitsLoss()(logits, y)`,
    symptom: "Occasional NaN losses, usually once the model becomes confident.",
    explanation:
      "If sigmoid saturates to exactly 0 or 1 in float32, BCELoss takes log(0) = -inf. The fused BCEWithLogitsLoss computes everything from the logit with the log-sum-exp trick and never materialises the probability.",
    tags: ["bce", "stability", "nan"],
  },
  {
    id: "mk-lr",
    title: "Learning rate far too high",
    topic: "optimizers",
    importance: "medium",
    wrong: `optimizer = torch.optim.Adam(model.parameters(), lr=1.0)`,
    right: `optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)`,
    symptom: "The loss oscillates wildly or becomes NaN within the first few dozen steps.",
    explanation:
      "Adam's default 1e-3 is a good starting point for most models, and transformers usually want 1e-4 to 3e-4. When the loss diverges immediately, the learning rate is the first thing to cut by 10×.",
    tags: ["learning rate", "divergence"],
  },
  {
    id: "mk-shuffle-val",
    title: "Shuffling the validation set (and not the training set)",
    topic: "data",
    importance: "low",
    wrong: `train_loader = DataLoader(train_ds, batch_size=32)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=True)`,
    right: `train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)`,
    symptom: "Training converges poorly, especially when the data is sorted by class.",
    explanation:
      "Unshuffled training on class-sorted data means whole batches share one label, which makes gradients wildly biased. Shuffling validation is merely pointless — it does not change the average metric.",
    tags: ["shuffle", "DataLoader"],
  },
  {
    id: "mk-batchnorm-1",
    title: "BatchNorm with a final batch of size 1",
    topic: "nn",
    importance: "low",
    wrong: `loader = DataLoader(ds, batch_size=32)   # len(ds) % 32 == 1`,
    right: `loader = DataLoader(ds, batch_size=32, drop_last=True)`,
    symptom: "ValueError: Expected more than 1 value per channel when training",
    explanation:
      "BatchNorm cannot compute a variance from a single sample. drop_last=True removes the ragged final batch; switching to GroupNorm or LayerNorm removes the batch dependence entirely.",
    tags: ["batchnorm", "drop_last"],
  },
  {
    id: "mk-numpy",
    title: "Calling .numpy() on a GPU or grad-tracking tensor",
    topic: "interop",
    importance: "medium",
    wrong: `arr = pred.numpy()`,
    right: `arr = pred.detach().cpu().numpy()`,
    symptom: "Can't call numpy() on Tensor that requires grad — or — can't convert cuda:0 device type tensor to numpy.",
    explanation:
      "NumPy cannot see GPU memory and refuses to silently break the autograd graph. detach() drops the graph, cpu() moves the data, then numpy() shares the buffer.",
    tags: ["numpy", "detach", "cpu"],
  },
  {
    id: "mk-inplace-autograd",
    title: "In-place op that autograd needs",
    topic: "autograd",
    importance: "medium",
    wrong: `x = self.fc(x)
x += residual        # in place on a tensor autograd needs
return self.act(x)`,
    right: `x = self.fc(x)
x = x + residual
return self.act(x)`,
    symptom: "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation",
    explanation:
      "Some backward functions need the ORIGINAL output value. Overwriting it in place makes the gradient uncomputable. When autograd raises this, remove the in-place form first.",
    tags: ["in place", "autograd", "residual"],
  },
  {
    id: "mk-leak",
    title: "Normalising before splitting the data",
    topic: "data",
    importance: "medium",
    wrong: `X = (X - X.mean()) / X.std()
train_ds, val_ds = random_split(ds, [800, 200])`,
    right: `train_ds, val_ds = random_split(ds, [800, 200])
mean, std = train_stats(train_ds)
# apply (x - mean) / std to both splits`,
    symptom: "Validation accuracy looks excellent and production performance is far worse.",
    explanation:
      "Statistics computed over the whole dataset carry information about the validation samples into training. Split first, fit every transform on the training portion, then apply it to both.",
    tags: ["leakage", "normalisation", "split"],
  },
  {
    id: "mk-single-sample",
    title: "Feeding a single sample without a batch dimension",
    topic: "shapes",
    importance: "medium",
    wrong: `img = load_image()      # (3, 224, 224)
pred = model(img)`,
    right: `img = load_image().unsqueeze(0)   # (1, 3, 224, 224)
pred = model(img)`,
    symptom: "RuntimeError: Expected 4-dimensional input for 4-dimensional weight, but got 3-dimensional input",
    explanation:
      "Models always expect a leading batch dimension. unsqueeze(0) adds it; remember to squeeze it back out of the prediction.",
    tags: ["unsqueeze", "batch", "inference"],
  },
  {
    id: "mk-save-model",
    title: "Pickling the whole model",
    topic: "saving",
    importance: "medium",
    wrong: `torch.save(model, "model.pth")`,
    right: `torch.save(model.state_dict(), "model.pth")`,
    symptom: "Loading fails after a refactor: ModuleNotFoundError or AttributeError for the model class.",
    explanation:
      "Pickle stores the class's import path, so renaming a file or moving a class breaks every old checkpoint. A state_dict is just named tensors and survives refactoring.",
    tags: ["save", "state_dict", "pickle"],
  },
  {
    id: "mk-softmax-dim",
    title: "Softmax over the wrong dimension",
    topic: "math",
    importance: "medium",
    wrong: `probs = torch.softmax(logits, dim=0)   # logits: (B, C)`,
    right: `probs = torch.softmax(logits, dim=1)   # or dim=-1`,
    symptom: "Predicted probabilities change depending on which other samples are in the batch.",
    explanation:
      "dim=0 normalises down the batch instead of across the classes. Remember: for softmax, `dim` names the axis that sums to 1 — the opposite convention from reductions, where `dim` names the axis that disappears.",
    tags: ["softmax", "dim"],
  },
  {
    id: "mk-device-tensor",
    title: "Creating tensors on the default device inside forward()",
    topic: "gpu",
    importance: "medium",
    wrong: `def forward(self, x):
    mask = torch.zeros(x.shape)     # lands on the CPU
    return x * mask`,
    right: `def forward(self, x):
    mask = torch.zeros_like(x)      # same shape, dtype AND device
    return x * mask`,
    symptom: "Device mismatch that only appears when you move the model to a GPU.",
    explanation:
      "Constructors default to the CPU unless told otherwise. Use the *_like family, or pass device=x.device, for anything created inside forward().",
    tags: ["device", "zeros_like", "forward"],
  },
];

export function mistakeById(id: string): CommonMistake | undefined {
  return MISTAKES.find((mistake) => mistake.id === id);
}
