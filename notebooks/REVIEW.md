# PyTorch notebook review

Reviewed 8 September 2026. The notebooks were not modified.

The collection provides useful coverage and many good small examples, but it contains incorrect answers, broken exercise feedback, and a few misleading reusable patterns. For learning quickly, use notebook 15 as the main route after correcting its reshape answer; treat the longer notebooks as references rather than prerequisites.

## Highest-priority corrections

Cell numbers below are one-based positions counting both Markdown and code cells, not execution counts such as `In [12]`.

### 1. The sprint marks a valid reshape as invalid

**Location:** [15_two_day_pytorch_interview_sprint.ipynb](/Users/kyawminthein/02_Job/02_PyTorch/PyTorch_Notebooks/15_two_day_pytorch_interview_sprint.ipynb:245), cell 42, “Day 1 Closed-Book Checkpoint.”

The answer says `(8,12).reshape(3,-1)` is invalid because 96 is not divisible by 3. The correct answer is **`(3,32)`**: `8 × 12 = 96 = 3 × 32`.

**Verified:** `torch.empty(8,12).reshape(3,-1).shape` returns `(3,32)`. Correct the answer and add an executable assertion alongside the key so correct learners are not marked wrong.

### 2. Nine interview reference solutions have broken indentation

**Location:** [11_pytorch_interview_challenge.ipynb](/Users/kyawminthein/02_Job/02_PyTorch/PyTorch_Notebooks/11_pytorch_interview_challenge.ipynb:308), solution cells **11, 36, 48, 53, 60, 65, 80, 88, 92**.

These include the model, training loop, evaluation loop, Dataset, padding collator, attention, and Transformer block. Several function bodies begin at the same indentation as their `def`; other blocks indent only their first top-level statement. The supplied answers cannot be copied into a code cell and run as written.

**Verified:** parsing each reference code block, including removal of common Markdown indentation, raises `IndentationError`/`SyntaxError`. Restore valid Python indentation inside the fences and check the solutions independently. This finding excludes intentionally broken debugging snippets and blank exercise templates.

### 3. The custom Dataset validator rejects a correct implementation

**Location:** [11_pytorch_interview_challenge.ipynb](/Users/kyawminthein/02_Job/02_PyTorch/PyTorch_Notebooks/11_pytorch_interview_challenge.ipynb:382), cells 58–60.

The solution constructs `custom_ds` from `_cd_X` and `_cd_y`, but those inputs are first created in the following validation cell. Worse, every rerun of that validator replaces `_cd_X` with fresh random data, so an already-correct Dataset fails the comparison.

**Verified:** ran the validator once to create inputs, instantiated a correct indexed Dataset, then reran validation; the feature assertion failed.

Move the fixed inputs into a setup cell before the exercise. Validation should inspect the learner's result without regenerating the input.

### 4. The checkpoint cheat sheet restores an optimizer for the wrong model

**Location:** [PyTorch_Cheat_Sheet.ipynb](/Users/kyawminthein/02_Job/02_PyTorch/PyTorch_Notebooks/PyTorch_Cheat_Sheet.ipynb:1136), cell 59.

Weights are restored into `restored_model`, but optimizer state is loaded into `optimizer`, which still owns the original `model` parameters. Continuing training with `restored_model` and this optimizer will not update the restored parameters.

**Verified:** the optimizer and restored model have no parameter objects in common. Create a new optimizer from `restored_model.parameters()` after placing that model on the target device, then load its optimizer state. Validate one resumed update against the restored parameters.

### 5. The cheat sheet labels training-set evaluation as validation

**Location:** [PyTorch_Cheat_Sheet.ipynb](/Users/kyawminthein/02_Job/02_PyTorch/PyTorch_Notebooks/PyTorch_Cheat_Sheet.ipynb:817), cell 46.

The “Validation / Inference Loop” evaluates `train_dataset` and prints `val loss` and `val accuracy`. It runs, but these metrics do not measure held-out performance. The later checkpoint even saves this result as `best_val_loss`.

Use an actual held-out dataset, or explicitly label this as training-set evaluation. The sprint and several full lessons already demonstrate proper held-out evaluation; make the quick reference consistent with them.

### 6. Final-window gradient accumulation has the wrong scale

**Location:** [06_training_loop.ipynb](/Users/kyawminthein/02_Job/02_PyTorch/PyTorch_Notebooks/06_training_loop.ipynb:706), cell 36, “Gradient Accumulation.”

The loader has 400 samples in batches of 32: twelve full batches and one batch of 16. With four accumulation steps, the last update contains only that final batch, but its loss is still divided by four. Its gradient is therefore **one quarter of the mean gradient for the actual final window**.

The code correctly performs a final update, but does not preserve mean-loss scaling for that window. For the demonstrated unweighted classification objective, accumulate summed losses and divide accumulated gradients by the actual number of samples in the window before stepping. Merely dividing by the number of microbatches is insufficient when their sizes differ. PyTorch's accumulation examples describe scaling at effective-batch granularity. [Official AMP examples](https://docs.pytorch.org/docs/main/notes/amp_examples.html)

### 7. A device assertion fails on CUDA despite correct placement

**Location:** [09_debugging_and_performance.ipynb](/Users/kyawminthein/02_Job/02_PyTorch/PyTorch_Notebooks/09_debugging_and_performance.ipynb:93), cell 16.

The assertion compares the output's concrete device, e.g. `cuda:0`, against `torch.device("cuda")`, which has no explicit index. These device objects compare unequal.

**Verified without CUDA hardware:** `torch.device("cuda") == torch.device("cuda:0")` is false. The full CUDA branch was not executed. Compare the output device with `next(device_model.parameters()).device`, as the Dataset notebook already does. [Official device semantics](https://docs.pytorch.org/docs/stable/tensor_attributes)

### 8. The anomaly-detection example does not demonstrate catching its error

**Location:** [09_debugging_and_performance.ipynb](/Users/kyawminthein/02_Job/02_PyTorch/PyTorch_Notebooks/09_debugging_and_performance.ipynb:211), cell 36.

Differentiating `sqrt(x)` at zero produces an infinite gradient. In the tested environment, `detect_anomaly()` does not raise, so the “Anomaly detection caught” branch never runs. Its documented automatic check concerns NaN gradients, not every infinity.

**Verified:** the cell completes with `anomaly_x.grad == inf`. Use a deliberately NaN-producing backward example, assert that the expected error is caught, and retain explicit `torch.isfinite` checks for infinities. [Official anomaly-detection documentation](https://docs.pytorch.org/docs/stable/autograd)

### 9. Some exercise validators give false confidence

**Locations:** [03_autograd.ipynb](/Users/kyawminthein/02_Job/02_PyTorch/PyTorch_Notebooks/03_autograd.ipynb:2825), cells 202–203; notebook 11, cell 47.

The autograd “Optimizer order” exercise asks for an optimizer step, but its validator checks only scalar loss and populated gradients. **Verified:** a solution that never calls `opt.step()` prints “Correct!” while all parameters remain unchanged. Add a before/after parameter comparison on controlled data.

The interview training-loop validator also begins by passing one `nn.Linear` as the model and a different `nn.Linear` to the optimizer. Its later learning check uses a correctly paired model and optimizer, but the initial probe is invalid and can reject a robust learner implementation. Use a single paired model/optimizer throughout.

## Additional improvements

| Area | Improvement |
|---|---|
| Notebook 12, cell 5 | `parameters_changed()` calls `model.cpu()` while inspecting it. This unexpectedly moves an accelerator model. Compare detached CPU copies of individual parameters and snapshots without moving the model. The helper is currently unused in the default run. |
| Notebook 06, cell 15, and reusable metric loops | State that `batch_mean × batch_size / sample_count` assumes the demonstrated unweighted, non-ignored, per-example objective. Weighted cross entropy and ignored/padded tokens need the loss's actual denominator. [CrossEntropyLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) |
| Notebook 10, manual attention helpers | All-masked attention rows produce NaNs; reproduced with an all-padding example. Document the input restriction or define and test the intended handling. Also distinguish the custom helper's `True = blocked` mask from functional SDPA's `True = allowed`. [Functional SDPA documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html) |
| Notebook 13 and cheat sheet | The shown tensor/dictionary/primitive training checkpoints can normally load with `weights_only=True`; a training checkpoint does not inherently require `False`. Teach the flag separately from checkpoint contents. [torch.load documentation](https://docs.pytorch.org/docs/stable/generated/torch.load.html) |
| Notebook 13 | Qualify “exact resume”: a general shuffled or stochastic training run also needs relevant RNG/sampler state, besides model, optimizer, and optional scheduler/scaler state. The runnable examples show continuation, not a comparison against an uninterrupted stochastic run. |
| Navigation | Notebook 13 points to missing `14_pytorch_training_techniques.ipynb`. Replace the dead-end instruction with an existing destination. |
| Example sizes | Notebook 10's early `Embedding(50_000,768)` allocates 38.4 million parameters, about 154 MB in float32, just to illustrate a lookup. Use a small runnable table and retain the large parameter count as an arithmetic exercise. |

## A shorter learning route

The collection contains **2,802 cells across 15 notebooks**. Completing every repeated quiz, flashcard, and syntax exercise is unnecessary for a first practical pass.

Use **notebook 15** as the main workbook. Its existing 10–12 focused-hour scope is a reasonable planning target for someone who already knows Python and basic ML concepts; a beginner may need longer. The objective is to build and explain a small working model, not master every PyTorch API.

| Stage | Time budget | What to do |
|---|---:|---|
| Shapes and tensors | 45–60 min | Notebook 15 §2. Predict shapes before execution. Consult 01/02 only for missed operations. Prioritize indexing, broadcasting, reductions, reshape, and `cat`/`stack`. |
| Autograd and a model | 60 min | Notebook 15 §§3–4. Write an MLP and demonstrate backward, gradient clearing, and a parameter update. Use 03/04 for explanations. |
| Loss contracts | 30–45 min | Notebook 15 §5. Memorize output/target shapes and dtypes for regression, binary, and multiclass tasks. |
| Training and validation | 2 hr | Notebook 15 §§6–8. Write both loops in blank cells. Check that training changes weights and evaluation leaves state unchanged. |
| Data loading | 45–60 min | Notebook 15 §9. Implement one Dataset, inspect one batch, and retain the final validation batch. |
| One complete project | 2 hr | Notebook 15 §10. Build the tabular classifier from a blank notebook, then compare against the reference. |
| Debugging | 45–60 min | Diagnose 6–8 drills you previously missed. Include wrong target shape, detached loss, missing step, and wrong optimizer parameters. |
| Save, reload, and infer | 30 min | Notebook 15 §12. Restore into a new model and compare outputs. Use notebook 13 for deeper questions. |
| Closed-book check | 60 min | Notebook 15 §13. Rebuild the pipeline without copying and explain every shape. Review mistakes only. |

Defer advanced `einsum`, manual multi-head attention, AMP, profiling, worker tuning, and the extra CNN/text/LM projects until the basic pipeline is comfortable or a specific role requires them. Use notebook 11 after repairing its answers and validators. Use the cheat sheet for lookup after repairing its validation/checkpoint patterns.

### Changes that would most improve the teaching format

1. Add a folder-level start page linking directly to the sprint, with prerequisites and setup instructions.
2. Separate **worked example → blank attempt → independent validator → hidden solution**. Several later notebooks put completed solutions directly in executable cells or validate the reference instead of the learner's draft.
3. Tag sections as **core / practice if needed / optional**, with short time budgets. Avoid marking nearly every advanced subject “Interview Essential.”
4. Use one recurring tabular problem to connect Dataset, model, loss, training, evaluation, and saving before introducing more architectures.
5. Test reference answers and completed exercise paths, not just the untouched starter notebooks. Include batch size 1, an incomplete final batch, and checking that the optimizer owns the model's parameters.

## Verification and limits

- Parsed all 15 top-level notebooks and executed their **904 code cells** sequentially, with a fresh Python process for each notebook. All completed without uncaught exceptions on the selected CPU paths.
- Environment: Python 3.9.6, PyTorch 2.8.0. Accelerator availability was disabled in the review processes, and CPU threads limited to one. This was direct execution of code-cell source, not a Jupyter rendering or kernel-integration test.
- Intentional caught exceptions and `None`/pending exercise branches remain expected. A successful starter run does **not** prove every completed exercise is correct. Selected completed paths and failure cases were tested separately above.
- Inspected executable examples, exercise structures, reference code, and selected quiz/explanation content. This is not an exhaustive certification of every verbal answer or every possible learner solution.
- CUDA, MPS, CUDA AMP, and multiprocessing DataLoader execution were not hardware-tested. Markdown solution rendering was not visually tested; the indentation defects were checked in the underlying supplied Python text.
- Notebook outputs and the user's completed answers were preserved. Only this report was added; tests and temporary output were kept outside the notebook folder.
