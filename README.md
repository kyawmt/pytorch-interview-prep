# PyTorch Prep

An interactive PyTorch learning site built for **AI Engineer / ML Engineer interview
preparation**. It is designed around active recall rather than passive reading: see a
concept, hide the syntax, type it from memory, check it, and review what you missed.

Nothing is executed. There is no Python backend, no Pyodide and no remote sandbox —
answers are validated with a text-analysis engine that accepts any reasonable spelling
of the right answer.

```bash
npm install
npm run dev          # http://localhost:3000
```

---

## What is in it

| Section | Route | Content |
| --- | --- | --- |
| Dashboard | `/` | Progress, streak, weak areas, continue-learning |
| Cheat Sheet | `/cheatsheet`, `/cheatsheet/[topic]` | **100 syntax cards** with examples and interview notes |
| Practice | `/practice`, `/practice/[topic]` | **180 exercises** across 8 levels and 7 formats |
| Interview Questions | `/interview` | **58 questions** with a spoken answer + deep explanation |
| Mini Projects | `/projects/[project]` | **7 guided builds**, 58 fill-in-the-code steps |
| Shape Playground | `/shapes` | 32 tensor-shape reasoning drills |
| Training Loop Builder | `/builder` | 6 ordering drills, including deliberately wrong blocks |
| Write From Memory | `/memory` | 12 structure-graded reproduction tasks |
| Interview Challenge | `/challenge` | Timed 20-question quiz with a per-topic breakdown |
| Common Mistakes | `/mistakes` | 26 real bugs, framed by the symptom you would observe |
| 15-Minute Review | `/rapid-review` | 30 flashcards to read before an interview |
| Review | `/review` | Auto-collected wrong / hinted / revealed / flagged questions |
| Progress | `/progress` | Topic mastery, learning path, export & import |
| Daily Practice | `/daily` | 5 syntax + 3 concept + 2 interview questions, fixed per day |

Coverage runs from tensor creation and broadcasting through autograd, `nn.Module`,
losses, optimizers, the training loop, `Dataset`/`DataLoader`, devices, checkpoints,
mixed precision and transformer building blocks.

---

## Stack

- **Next.js 16** (App Router) · **React 19** · **TypeScript** · **Tailwind CSS 4**
- No database, no state-management library, no runtime dependencies beyond the above.
- Progress lives in `localStorage`; the storage layer is isolated so a real backend can
  replace it without touching the UI.

Syntax highlighting is a ~120-line Python tokenizer (`src/lib/highlight.ts`) rather than
Monaco or Shiki — snippets here are rarely longer than 15 lines, and it keeps the bundle
small and SSR-clean. The answer editor is a transparent `<textarea>` layered over a
highlighted `<pre>`.

### Scripts

```bash
npm run dev              # dev server
npm run build            # production build (56 prerendered routes)
npm run start            # serve the production build
npm run typecheck        # tsc --noEmit
npm run lint             # eslint
npm run check:content    # validate every exercise against its own validator
npm run check            # typecheck + content check
```

---

## Project layout

```text
src/
├─ app/                        # routes (App Router)
│  ├─ page.tsx                 # dashboard
│  ├─ cheatsheet/[topic]/      # topic pages (statically generated)
│  ├─ practice/[topic]/
│  ├─ projects/[project]/
│  └─ …                        # interview, review, progress, drills
│
├─ components/
│  ├─ layout/                  # AppShell, Sidebar, SearchDialog, ThemeToggle
│  ├─ cheatsheet/              # CheatSheetCard, CheatSheetBrowser
│  ├─ exercises/               # ExerciseCard, ExerciseRunner, CodeEditor, filters
│  ├─ interview/               # InterviewQuestionCard
│  ├─ projects/                # ProjectRunner, ProjectStepCard
│  ├─ progress/                # ProgressBar, StatsCard, TopicProgress
│  └─ common/                  # CodeBlock, Badge, Button, PageHeader, InlineCode
│
├─ data/                       # ALL content lives here as typed data
│  ├─ topics.ts                # topic list, levels, learning path
│  ├─ cheatsheet/              # tensors, shapes, math, autograd, nn, training, data, gpu
│  ├─ exercises/               # one file per level
│  ├─ interview/               # fundamentals, training, advanced
│  ├─ projects/                # one file per mini project
│  ├─ mistakes.ts  revision.ts  shape-drills.ts  memory-drills.ts  builder-drills.ts
│
├─ lib/
│  ├─ types.ts                 # every content + progress model
│  ├─ validation.ts            # the answer-checking engine
│  ├─ storage.ts               # localStorage store (subscribe/getSnapshot)
│  ├─ progress.ts              # derived stats: mastery, weak areas, review queue
│  ├─ search.ts                # global ⌘K search index
│  ├─ session.ts               # daily session + challenge builders
│  └─ highlight.ts  utils.ts
│
├─ hooks/                      # useAppState, useKeyboardShortcuts, useCopy
└─ scripts/check-content.ts    # content linter (npm run check:content)
```

---

## The validation engine

`validateAnswer(answer, exercise)` in `src/lib/validation.ts` returns:

```ts
{ correct: boolean; feedback: string; matchedSolution?: number;
  nearMiss?: string; missing?: string[]; warnings?: string[] }
```

**Normalization** (`normalizeCode`) makes formatting irrelevant. It normalizes line
endings and quotes, drops comments, blank lines and trailing semicolons, collapses
whitespace, and removes spaces adjacent to punctuation — so all of these are equal:

```python
x = torch.tensor([1, 2, 3])
x=torch.tensor([1,2,3])
x = torch.tensor([1,2,3]);      # trailing comment
```

**Two validation modes:**

- **`acceptedAnswers` present** → authoritative. The answer must normalize to one of
  them. `requiredPatterns` are then used only to generate targeted feedback
  (*"your answer is missing `optimizer.zero_grad()`"*).
- **No `acceptedAnswers`** → `requiredPatterns` become authoritative. This is how
  `memory`-type exercises work: `outputs = network(inputs)` and `pred = model(X)` both
  pass, because only the structure is graded.

`forbiddenPatterns` flag anti-patterns even when the rest is right (e.g. a softmax
before `CrossEntropyLoss`).

Per-type normalizers handle the rest: shapes accept `(32, 128)`, `32,128`,
`[32, 128]` and `torch.Size([32, 128])`; fill-in-the-blank accepts `numel`,
`numel()` and `x.numel()`.

There is no `eval()` and no code execution anywhere.

---

## Adding content

All content is plain typed data — no JSX, no per-exercise components.

**A new exercise** — append to the right file in `src/data/exercises/`:

```ts
{
  id: "ex-tensor-99",              // must be globally unique
  title: "Create a zero tensor",
  topic: "tensors",                // a TopicId from src/data/topics.ts
  level: 1,                        // 1-8, drives the Practice page grouping
  difficulty: "easy",              // easy | medium | hard
  importance: "high",              // drives the 🔥 badge and the priority filter
  type: "code",                    // code | fill-blank | multiple-choice
                                   // shape | debug | ordering | memory
  question: "Create a 3 × 4 tensor filled with zeros.",
  context: "import torch",         // optional code shown above the prompt
  acceptedAnswers: ["x = torch.zeros(3, 4)", "x = torch.zeros((3, 4))"],
  requiredPatterns: ["torch\\.zeros"],
  hint: "torch.zeros() takes the dimensions positionally.",
  solution: "x = torch.zeros(3, 4)",
  explanation: "torch.zeros(3, 4) gives 3 rows and 4 columns, float32 by default.",
  interviewNote: "Default dtype is float32, unlike torch.tensor().",
}
```

Then run the content linter, which catches the mistakes that are invisible until a
learner hits them:

```bash
npm run check:content
```

It verifies that every exercise's own reference solution passes its validator, that
every `acceptedAnswer` validates, that `requiredPatterns` actually match the solution,
and that ids are unique and cross-references resolve.

**A new cheat-sheet card** goes in `src/data/cheatsheet/<topic>.ts`. Set
`relatedExercises: ["ex-tensor-99"]` to wire up its *Practise* button.

**A new topic** goes in `src/data/topics.ts` (add to `TOPICS`, and to `LEVELS` /
`LEARNING_PATH` as appropriate). Cheat-sheet and practice pages for it are generated
automatically.

**A new mini project** is a file in `src/data/projects/` exported from that folder's
`index.ts`; its route, progress tracking and step navigation come for free.

---

## Progress and spaced repetition

State is a single object in `localStorage` under `pytorch-prep:v1`, read through a small
external store (`subscribe` / `getSnapshot`) consumed with `useSyncExternalStore`. The
server snapshot is always the empty state, so there are no hydration mismatches.

```ts
interface ExerciseProgress {
  exerciseId: string;
  attempts: number;
  correctAttempts: number;
  correct: boolean;          // was the most recent attempt correct?
  usedHint: boolean;
  revealedAnswer: boolean;
  lastAttempt: string;
  reviewState: "new" | "learning" | "review" | "mastered";
  dueAt?: string;
  intervalDays: number;
  flagged?: boolean;
}
```

After answering, rating a card **Again / Hard / Good / Easy** schedules it with a simple
SM-2-flavoured interval (0 / ×1.2 / ×2.2 / ×3 days); at 21+ days it is marked *mastered*.
Anything answered wrong, hinted, revealed or rated *Again* lands in `/review`
automatically.

**Mastery** blends coverage with accuracy and discounts questions that needed a hint
(×0.85) or a revealed answer (×0.7), so it reflects what you could reproduce under
interview pressure rather than what you have merely seen.

Progress can be exported and re-imported as JSON from `/progress`.

### Moving to a database later

`src/lib/storage.ts` is the only module that touches `localStorage`. Everything else
either reads derived selectors from `src/lib/progress.ts` or calls the named mutations
(`recordAttempt`, `scheduleReview`, `setProjectStep`, …). Swapping the read/write
internals for API calls is the whole migration.

---

## Keyboard shortcuts

| Keys | Action |
| --- | --- |
| `⌘K` / `Ctrl+K`, `/` | Global search |
| `⌘↵` / `Ctrl+↵` | Check answer |
| `H` | Show hint |
| `N` | Next question |
| `B` | Bookmark |
| `↑ ↓ ↵ Esc` | Navigate search results |

Single-letter shortcuts are suppressed while you are typing in the editor; `⌘`-combos
still fire.

---

## Design notes

- **Themes**: light / dark / system, applied by a blocking inline script before
  hydration so there is no flash. Dark mode is the primary target — long sessions.
- **Responsive**: sidebar becomes a drawer under `lg`; code blocks scroll horizontally
  rather than wrapping; cards stack.
- **Accessibility**: semantic landmarks, a skip link, labelled controls, `aria-pressed`
  on toggles, `aria-live` feedback regions, visible focus rings, and full keyboard
  operation of the ordering exercises (click/arrow, not drag-and-drop).
- Animations are subtle and honour `prefers-reduced-motion`.
