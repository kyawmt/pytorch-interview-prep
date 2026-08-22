/**
 * Answer validation.
 *
 * The site never executes Python. Everything here is text analysis:
 * normalization + accepted-answer matching + regex "required pattern" checks.
 * `eval()` and friends are deliberately absent.
 */

import type { Exercise, ProjectStep } from "./types";

export interface ValidationResult {
  correct: boolean;
  /** Headline shown in the feedback panel. */
  feedback: string;
  /** Index into `acceptedAnswers` when the match came from that list. */
  matchedSolution?: number;
  /** "so close" nudges: capitalization, missing parens, ... */
  nearMiss?: string;
  /** Required patterns the answer did not satisfy (human readable). */
  missing?: string[];
  /** Triggered anti-patterns. */
  warnings?: string[];
}

/** Strip a trailing `# comment`, ignoring `#` inside string literals. */
function stripComment(line: string): string {
  let inSingle = false;
  let inDouble = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === "'" && !inDouble) inSingle = !inSingle;
    else if (ch === '"' && !inSingle) inDouble = !inDouble;
    else if (ch === "#" && !inSingle && !inDouble) return line.slice(0, i);
  }
  return line;
}

/**
 * Canonical form used for comparison.
 *
 * - normalizes line endings and quotes
 * - drops comments, blank lines and trailing semicolons
 * - collapses runs of whitespace
 * - removes whitespace that sits next to punctuation/operators, so
 *   `torch.tensor([1,2,3])` === `torch.tensor([1, 2, 3])`
 *
 * Indentation is intentionally discarded: learners type these snippets into a
 * small editor and Python indentation rules are not what is being tested.
 */
export function normalizeCode(input: string): string {
  const lines = input
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .map((line) => stripComment(line).trim().replace(/;+$/, "").trim())
    .filter((line) => line.length > 0);

  return lines
    .map((line) =>
      line
        // single quotes -> double quotes so "model.pth" == 'model.pth'
        .replace(/'([^'\n]*)'/g, '"$1"')
        .replace(/\s+/g, " ")
        // drop spaces hugging any non-word character (operators, brackets, commas)
        .replace(/\s*([^\w\s"])\s*/g, "$1")
        .trim(),
    )
    .join("\n");
}

/** Aggressive form: only alphanumerics. Used for "you're very close" hints. */
function skeleton(input: string): string {
  return normalizeCode(input).toLowerCase().replace(/[^a-z0-9]/g, "");
}

/**
 * Answers to `fill-blank` are usually a bare identifier. We also accept the
 * fully-qualified form a learner may type out of habit (`x.numel()` for
 * `numel`) by keeping only the final attribute segment.
 */
function normalizeToken(input: string): string {
  const token = normalizeCode(input)
    .replace(/\(\s*\)$/, "")
    .replace(/^[.\s]+/, "")
    .replace(/[.\s]+$/, "")
    .trim();

  // Only collapse things that read as an attribute path, so numeric answers
  // such as "4.0" are left alone.
  if (/^[A-Za-z_][\w.]*$/.test(token) && token.includes(".")) {
    const last = token.slice(token.lastIndexOf(".") + 1);
    if (/^[A-Za-z_]\w*$/.test(last)) return last;
  }
  return token;
}

/**
 * Shape answers: accept `(32, 12288)`, `32,12288`, `[32, 12288]`,
 * `torch.Size([32, 12288])`, `32 x 12288`.
 * Scalars are normalized to `()`.
 */
export function normalizeShape(input: string): string {
  const numbers = input.match(/-?\d+/g);
  if (!numbers) return "()";
  return `(${numbers.join(", ")})`;
}

function patternToRegex(pattern: string): RegExp {
  return new RegExp(pattern, "i");
}

/** Turn a regex source into something readable for the "missing" list. */
export function describePattern(pattern: string): string {
  return pattern
    .replace(/\\s\*/g, " ")
    .replace(/\\s\+/g, " ")
    .replace(/\[\^\)\]\*/g, "...")
    .replace(/\.\*\?/g, "...")
    .replace(/\.\*/g, "...")
    .replace(/\\\./g, ".")
    .replace(/\\\(/g, "(")
    .replace(/\\\)/g, ")")
    .replace(/\\w\+/g, "name")
    .replace(/[()]\?/g, "")
    .replace(/\(\?:/g, "(")
    .trim();
}

function checkPatterns(normalized: string, patterns: string[] | undefined) {
  if (!patterns || patterns.length === 0) return { ok: true, missing: [] as string[] };
  const missing = patterns.filter((p) => !patternToRegex(p).test(normalized));
  return { ok: missing.length === 0, missing: missing.map(describePattern) };
}

function checkForbidden(normalized: string, exercise: Pick<Exercise, "forbiddenPatterns">) {
  if (!exercise.forbiddenPatterns) return [];
  return exercise.forbiddenPatterns
    .filter((f) => patternToRegex(f.pattern).test(normalized))
    .map((f) => f.message);
}

/** Shared "did the text match one of the accepted answers" routine. */
function matchAccepted(
  userNormalized: string,
  accepted: string[] | undefined,
  normalizer: (s: string) => string,
): number {
  if (!accepted) return -1;
  return accepted.findIndex((answer) => normalizer(answer) === userNormalized);
}

function nearMissHint(userAnswer: string, accepted: string[] | undefined): string | undefined {
  if (!accepted || accepted.length === 0) return undefined;
  const user = normalizeCode(userAnswer);
  const userLower = user.toLowerCase();
  const userSkeleton = skeleton(userAnswer);

  for (const answer of accepted) {
    const target = normalizeCode(answer);
    if (target.toLowerCase() === userLower) {
      return "Almost — check your capitalization. PyTorch names are case-sensitive.";
    }
    if (skeleton(answer) === userSkeleton) {
      return "Very close — check punctuation: brackets, commas or parentheses.";
    }
    if (target.replace(/\(\)/g, "") === user.replace(/\(\)/g, "")) {
      return "Very close — check whether the call needs parentheses.";
    }
    if (target.includes(user) && user.length > 3) {
      return "You're on the right track, but the answer is incomplete.";
    }
  }
  return undefined;
}

const CORRECT_MESSAGES = "Correct.";

/**
 * Validate any exercise type. `answer` is a string for text-based types,
 * a number for multiple choice, and a string[] (ordered blocks) for ordering.
 */
export function validateAnswer(
  answer: string | number | string[],
  exercise: Exercise,
): ValidationResult {
  switch (exercise.type) {
    case "multiple-choice":
      return validateMultipleChoice(answer as number, exercise);
    case "ordering":
      return validateOrdering(answer as string[], exercise);
    case "shape":
      return validateShape(String(answer ?? ""), exercise);
    case "fill-blank":
      return validateFillBlank(String(answer ?? ""), exercise);
    default:
      return validateCode(String(answer ?? ""), exercise);
  }
}

function validateMultipleChoice(answer: number, exercise: Exercise): ValidationResult {
  if (answer === undefined || answer === null || Number.isNaN(answer)) {
    return { correct: false, feedback: "Pick an option first." };
  }
  const correct = answer === exercise.correctOption;
  return {
    correct,
    feedback: correct ? CORRECT_MESSAGES : "Not quite — read the options again.",
  };
}

function validateOrdering(answer: string[], exercise: Exercise): ValidationResult {
  const expected = (exercise.correctOrder ?? []).map((i) => exercise.blocks?.[i] ?? "");
  if (!answer || answer.length === 0) {
    return { correct: false, feedback: "Add the steps in the order they run." };
  }
  if (answer.length !== expected.length) {
    return {
      correct: false,
      feedback:
        answer.length < expected.length
          ? `The sequence is incomplete — a correct loop uses ${expected.length} of these blocks.`
          : "Too many blocks — some of these do not belong in the loop.",
    };
  }
  const firstWrong = answer.findIndex((block, i) => normalizeCode(block) !== normalizeCode(expected[i]));
  if (firstWrong === -1) return { correct: true, feedback: CORRECT_MESSAGES };
  return {
    correct: false,
    feedback: `The first ${firstWrong} step${firstWrong === 1 ? "" : "s"} ${
      firstWrong === 0 ? "are" : firstWrong === 1 ? "is" : "are"
    } fine — step ${firstWrong + 1} is out of place.`,
  };
}

function validateShape(answer: string, exercise: Exercise): ValidationResult {
  const trimmed = answer.trim();
  if (!trimmed) return { correct: false, feedback: "Enter a shape, e.g. (32, 128)." };
  const user = normalizeShape(trimmed);
  const index = matchAccepted(user, exercise.acceptedAnswers, normalizeShape);
  if (index >= 0) return { correct: true, feedback: CORRECT_MESSAGES, matchedSolution: index };

  const expected = exercise.acceptedAnswers?.[0] ? normalizeShape(exercise.acceptedAnswers[0]) : "";
  const userDims = (user.match(/\d+/g) ?? []).length;
  const expectedDims = (expected.match(/\d+/g) ?? []).length;
  let nearMiss: string | undefined;
  if (userDims !== expectedDims && expectedDims > 0) {
    nearMiss = `The result has ${expectedDims} dimension${expectedDims === 1 ? "" : "s"}, you gave ${userDims}.`;
  } else if (expectedDims > 0) {
    nearMiss = "Right number of dimensions — one of the sizes is off.";
  }
  return { correct: false, feedback: "Not quite.", nearMiss };
}

function validateFillBlank(answer: string, exercise: Exercise): ValidationResult {
  const trimmed = answer.trim();
  if (!trimmed) return { correct: false, feedback: "Fill in the blank to check your answer." };
  const user = normalizeToken(trimmed);
  const index = matchAccepted(user, exercise.acceptedAnswers, normalizeToken);
  if (index >= 0) return { correct: true, feedback: CORRECT_MESSAGES, matchedSolution: index };

  const caseInsensitive = exercise.acceptedAnswers?.some(
    (a) => normalizeToken(a).toLowerCase() === user.toLowerCase(),
  );
  return {
    correct: false,
    feedback: "Not quite.",
    nearMiss: caseInsensitive ? "Right word — check the capitalization." : undefined,
  };
}

function validateCode(answer: string, exercise: Exercise): ValidationResult {
  const trimmed = answer.trim();
  if (!trimmed) return { correct: false, feedback: "Write some code to check your answer." };

  const user = normalizeCode(trimmed);
  const warnings = checkForbidden(user, exercise);
  const hasAccepted = (exercise.acceptedAnswers?.length ?? 0) > 0;

  if (hasAccepted) {
    const index = matchAccepted(user, exercise.acceptedAnswers, normalizeCode);
    if (index >= 0) {
      return {
        correct: warnings.length === 0,
        feedback: warnings.length === 0 ? CORRECT_MESSAGES : "That runs, but look at the note below.",
        matchedSolution: index,
        warnings: warnings.length ? warnings : undefined,
      };
    }
    const { missing } = checkPatterns(user, exercise.requiredPatterns);
    return {
      correct: false,
      feedback: "Not quite.",
      nearMiss: nearMissHint(trimmed, exercise.acceptedAnswers),
      missing: missing.length ? missing : undefined,
      warnings: warnings.length ? warnings : undefined,
    };
  }

  // Pattern-driven exercises ("write the training loop from memory"): the
  // structure matters, variable names do not.
  const { ok, missing } = checkPatterns(user, exercise.requiredPatterns);
  if (ok) {
    return {
      correct: warnings.length === 0,
      feedback:
        warnings.length === 0
          ? "Correct — every required piece is there."
          : "The structure is right, but see the note below.",
      warnings: warnings.length ? warnings : undefined,
    };
  }
  return {
    correct: false,
    feedback: `Missing ${missing.length} required piece${missing.length === 1 ? "" : "s"}.`,
    missing,
    warnings: warnings.length ? warnings : undefined,
  };
}

/** Project steps reuse the same engine via a light adapter. */
export function validateProjectStep(
  answer: string | number,
  step: ProjectStep,
): ValidationResult {
  const asExercise = {
    id: step.id,
    title: step.title,
    topic: "training",
    level: 1,
    difficulty: "medium",
    importance: "high",
    type: step.kind === "multiple-choice" ? "multiple-choice" : step.kind,
    question: step.brief,
    acceptedAnswers: step.acceptedAnswers,
    requiredPatterns: step.requiredPatterns,
    options: step.options,
    correctOption: step.correctOption,
    solution: step.solution,
    explanation: step.explanation,
  } as Exercise;
  return validateAnswer(answer, asExercise);
}
