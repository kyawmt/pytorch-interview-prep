/**
 * Shared content + progress types.
 *
 * All learning content (cheat sheet, exercises, interview questions, projects)
 * is plain data living under `src/data`. Nothing here depends on React, so the
 * same models can later be served from a database without touching the UI.
 */

export type Difficulty = "easy" | "medium" | "hard";
export type Importance = "high" | "medium" | "low";

export type ExerciseType =
  | "code"
  | "fill-blank"
  | "multiple-choice"
  | "shape"
  | "debug"
  | "ordering"
  | "memory";

export type TopicId =
  | "tensors"
  | "shapes"
  | "broadcasting"
  | "math"
  | "autograd"
  | "nn"
  | "losses"
  | "optimizers"
  | "training"
  | "evaluation"
  | "data"
  | "gpu"
  | "saving"
  | "performance"
  | "transformers"
  | "debugging"
  | "interop";

export interface Topic {
  id: TopicId;
  title: string;
  /** Short blurb used on index cards. */
  summary: string;
  /** Interview weight, drives the "high priority only" filter and badges. */
  importance: Importance;
  /** Position in the recommended learning path (1-based). */
  pathOrder: number;
  icon: string;
}

/** A single cheat-sheet card: concept -> syntax -> example -> interview note. */
export interface CheatSheetEntry {
  id: string;
  topic: TopicId;
  /** Sub-heading inside a topic page, e.g. "Tensor Creation". */
  section: string;
  title: string;
  /** One or two sentences. What it does / why it matters. */
  description: string;
  /** Canonical signature(s), shown as the "syntax" line. */
  syntax: string;
  /** Runnable-looking snippet illustrating the concept. */
  example: string;
  /** Printed output or resulting shape, when useful. */
  result?: string;
  /** Extra context an interviewer would probe. */
  interviewNote?: string;
  importance: Importance;
  /** Free-text keywords that feed global search. */
  tags?: string[];
  /** Exercise ids that drill this entry ("Practice" button). */
  relatedExercises?: string[];
}

export interface ForbiddenPattern {
  /** Regex source, matched against the normalized answer. */
  pattern: string;
  /** Shown to the learner when the pattern is present. */
  message: string;
}

export interface Exercise {
  id: string;
  title: string;
  topic: TopicId;
  /** Progressive level 1-8 (see the Practice page grouping). */
  level: number;
  difficulty: Difficulty;
  importance: Importance;
  type: ExerciseType;

  /** The prompt shown to the learner (plain text / light markdown). */
  question: string;
  /** Optional code shown above the prompt (setup, buggy snippet, ...). */
  context?: string;
  /** Pre-filled editor content. */
  starterCode?: string;

  /**
   * Authoritative answers for `code` / `fill-blank` / `shape` / `debug`.
   * When present, an answer is correct if it normalizes to one of these.
   */
  acceptedAnswers?: string[];
  /**
   * Regex sources every answer must contain. Authoritative when there are no
   * `acceptedAnswers` (used by `memory` exercises); otherwise only used to
   * generate targeted "you're missing X" feedback.
   */
  requiredPatterns?: string[];
  /** Anti-patterns worth calling out even when the rest is right. */
  forbiddenPatterns?: ForbiddenPattern[];

  /** multiple-choice */
  options?: string[];
  correctOption?: number;

  /** ordering: `blocks` may contain distractors that are not in `correctOrder` */
  blocks?: string[];
  correctOrder?: number[];

  hint?: string;
  solution: string;
  explanation: string;
  interviewNote?: string;
  tags?: string[];
}

export interface InterviewQuestion {
  id: string;
  category: string;
  topic: TopicId;
  question: string;
  /** 1-3 sentences you could actually say out loud in an interview. */
  shortAnswer: string;
  /** Study-grade detail. Rendered as paragraphs. */
  detailedAnswer: string;
  /** Related PyTorch snippet. */
  code?: string;
  importance: Importance;
  followUps?: string[];
  tags?: string[];
}

export type ProjectStepKind = "code" | "fill-blank" | "multiple-choice" | "shape";

export interface ProjectStep {
  id: string;
  title: string;
  /** What the learner is building in this step. */
  brief: string;
  kind: ProjectStepKind;
  context?: string;
  starterCode?: string;
  acceptedAnswers?: string[];
  requiredPatterns?: string[];
  options?: string[];
  correctOption?: number;
  hint?: string;
  solution: string;
  explanation: string;
}

export interface MiniProject {
  slug: string;
  title: string;
  tagline: string;
  difficulty: Difficulty;
  estimatedMinutes: number;
  topics: TopicId[];
  /** Rendered as an ASCII-ish pipeline diagram on the project page. */
  outline: string[];
  overview: string;
  steps: ProjectStep[];
}

export interface CommonMistake {
  id: string;
  title: string;
  topic: TopicId;
  importance: Importance;
  /** The code as people usually write it (wrong or risky). */
  wrong: string;
  /** The corrected version. */
  right: string;
  /** Symptom the learner will actually observe. */
  symptom: string;
  explanation: string;
  tags?: string[];
}

/** One card in the 15-minute rapid revision deck. */
export interface RevisionCard {
  id: string;
  topic: TopicId;
  prompt: string;
  answer: string;
  code?: string;
}

/* ------------------------------------------------------------------ */
/* Progress                                                            */
/* ------------------------------------------------------------------ */

export type ReviewState = "new" | "learning" | "review" | "mastered";
export type Rating = "again" | "hard" | "good" | "easy";

export interface ExerciseProgress {
  exerciseId: string;
  attempts: number;
  correctAttempts: number;
  /** Was the most recent attempt correct? */
  correct: boolean;
  usedHint: boolean;
  revealedAnswer: boolean;
  lastAttempt: string;
  reviewState: ReviewState;
  /** ISO date the card is next due (spaced repetition). */
  dueAt?: string;
  /** Current interval in days. */
  intervalDays: number;
  /** Manually flagged with "Need Review". */
  flagged?: boolean;
}

export type InterviewStatus = "unseen" | "knew" | "review";

export interface InterviewProgress {
  id: string;
  status: InterviewStatus;
  lastSeen: string;
  timesSeen: number;
}

export interface ProjectProgress {
  slug: string;
  steps: Record<string, boolean>;
  updatedAt: string;
}

export interface ChallengeResult {
  id: string;
  date: string;
  score: number;
  total: number;
  durationSeconds: number;
  byTopic: Record<string, { correct: number; total: number }>;
  missedExerciseIds: string[];
}

export type ThemePreference = "light" | "dark" | "system";

export interface Preferences {
  theme: ThemePreference;
  /** Cheat sheet starts with code hidden (active-recall mode). */
  recallMode: boolean;
  highPriorityOnly: boolean;
}

export interface AppState {
  version: number;
  exercises: Record<string, ExerciseProgress>;
  interview: Record<string, InterviewProgress>;
  projects: Record<string, ProjectProgress>;
  /** Namespaced ids: "cheat:xxx", "ex:xxx", "iq:xxx". */
  bookmarks: string[];
  challenges: ChallengeResult[];
  streak: {
    current: number;
    longest: number;
    lastActiveDate: string | null;
    /** ISO dates (YYYY-MM-DD) with at least one attempt. */
    activeDays: string[];
  };
  prefs: Preferences;
}
