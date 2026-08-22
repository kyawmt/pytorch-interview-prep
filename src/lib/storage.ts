/**
 * localStorage-backed application state.
 *
 * Implemented as a tiny external store (subscribe / getSnapshot) so components
 * can read it with `useSyncExternalStore` without a global state library and
 * without hydration mismatches: the server snapshot is always the empty state.
 *
 * Swapping this file for API calls is the only change needed to move progress
 * to a real database later — nothing else imports localStorage.
 */

import type {
  AppState,
  ChallengeResult,
  ExerciseProgress,
  InterviewProgress,
  InterviewStatus,
  Preferences,
  Rating,
  ReviewState,
} from "./types";

const STORAGE_KEY = "pytorch-prep:v1";
const STATE_VERSION = 1;

export function createEmptyState(): AppState {
  return {
    version: STATE_VERSION,
    exercises: {},
    interview: {},
    projects: {},
    bookmarks: [],
    challenges: [],
    streak: { current: 0, longest: 0, lastActiveDate: null, activeDays: [] },
    prefs: { theme: "system", recallMode: false, highPriorityOnly: false },
  };
}

const EMPTY_STATE = createEmptyState();

let state: AppState | null = null;
const listeners = new Set<() => void>();

function isBrowser() {
  return typeof window !== "undefined";
}

function read(): AppState {
  if (!isBrowser()) return EMPTY_STATE;
  if (state) return state;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      state = createEmptyState();
    } else {
      const parsed = JSON.parse(raw) as Partial<AppState>;
      state = { ...createEmptyState(), ...parsed, prefs: { ...createEmptyState().prefs, ...parsed.prefs } };
    }
  } catch {
    state = createEmptyState();
  }
  return state;
}

function write(next: AppState) {
  state = next;
  if (isBrowser()) {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      // Quota errors are non-fatal: progress simply won't persist this session.
    }
  }
  listeners.forEach((listener) => listener());
}

export function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getSnapshot(): AppState {
  return read();
}

export function getServerSnapshot(): AppState {
  return EMPTY_STATE;
}

/** All mutations funnel through here so every change persists + notifies. */
export function update(mutate: (draft: AppState) => AppState | void) {
  const current = read();
  const draft: AppState = structuredClone(current);
  const result = mutate(draft);
  write(result ?? draft);
}

export function resetAll() {
  write(createEmptyState());
}

/* ------------------------------------------------------------------ */
/* Dates + streak                                                      */
/* ------------------------------------------------------------------ */

export function todayKey(date = new Date()): string {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(
    date.getDate(),
  ).padStart(2, "0")}`;
}

function daysBetween(a: string, b: string): number {
  const toDate = (key: string) => {
    const [y, m, d] = key.split("-").map(Number);
    return Date.UTC(y, m - 1, d);
  };
  return Math.round((toDate(b) - toDate(a)) / 86_400_000);
}

function touchStreak(draft: AppState) {
  const today = todayKey();
  const { streak } = draft;
  if (streak.lastActiveDate === today) return;
  if (streak.lastActiveDate && daysBetween(streak.lastActiveDate, today) === 1) {
    streak.current += 1;
  } else {
    streak.current = 1;
  }
  streak.lastActiveDate = today;
  streak.longest = Math.max(streak.longest, streak.current);
  if (!streak.activeDays.includes(today)) streak.activeDays.push(today);
  // keep the heat-map bounded
  if (streak.activeDays.length > 400) streak.activeDays = streak.activeDays.slice(-400);
}

/* ------------------------------------------------------------------ */
/* Exercises                                                           */
/* ------------------------------------------------------------------ */

export function emptyExerciseProgress(exerciseId: string): ExerciseProgress {
  return {
    exerciseId,
    attempts: 0,
    correctAttempts: 0,
    correct: false,
    usedHint: false,
    revealedAnswer: false,
    lastAttempt: "",
    reviewState: "new",
    intervalDays: 0,
  };
}

function ensureExercise(draft: AppState, exerciseId: string): ExerciseProgress {
  if (!draft.exercises[exerciseId]) draft.exercises[exerciseId] = emptyExerciseProgress(exerciseId);
  return draft.exercises[exerciseId];
}

export function recordAttempt(exerciseId: string, correct: boolean) {
  update((draft) => {
    const entry = ensureExercise(draft, exerciseId);
    entry.attempts += 1;
    if (correct) entry.correctAttempts += 1;
    entry.correct = correct;
    entry.lastAttempt = new Date().toISOString();
    if (entry.reviewState === "new") entry.reviewState = correct ? "review" : "learning";
    else if (!correct && entry.reviewState === "mastered") entry.reviewState = "review";
    if (correct && entry.flagged && entry.correctAttempts > 1) entry.flagged = false;
    touchStreak(draft);
  });
}

export function markHintUsed(exerciseId: string) {
  update((draft) => {
    ensureExercise(draft, exerciseId).usedHint = true;
  });
}

export function markAnswerRevealed(exerciseId: string) {
  update((draft) => {
    const entry = ensureExercise(draft, exerciseId);
    entry.revealedAnswer = true;
    if (entry.reviewState === "new") entry.reviewState = "learning";
  });
}

export function toggleFlag(exerciseId: string) {
  update((draft) => {
    const entry = ensureExercise(draft, exerciseId);
    entry.flagged = !entry.flagged;
  });
}

export function resetExercise(exerciseId: string) {
  update((draft) => {
    delete draft.exercises[exerciseId];
  });
}

/* ------------------------------------------------------------------ */
/* Spaced repetition                                                   */
/* ------------------------------------------------------------------ */

/** Intentionally simple SM-2-flavoured scheduler. Intervals are in days. */
export function scheduleReview(exerciseId: string, rating: Rating) {
  update((draft) => {
    const entry = ensureExercise(draft, exerciseId);
    const previous = entry.intervalDays || 0;
    let interval: number;
    let reviewState: ReviewState;

    switch (rating) {
      case "again":
        interval = 0;
        reviewState = "learning";
        break;
      case "hard":
        interval = previous < 1 ? 1 : Math.max(1, Math.round(previous * 1.2));
        reviewState = "learning";
        break;
      case "good":
        interval = previous < 1 ? 2 : Math.round(previous * 2.2);
        reviewState = "review";
        break;
      default:
        interval = previous < 1 ? 4 : Math.round(previous * 3);
        reviewState = "review";
        break;
    }

    entry.intervalDays = interval;
    entry.reviewState = interval >= 21 ? "mastered" : reviewState;
    const due = new Date();
    due.setDate(due.getDate() + interval);
    entry.dueAt = due.toISOString();
    if (rating === "again") entry.flagged = true;
    if (rating === "easy") entry.flagged = false;
    touchStreak(draft);
  });
}

export function isDue(entry: ExerciseProgress | undefined, now = new Date()): boolean {
  if (!entry) return true;
  if (!entry.dueAt) return true;
  return new Date(entry.dueAt).getTime() <= now.getTime();
}

/* ------------------------------------------------------------------ */
/* Interview questions                                                 */
/* ------------------------------------------------------------------ */

export function setInterviewStatus(id: string, status: InterviewStatus) {
  update((draft) => {
    const existing: InterviewProgress = draft.interview[id] ?? {
      id,
      status: "unseen",
      lastSeen: "",
      timesSeen: 0,
    };
    draft.interview[id] = {
      ...existing,
      status,
      lastSeen: new Date().toISOString(),
      timesSeen: existing.timesSeen + 1,
    };
    touchStreak(draft);
  });
}

/* ------------------------------------------------------------------ */
/* Projects                                                            */
/* ------------------------------------------------------------------ */

export function setProjectStep(slug: string, stepId: string, done: boolean) {
  update((draft) => {
    const existing = draft.projects[slug] ?? { slug, steps: {}, updatedAt: "" };
    existing.steps = { ...existing.steps, [stepId]: done };
    existing.updatedAt = new Date().toISOString();
    draft.projects[slug] = existing;
    touchStreak(draft);
  });
}

export function resetProject(slug: string) {
  update((draft) => {
    delete draft.projects[slug];
  });
}

/* ------------------------------------------------------------------ */
/* Bookmarks, challenges, preferences                                  */
/* ------------------------------------------------------------------ */

export function toggleBookmark(key: string) {
  update((draft) => {
    draft.bookmarks = draft.bookmarks.includes(key)
      ? draft.bookmarks.filter((b) => b !== key)
      : [...draft.bookmarks, key];
  });
}

export function saveChallengeResult(result: ChallengeResult) {
  update((draft) => {
    draft.challenges = [result, ...draft.challenges].slice(0, 25);
    touchStreak(draft);
  });
}

export function setPreference<K extends keyof Preferences>(key: K, value: Preferences[K]) {
  update((draft) => {
    draft.prefs[key] = value;
  });
}

export function exportState(): string {
  return JSON.stringify(read(), null, 2);
}

export function importState(json: string): boolean {
  try {
    const parsed = JSON.parse(json) as AppState;
    if (typeof parsed !== "object" || parsed === null) return false;
    write({ ...createEmptyState(), ...parsed });
    return true;
  } catch {
    return false;
  }
}
