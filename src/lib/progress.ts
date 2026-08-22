/**
 * Derived progress selectors.
 *
 * Everything here is a pure function of (AppState, content), so the same
 * calculations power the dashboard, the progress page and the review queue.
 */

import { EXERCISES } from "@/data/exercises";
import { INTERVIEW_QUESTIONS } from "@/data/interview";
import { PROJECTS, TOTAL_PROJECT_STEPS } from "@/data/projects";
import { LEARNING_PATH, LEVELS, TOPICS, TOPIC_MAP } from "@/data/topics";
import { isDue } from "./storage";
import type { AppState, Exercise, ExerciseProgress, TopicId } from "./types";
import { clamp, percent } from "./utils";

export interface TopicStats {
  topic: TopicId;
  title: string;
  total: number;
  attempted: number;
  correct: number;
  /** Coverage: share of the topic's exercises answered correctly at least once. */
  completion: number;
  /** Accuracy over attempted exercises. */
  accuracy: number;
  /** Blended score used for the mastery bars. */
  mastery: number;
  needsReview: number;
}

function isSolved(entry: ExerciseProgress | undefined): boolean {
  return Boolean(entry && entry.correctAttempts > 0);
}

/**
 * Mastery blends coverage and accuracy, then applies a small penalty for hints
 * and revealed answers so "I looked it up" does not read as "I know it".
 */
function masteryScore(items: Exercise[], state: AppState): number {
  if (items.length === 0) return 0;
  let score = 0;
  for (const exercise of items) {
    const entry = state.exercises[exercise.id];
    if (!entry || entry.attempts === 0) continue;
    let value = entry.correctAttempts > 0 ? 1 : 0.15;
    if (entry.revealedAnswer) value *= 0.7;
    else if (entry.usedHint) value *= 0.85;
    if (entry.reviewState === "mastered") value = Math.min(1, value * 1.1);
    if (entry.attempts > 0) {
      const hitRate = entry.correctAttempts / entry.attempts;
      value *= 0.6 + 0.4 * hitRate;
    }
    score += value;
  }
  return clamp(Math.round((score / items.length) * 100), 0, 100);
}

export function topicStats(state: AppState, topic: TopicId): TopicStats {
  const items = EXERCISES.filter((exercise) => exercise.topic === topic);
  const entries = items.map((item) => state.exercises[item.id]).filter(Boolean) as ExerciseProgress[];
  const attempted = entries.length;
  const solved = items.filter((item) => isSolved(state.exercises[item.id])).length;
  const totalAttempts = entries.reduce((sum, entry) => sum + entry.attempts, 0);
  const totalCorrect = entries.reduce((sum, entry) => sum + entry.correctAttempts, 0);

  return {
    topic,
    title: TOPIC_MAP[topic]?.title ?? topic,
    total: items.length,
    attempted,
    correct: solved,
    completion: percent(solved, items.length),
    accuracy: totalAttempts > 0 ? percent(totalCorrect, totalAttempts) : 0,
    mastery: masteryScore(items, state),
    needsReview: items.filter((item) => needsReview(state.exercises[item.id])).length,
  };
}

export function allTopicStats(state: AppState): TopicStats[] {
  return TOPICS.map((topic) => topicStats(state, topic.id));
}

export function levelStats(state: AppState) {
  return LEVELS.map((level) => {
    const items = EXERCISES.filter((exercise) => exercise.level === level.level);
    const solved = items.filter((item) => isSolved(state.exercises[item.id])).length;
    return {
      ...level,
      total: items.length,
      solved,
      completion: percent(solved, items.length),
    };
  });
}

/** An exercise belongs in the review queue if it went badly or was flagged. */
export function needsReview(entry: ExerciseProgress | undefined): boolean {
  if (!entry) return false;
  if (entry.flagged) return true;
  if (entry.attempts === 0) return false;
  if (!entry.correct) return true;
  if (entry.revealedAnswer && entry.correctAttempts < 2) return true;
  if (entry.usedHint && entry.correctAttempts < 2) return true;
  return false;
}

export type ReviewFilter = "all" | "wrong" | "hint" | "revealed" | "flagged" | "due";

export function reviewQueue(state: AppState, filter: ReviewFilter = "all"): Exercise[] {
  return EXERCISES.filter((exercise) => {
    const entry = state.exercises[exercise.id];
    if (!entry) return false;
    switch (filter) {
      case "wrong":
        return entry.attempts > 0 && !entry.correct;
      case "hint":
        return entry.usedHint;
      case "revealed":
        return entry.revealedAnswer;
      case "flagged":
        return Boolean(entry.flagged);
      case "due":
        return entry.reviewState !== "new" && isDue(entry);
      default:
        return needsReview(entry);
    }
  });
}

export interface OverallStats {
  totalExercises: number;
  attempted: number;
  solved: number;
  accuracy: number;
  overallProgress: number;
  streak: number;
  longestStreak: number;
  topicsMastered: number;
  needingReview: number;
  interviewKnown: number;
  interviewTotal: number;
  projectSteps: number;
  projectStepsTotal: number;
  projectsCompleted: number;
  dueCount: number;
}

export function overallStats(state: AppState): OverallStats {
  const entries = Object.values(state.exercises);
  const attempted = entries.filter((entry) => entry.attempts > 0).length;
  const solved = entries.filter((entry) => entry.correctAttempts > 0).length;
  const totalAttempts = entries.reduce((sum, entry) => sum + entry.attempts, 0);
  const totalCorrect = entries.reduce((sum, entry) => sum + entry.correctAttempts, 0);
  const stats = allTopicStats(state);

  const projectSteps = Object.values(state.projects).reduce(
    (sum, project) => sum + Object.values(project.steps).filter(Boolean).length,
    0,
  );
  const projectsCompleted = PROJECTS.filter((project) => {
    const progress = state.projects[project.slug];
    if (!progress) return false;
    return project.steps.every((step) => progress.steps[step.id]);
  }).length;

  return {
    totalExercises: EXERCISES.length,
    attempted,
    solved,
    accuracy: totalAttempts > 0 ? percent(totalCorrect, totalAttempts) : 0,
    overallProgress: percent(solved, EXERCISES.length),
    streak: state.streak.current,
    longestStreak: state.streak.longest,
    topicsMastered: stats.filter((stat) => stat.mastery >= 80 && stat.attempted > 0).length,
    needingReview: reviewQueue(state).length,
    interviewKnown: Object.values(state.interview).filter((entry) => entry.status === "knew").length,
    interviewTotal: INTERVIEW_QUESTIONS.length,
    projectSteps,
    projectStepsTotal: TOTAL_PROJECT_STEPS,
    projectsCompleted,
    dueCount: reviewQueue(state, "due").length,
  };
}

/** Topics with the weakest signal, worst first. Only considers attempted topics. */
export function weakAreas(state: AppState, limit = 5): TopicStats[] {
  return allTopicStats(state)
    .filter((stat) => stat.attempted > 0 && stat.mastery < 75)
    .sort((a, b) => a.mastery - b.mastery)
    .slice(0, limit);
}

/** The next topic on the learning path that is not yet finished. */
export function continueLearning(state: AppState): TopicStats {
  const stats = LEARNING_PATH.map((topic) => topicStats(state, topic));
  const started = stats.find((stat) => stat.attempted > 0 && stat.completion < 100);
  if (started) return started;
  const next = stats.find((stat) => stat.completion < 100);
  return next ?? stats[0];
}

export function learningPath(state: AppState) {
  return LEARNING_PATH.map((topic, index) => {
    const stat = topicStats(state, topic);
    return {
      ...stat,
      step: index + 1,
      status: stat.completion >= 80 ? "done" : stat.attempted > 0 ? "active" : "locked",
    } as const;
  });
}

export function exerciseProgressOf(state: AppState, id: string) {
  return state.exercises[id];
}

export function isBookmarked(state: AppState, key: string): boolean {
  return state.bookmarks.includes(key);
}
