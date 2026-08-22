/**
 * Session builders: daily practice and the timed interview challenge.
 *
 * Both are deterministic given their seed so a reload does not reshuffle the
 * questions mid-session.
 */

import { EXERCISES } from "@/data/exercises";
import { INTERVIEW_QUESTIONS } from "@/data/interview";
import { isDue } from "./storage";
import type { AppState, Exercise, InterviewQuestion } from "./types";
import { hashString, seededRandom, shuffle } from "./utils";

export interface DailySession {
  syntax: Exercise[];
  concepts: Exercise[];
  interview: InterviewQuestion[];
  dateKey: string;
}

const SYNTAX_TYPES = new Set(["code", "fill-blank", "memory"]);

/**
 * ~5 syntax + 3 concept exercises + 2 interview questions.
 * Prefers due/unseen items but stays stable for a given day.
 */
export function buildDailySession(state: AppState, dateKey: string): DailySession {
  const random = seededRandom(hashString(dateKey) || 1);

  const score = (exercise: Exercise) => {
    const entry = state.exercises[exercise.id];
    if (!entry || entry.attempts === 0) return 2; // unseen
    if (!entry.correct) return 3; // got it wrong
    if (isDue(entry)) return 2.5; // due for review
    if (entry.reviewState === "mastered") return 0.2;
    return 1;
  };

  const pick = (pool: Exercise[], count: number) =>
    shuffle(pool, random)
      .map((exercise) => ({ exercise, weight: score(exercise) * (0.6 + random() * 0.8) }))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, count)
      .map((item) => item.exercise);

  const syntaxPool = EXERCISES.filter((e) => SYNTAX_TYPES.has(e.type));
  const conceptPool = EXERCISES.filter((e) => !SYNTAX_TYPES.has(e.type));

  const interview = shuffle(INTERVIEW_QUESTIONS, random)
    .sort((a, b) => {
      const aSeen = state.interview[a.id]?.status === "knew" ? 1 : 0;
      const bSeen = state.interview[b.id]?.status === "knew" ? 1 : 0;
      return aSeen - bSeen;
    })
    .slice(0, 2);

  return {
    syntax: pick(syntaxPool, 5),
    concepts: pick(conceptPool, 3),
    interview,
    dateKey,
  };
}

export interface ChallengeConfig {
  questionCount: number;
  minutes: number;
}

export const DEFAULT_CHALLENGE: ChallengeConfig = { questionCount: 20, minutes: 30 };

/**
 * A balanced 20-question mix: syntax, shapes, debugging, concepts and
 * code completion, weighted toward high-importance items.
 */
export function buildChallenge(seed: number, config = DEFAULT_CHALLENGE): Exercise[] {
  const random = seededRandom(seed || 1);
  const quotas: Array<{ match: (e: Exercise) => boolean; share: number }> = [
    { match: (e) => e.type === "shape", share: 0.2 },
    { match: (e) => e.type === "debug" || e.topic === "debugging", share: 0.2 },
    { match: (e) => e.type === "multiple-choice", share: 0.25 },
    { match: (e) => e.type === "code" || e.type === "memory", share: 0.25 },
    { match: (e) => e.type === "ordering" || e.type === "fill-blank", share: 0.1 },
  ];

  const chosen: Exercise[] = [];
  const used = new Set<string>();

  for (const quota of quotas) {
    const target = Math.max(1, Math.round(config.questionCount * quota.share));
    const pool = shuffle(
      EXERCISES.filter((e) => quota.match(e) && !used.has(e.id)),
      random,
    ).sort((a, b) => importanceRank(b) - importanceRank(a));
    for (const exercise of pool.slice(0, target)) {
      chosen.push(exercise);
      used.add(exercise.id);
    }
  }

  // top up if rounding left us short
  const filler = shuffle(
    EXERCISES.filter((e) => !used.has(e.id) && e.importance === "high"),
    random,
  );
  while (chosen.length < config.questionCount && filler.length > 0) {
    const next = filler.pop();
    if (next) {
      chosen.push(next);
      used.add(next.id);
    }
  }

  return shuffle(chosen, random).slice(0, config.questionCount);
}

function importanceRank(exercise: Exercise): number {
  return exercise.importance === "high" ? 2 : exercise.importance === "medium" ? 1 : 0;
}
