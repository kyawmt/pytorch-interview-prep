"use client";

import { TOPICS } from "@/data/topics";
import type { Difficulty, ExerciseType, Importance, TopicId } from "@/lib/types";
import { cn } from "@/lib/utils";

export interface FilterState {
  topic: TopicId | "all";
  difficulty: Difficulty | "all";
  importance: Importance | "all";
  type: ExerciseType | "all";
  status: "all" | "unsolved" | "solved" | "wrong" | "bookmarked";
}

export const DEFAULT_FILTERS: FilterState = {
  topic: "all",
  difficulty: "all",
  importance: "all",
  type: "all",
  status: "all",
};

const SELECT_CLASS =
  "rounded-lg border border-border-base bg-surface px-2.5 py-1.5 text-xs text-text outline-none transition-colors hover:border-border-strong focus:border-accent";

export function ExerciseFilters({
  value,
  onChange,
  resultCount,
  /** What "Reset" returns to — a topic page keeps its topic locked. */
  baseline = DEFAULT_FILTERS,
  hideTopic = false,
}: {
  value: FilterState;
  onChange: (next: FilterState) => void;
  resultCount: number;
  baseline?: FilterState;
  hideTopic?: boolean;
}) {
  const set = <K extends keyof FilterState>(key: K, next: FilterState[K]) =>
    onChange({ ...value, [key]: next });

  const dirty = JSON.stringify(value) !== JSON.stringify(baseline);

  return (
    <div className="mb-5 flex flex-wrap items-center gap-2">
      {hideTopic ? null : (
        <>
          <label className="sr-only" htmlFor="filter-topic">
            Topic
          </label>
          <select
            id="filter-topic"
            className={SELECT_CLASS}
            value={value.topic}
            onChange={(event) => set("topic", event.target.value as FilterState["topic"])}
          >
            <option value="all">All topics</option>
            {TOPICS.map((topic) => (
              <option key={topic.id} value={topic.id}>
                {topic.title}
              </option>
            ))}
          </select>
        </>
      )}

      <label className="sr-only" htmlFor="filter-difficulty">
        Difficulty
      </label>
      <select
        id="filter-difficulty"
        className={SELECT_CLASS}
        value={value.difficulty}
        onChange={(event) => set("difficulty", event.target.value as FilterState["difficulty"])}
      >
        <option value="all">Any difficulty</option>
        <option value="easy">Easy</option>
        <option value="medium">Medium</option>
        <option value="hard">Hard</option>
      </select>

      <label className="sr-only" htmlFor="filter-type">
        Exercise type
      </label>
      <select
        id="filter-type"
        className={SELECT_CLASS}
        value={value.type}
        onChange={(event) => set("type", event.target.value as FilterState["type"])}
      >
        <option value="all">Any type</option>
        <option value="code">Write code</option>
        <option value="fill-blank">Fill the blank</option>
        <option value="multiple-choice">Multiple choice</option>
        <option value="shape">Predict the shape</option>
        <option value="debug">Spot the bug</option>
        <option value="ordering">Order the steps</option>
        <option value="memory">From memory</option>
      </select>

      <label className="sr-only" htmlFor="filter-status">
        Status
      </label>
      <select
        id="filter-status"
        className={SELECT_CLASS}
        value={value.status}
        onChange={(event) => set("status", event.target.value as FilterState["status"])}
      >
        <option value="all">All statuses</option>
        <option value="unsolved">Not solved yet</option>
        <option value="solved">Solved</option>
        <option value="wrong">Answered incorrectly</option>
        <option value="bookmarked">Bookmarked</option>
      </select>

      <button
        type="button"
        onClick={() => set("importance", value.importance === "high" ? "all" : "high")}
        aria-pressed={value.importance === "high"}
        className={cn(
          "rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors",
          value.importance === "high"
            ? "border-accent bg-accent-soft text-accent"
            : "border-border-base bg-surface text-muted hover:border-border-strong hover:text-text",
        )}
      >
        🔥 High priority only
      </button>

      {dirty ? (
        <button
          type="button"
          onClick={() => onChange(baseline)}
          className="text-xs text-muted underline underline-offset-2 hover:text-text"
        >
          Reset
        </button>
      ) : null}

      <span className="ml-auto text-xs text-faint">{resultCount} exercises</span>
    </div>
  );
}

export function applyFilters(
  exercises: import("@/lib/types").Exercise[],
  filters: FilterState,
  state: import("@/lib/types").AppState,
) {
  return exercises.filter((exercise) => {
    if (filters.topic !== "all" && exercise.topic !== filters.topic) return false;
    if (filters.difficulty !== "all" && exercise.difficulty !== filters.difficulty) return false;
    if (filters.importance !== "all" && exercise.importance !== filters.importance) return false;
    if (filters.type !== "all" && exercise.type !== filters.type) return false;

    const entry = state.exercises[exercise.id];
    switch (filters.status) {
      case "solved":
        return Boolean(entry && entry.correctAttempts > 0);
      case "unsolved":
        return !entry || entry.correctAttempts === 0;
      case "wrong":
        return Boolean(entry && entry.attempts > 0 && !entry.correct);
      case "bookmarked":
        return state.bookmarks.includes(`ex:${exercise.id}`);
      default:
        return true;
    }
  });
}
