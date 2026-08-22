/**
 * Global search across cheat-sheet entries, exercises, interview questions,
 * mistakes and projects. A small in-memory index is enough for a few hundred
 * documents and keeps the bundle free of a search dependency.
 */

import { CHEATSHEET } from "@/data/cheatsheet";
import { EXERCISES } from "@/data/exercises";
import { INTERVIEW_QUESTIONS } from "@/data/interview";
import { MISTAKES } from "@/data/mistakes";
import { PROJECTS } from "@/data/projects";
import { topicTitle } from "@/data/topics";
import type { TopicId } from "./types";

export type SearchKind = "cheatsheet" | "exercise" | "interview" | "mistake" | "project";

export interface SearchDoc {
  id: string;
  kind: SearchKind;
  title: string;
  subtitle: string;
  href: string;
  topic: TopicId;
  /** Lower-cased haystack. */
  text: string;
  /** Static bump so high-value content surfaces first on equal matches. */
  weight: number;
}

const KIND_LABEL: Record<SearchKind, string> = {
  cheatsheet: "Cheat sheet",
  exercise: "Exercise",
  interview: "Interview",
  mistake: "Mistake",
  project: "Project",
};

export function kindLabel(kind: SearchKind): string {
  return KIND_LABEL[kind];
}

function build(): SearchDoc[] {
  const docs: SearchDoc[] = [];

  for (const entry of CHEATSHEET) {
    docs.push({
      id: entry.id,
      kind: "cheatsheet",
      title: entry.title,
      subtitle: `${topicTitle(entry.topic)} · ${entry.section}`,
      href: `/cheatsheet/${entry.topic}#${entry.id}`,
      topic: entry.topic,
      text: [entry.title, entry.description, entry.syntax, entry.example, entry.interviewNote, entry.section, ...(entry.tags ?? [])]
        .join(" ")
        .toLowerCase(),
      weight: entry.importance === "high" ? 3 : entry.importance === "medium" ? 2 : 1,
    });
  }

  for (const exercise of EXERCISES) {
    docs.push({
      id: exercise.id,
      kind: "exercise",
      title: exercise.title,
      subtitle: `${topicTitle(exercise.topic)} · ${exercise.difficulty}`,
      href: `/practice/${exercise.topic}?exercise=${exercise.id}`,
      topic: exercise.topic,
      text: [exercise.title, exercise.question, exercise.solution, exercise.explanation, exercise.context, ...(exercise.tags ?? [])]
        .join(" ")
        .toLowerCase(),
      weight: exercise.importance === "high" ? 2 : 1,
    });
  }

  for (const question of INTERVIEW_QUESTIONS) {
    docs.push({
      id: question.id,
      kind: "interview",
      title: question.question,
      subtitle: question.category,
      href: `/interview?q=${question.id}`,
      topic: question.topic,
      text: [question.question, question.shortAnswer, question.detailedAnswer, question.code, ...(question.tags ?? [])]
        .join(" ")
        .toLowerCase(),
      weight: question.importance === "high" ? 3 : 2,
    });
  }

  for (const mistake of MISTAKES) {
    docs.push({
      id: mistake.id,
      kind: "mistake",
      title: mistake.title,
      subtitle: `Common mistake · ${topicTitle(mistake.topic)}`,
      href: `/mistakes#${mistake.id}`,
      topic: mistake.topic,
      text: [mistake.title, mistake.symptom, mistake.explanation, mistake.wrong, mistake.right, ...(mistake.tags ?? [])]
        .join(" ")
        .toLowerCase(),
      weight: mistake.importance === "high" ? 2 : 1,
    });
  }

  for (const project of PROJECTS) {
    docs.push({
      id: project.slug,
      kind: "project",
      title: project.title,
      subtitle: `Mini project · ${project.steps.length} steps`,
      href: `/projects/${project.slug}`,
      topic: project.topics[0],
      text: [project.title, project.tagline, project.overview, ...project.outline].join(" ").toLowerCase(),
      weight: 1,
    });
  }

  return docs;
}

let index: SearchDoc[] | null = null;

function getIndex(): SearchDoc[] {
  if (!index) index = build();
  return index;
}

export interface SearchResult extends SearchDoc {
  score: number;
}

/** Token-AND matching with title/tag boosts. Good enough for ~350 docs. */
export function search(query: string, limit = 20): SearchResult[] {
  const trimmed = query.trim().toLowerCase();
  if (trimmed.length < 2) return [];
  const tokens = trimmed.split(/\s+/).filter(Boolean);

  const results: SearchResult[] = [];
  for (const doc of getIndex()) {
    const title = doc.title.toLowerCase();
    let score = 0;
    let matchedAll = true;

    for (const token of tokens) {
      const inTitle = title.includes(token);
      const inText = doc.text.includes(token);
      if (!inTitle && !inText) {
        matchedAll = false;
        break;
      }
      if (inTitle) score += title.startsWith(token) ? 8 : 5;
      if (inText) score += 1;
    }
    if (!matchedAll) continue;
    if (title === trimmed) score += 20;
    score += doc.weight;
    results.push({ ...doc, score });
  }

  return results.sort((a, b) => b.score - a.score).slice(0, limit);
}
