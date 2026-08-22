import type { CheatSheetEntry, TopicId } from "@/lib/types";
import { tensorEntries } from "./tensors";
import { shapeEntries } from "./shapes";
import { mathEntries } from "./math";
import { autogradEntries } from "./autograd";
import { nnEntries } from "./neural-networks";
import { trainingEntries } from "./training";
import { dataEntries } from "./data";
import { gpuEntries } from "./gpu";

export const CHEATSHEET: CheatSheetEntry[] = [
  ...tensorEntries,
  ...shapeEntries,
  ...mathEntries,
  ...autogradEntries,
  ...nnEntries,
  ...trainingEntries,
  ...dataEntries,
  ...gpuEntries,
];

export function cheatSheetByTopic(topic: TopicId): CheatSheetEntry[] {
  return CHEATSHEET.filter((entry) => entry.topic === topic);
}

export function cheatSheetEntry(id: string): CheatSheetEntry | undefined {
  return CHEATSHEET.find((entry) => entry.id === id);
}

/** Group a list of entries by their `section`, preserving declaration order. */
export function groupBySection(entries: CheatSheetEntry[]) {
  const groups = new Map<string, CheatSheetEntry[]>();
  for (const entry of entries) {
    const list = groups.get(entry.section);
    if (list) list.push(entry);
    else groups.set(entry.section, [entry]);
  }
  return [...groups.entries()].map(([section, items]) => ({ section, items }));
}
