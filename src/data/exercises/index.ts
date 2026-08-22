import type { Exercise, TopicId } from "@/lib/types";
import { tensorExercises } from "./tensors";
import { shapeExercises } from "./shapes";
import { autogradExercises } from "./autograd";
import { nnExercises } from "./nn";
import { trainingExercises } from "./training";
import { dataExercises } from "./data";
import { gpuExercises } from "./gpu";
import { challengeExercises } from "./challenge";

export const EXERCISES: Exercise[] = [
  ...tensorExercises,
  ...shapeExercises,
  ...autogradExercises,
  ...nnExercises,
  ...trainingExercises,
  ...dataExercises,
  ...gpuExercises,
  ...challengeExercises,
];

export const EXERCISE_MAP = new Map(EXERCISES.map((exercise) => [exercise.id, exercise]));

export function exerciseById(id: string): Exercise | undefined {
  return EXERCISE_MAP.get(id);
}

export function exercisesByTopic(topic: TopicId): Exercise[] {
  return EXERCISES.filter((exercise) => exercise.topic === topic);
}

export function exercisesByLevel(level: number): Exercise[] {
  return EXERCISES.filter((exercise) => exercise.level === level);
}
