import type { MiniProject } from "@/lib/types";
import { linearRegression } from "./linear-regression";
import { binaryClassifier } from "./binary-classifier";
import { mnistClassifier } from "./mnist";
import { cnnClassifier } from "./cnn";
import { textClassifier } from "./text-classifier";
import { transformerBlock } from "./transformer-block";
import { fullPipeline } from "./full-pipeline";

/** Ordered from easiest to hardest — the projects page follows this order. */
export const PROJECTS: MiniProject[] = [
  linearRegression,
  binaryClassifier,
  mnistClassifier,
  cnnClassifier,
  textClassifier,
  transformerBlock,
  fullPipeline,
];

export function projectBySlug(slug: string): MiniProject | undefined {
  return PROJECTS.find((project) => project.slug === slug);
}

export const TOTAL_PROJECT_STEPS = PROJECTS.reduce((sum, p) => sum + p.steps.length, 0);
