/**
 * Content linter.
 *
 * Run with `npm run check:content` after adding or editing exercises, projects
 * or cheat-sheet entries. It catches the mistakes that are easy to make when
 * authoring content and impossible to see until a learner hits them:
 *
 *   - an exercise whose own reference solution its validator would reject
 *   - an accepted answer that does not actually validate
 *   - required patterns that do not match the reference solution
 *   - duplicate ids, dangling cross-references, missing explanations
 */

import { CHEATSHEET } from "@/data/cheatsheet";
import { EXERCISES, EXERCISE_MAP } from "@/data/exercises";
import { INTERVIEW_QUESTIONS } from "@/data/interview";
import { MISTAKES } from "@/data/mistakes";
import { PROJECTS } from "@/data/projects";
import { REVISION_CARDS } from "@/data/revision";
import { MEMORY_DRILLS } from "@/data/memory-drills";
import { BUILDER_DRILLS } from "@/data/builder-drills";
import { SHAPE_DRILL_EXERCISES } from "@/data/shape-drills";
import { TOPIC_MAP } from "@/data/topics";
import type { Exercise, TopicId } from "@/lib/types";
import { normalizeCode, validateAnswer, validateProjectStep } from "@/lib/validation";

const problems: string[] = [];
const fail = (message: string) => problems.push(message);

function checkUniqueIds(label: string, ids: string[]) {
  const seen = new Set<string>();
  for (const id of ids) {
    if (seen.has(id)) fail(`${label}: duplicate id "${id}"`);
    seen.add(id);
  }
}

function checkTopic(label: string, topic: TopicId) {
  if (!TOPIC_MAP[topic]) fail(`${label}: unknown topic "${topic}"`);
}

/* ------------------------------------------------------------------ */
/* Exercises (main bank + drills)                                      */
/* ------------------------------------------------------------------ */

const allExercises: Exercise[] = [
  ...EXERCISES,
  ...MEMORY_DRILLS,
  ...BUILDER_DRILLS,
  ...SHAPE_DRILL_EXERCISES,
];

checkUniqueIds("exercises", allExercises.map((exercise) => exercise.id));

// Titles show up in review lists and challenge results, so duplicates are
// confusing even though they are not strictly invalid.
{
  const seen = new Map<string, string>();
  for (const exercise of allExercises) {
    const previous = seen.get(exercise.title);
    if (previous) fail(`exercises: duplicate title "${exercise.title}" (${previous} and ${exercise.id})`);
    else seen.set(exercise.title, exercise.id);
  }
}

for (const exercise of allExercises) {
  const label = `exercise ${exercise.id}`;
  checkTopic(label, exercise.topic);
  if (!exercise.explanation) fail(`${label}: missing explanation`);
  if (!exercise.solution) fail(`${label}: missing solution`);
  if (!exercise.question) fail(`${label}: missing question`);

  switch (exercise.type) {
    case "multiple-choice": {
      if (!exercise.options?.length) fail(`${label}: multiple-choice without options`);
      if (
        exercise.correctOption === undefined ||
        exercise.correctOption < 0 ||
        exercise.correctOption >= (exercise.options?.length ?? 0)
      ) {
        fail(`${label}: correctOption out of range`);
        break;
      }
      if (!validateAnswer(exercise.correctOption, exercise).correct) {
        fail(`${label}: correctOption is rejected by the validator`);
      }
      break;
    }
    case "ordering": {
      if (!exercise.blocks?.length || !exercise.correctOrder?.length) {
        fail(`${label}: ordering without blocks/correctOrder`);
        break;
      }
      if (exercise.correctOrder.some((i) => i < 0 || i >= exercise.blocks!.length)) {
        fail(`${label}: correctOrder references a missing block`);
        break;
      }
      const answer = exercise.correctOrder.map((i) => exercise.blocks![i]);
      if (!validateAnswer(answer, exercise).correct) {
        fail(`${label}: the correct order is rejected by the validator`);
      }
      break;
    }
    case "fill-blank": {
      // `solution` is display text; acceptedAnswers are authoritative.
      if (!exercise.acceptedAnswers?.length) fail(`${label}: fill-blank without acceptedAnswers`);
      break;
    }
    default: {
      const result = validateAnswer(exercise.solution, exercise);
      if (!result.correct) {
        fail(`${label}: its own solution is rejected — ${result.feedback} ${JSON.stringify(result.missing ?? [])}`);
      }
      // Required patterns double as feedback, so they must hold for the solution.
      for (const pattern of exercise.requiredPatterns ?? []) {
        if (!new RegExp(pattern, "i").test(normalizeCode(exercise.solution))) {
          fail(`${label}: requiredPattern /${pattern}/ does not match its own solution`);
        }
      }
    }
  }

  for (const accepted of exercise.acceptedAnswers ?? []) {
    if (!validateAnswer(accepted, exercise).correct) {
      fail(`${label}: acceptedAnswer ${JSON.stringify(accepted)} is rejected`);
    }
  }
}

/* ------------------------------------------------------------------ */
/* Projects                                                            */
/* ------------------------------------------------------------------ */

checkUniqueIds("projects", PROJECTS.map((project) => project.slug));

for (const project of PROJECTS) {
  checkUniqueIds(`project ${project.slug}`, project.steps.map((step) => step.id));
  if (project.steps.length === 0) fail(`project ${project.slug}: no steps`);
  for (const topic of project.topics) checkTopic(`project ${project.slug}`, topic);

  for (const step of project.steps) {
    const label = `project ${project.slug}/${step.id}`;
    if (!step.explanation) fail(`${label}: missing explanation`);
    if (!step.solution) fail(`${label}: missing solution`);

    if (step.kind === "multiple-choice") {
      if (
        step.correctOption === undefined ||
        step.correctOption < 0 ||
        step.correctOption >= (step.options?.length ?? 0)
      ) {
        fail(`${label}: correctOption out of range`);
        continue;
      }
      if (!validateProjectStep(step.correctOption, step).correct) {
        fail(`${label}: correctOption is rejected by the validator`);
      }
      continue;
    }

    const result = validateProjectStep(step.solution, step);
    if (!result.correct) {
      fail(`${label}: its own solution is rejected — ${result.feedback} ${JSON.stringify(result.missing ?? [])}`);
    }
    for (const accepted of step.acceptedAnswers ?? []) {
      if (!validateProjectStep(accepted, step).correct) {
        fail(`${label}: acceptedAnswer ${JSON.stringify(accepted)} is rejected`);
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* Cheat sheet, interview questions, mistakes, revision                */
/* ------------------------------------------------------------------ */

checkUniqueIds("cheatsheet", CHEATSHEET.map((entry) => entry.id));
for (const entry of CHEATSHEET) {
  const label = `cheatsheet ${entry.id}`;
  checkTopic(label, entry.topic);
  if (!entry.example) fail(`${label}: missing example`);
  if (!entry.syntax) fail(`${label}: missing syntax`);
  for (const id of entry.relatedExercises ?? []) {
    if (!EXERCISE_MAP.has(id)) fail(`${label}: relatedExercises points at unknown exercise "${id}"`);
  }
}

checkUniqueIds("interview", INTERVIEW_QUESTIONS.map((question) => question.id));
for (const question of INTERVIEW_QUESTIONS) {
  const label = `interview ${question.id}`;
  checkTopic(label, question.topic);
  if (!question.shortAnswer) fail(`${label}: missing shortAnswer`);
  if (!question.detailedAnswer) fail(`${label}: missing detailedAnswer`);
}

checkUniqueIds("mistakes", MISTAKES.map((mistake) => mistake.id));
for (const mistake of MISTAKES) {
  checkTopic(`mistake ${mistake.id}`, mistake.topic);
  if (mistake.wrong === mistake.right) fail(`mistake ${mistake.id}: wrong and right are identical`);
}

checkUniqueIds("revision", REVISION_CARDS.map((card) => card.id));
for (const card of REVISION_CARDS) checkTopic(`revision ${card.id}`, card.topic);

/* ------------------------------------------------------------------ */

const counts = [
  `${CHEATSHEET.length} cheat-sheet entries`,
  `${EXERCISES.length} exercises`,
  `${MEMORY_DRILLS.length + BUILDER_DRILLS.length + SHAPE_DRILL_EXERCISES.length} drills`,
  `${INTERVIEW_QUESTIONS.length} interview questions`,
  `${PROJECTS.length} projects (${PROJECTS.reduce((n, p) => n + p.steps.length, 0)} steps)`,
  `${MISTAKES.length} common mistakes`,
  `${REVISION_CARDS.length} revision cards`,
].join("\n  ");

console.log(`Content check\n  ${counts}\n`);

if (problems.length === 0) {
  console.log("✓ No problems found.");
} else {
  console.error(`✗ ${problems.length} problem${problems.length === 1 ? "" : "s"}:`);
  for (const problem of problems) console.error(`  - ${problem}`);
  process.exit(1);
}
