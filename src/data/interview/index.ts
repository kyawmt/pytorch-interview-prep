import type { InterviewQuestion } from "@/lib/types";
import { fundamentalQuestions } from "./fundamentals";
import { trainingQuestions } from "./training";
import { advancedQuestions } from "./advanced";

export const INTERVIEW_QUESTIONS: InterviewQuestion[] = [
  ...fundamentalQuestions,
  ...trainingQuestions,
  ...advancedQuestions,
];

export const INTERVIEW_CATEGORIES = [
  ...new Set(INTERVIEW_QUESTIONS.map((question) => question.category)),
].sort();

export function interviewQuestionById(id: string): InterviewQuestion | undefined {
  return INTERVIEW_QUESTIONS.find((question) => question.id === id);
}
