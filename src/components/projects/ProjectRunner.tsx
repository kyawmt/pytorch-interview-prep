"use client";

import { useState } from "react";
import { Button, LinkButton } from "@/components/common/Button";
import { ProgressBar } from "@/components/progress/ProgressBar";
import { useAppState } from "@/hooks/useAppState";
import { resetProject } from "@/lib/storage";
import type { MiniProject } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ProjectStepCard } from "./ProjectStepCard";

export function ProjectRunner({ project }: { project: MiniProject }) {
  const state = useAppState();
  const progress = state.projects[project.slug];
  const completed = project.steps.filter((step) => progress?.steps[step.id]).length;

  const firstIncomplete = project.steps.findIndex((step) => !progress?.steps[step.id]);
  const [index, setIndex] = useState(() => (firstIncomplete === -1 ? 0 : firstIncomplete));
  const step = project.steps[index];
  const done = Boolean(progress?.steps[step.id]);

  return (
    <div>
      <div className="card mb-5 p-4">
        <ProgressBar
          value={(completed / project.steps.length) * 100}
          label={`${completed} of ${project.steps.length} steps complete`}
        />
        <ol className="mt-3.5 flex flex-wrap gap-1.5">
          {project.steps.map((item, itemIndex) => {
            const itemDone = Boolean(progress?.steps[item.id]);
            return (
              <li key={item.id}>
                <button
                  type="button"
                  onClick={() => setIndex(itemIndex)}
                  aria-current={itemIndex === index ? "step" : undefined}
                  title={item.title}
                  className={cn(
                    "flex h-7 w-7 items-center justify-center rounded-md border text-[11px] font-medium transition-colors",
                    itemIndex === index
                      ? "border-accent bg-accent-soft text-accent"
                      : itemDone
                        ? "border-success/40 bg-success-soft text-success"
                        : "border-border-base bg-surface text-muted hover:border-border-strong",
                  )}
                >
                  {itemDone ? "✓" : itemIndex + 1}
                </button>
              </li>
            );
          })}
        </ol>
        {completed > 0 ? (
          <button
            type="button"
            onClick={() => {
              resetProject(project.slug);
              setIndex(0);
            }}
            className="mt-3 text-xs text-muted underline underline-offset-2 hover:text-danger"
          >
            Reset progress
          </button>
        ) : null}
      </div>

      <ProjectStepCard
        key={step.id}
        step={step}
        slug={project.slug}
        index={index}
        total={project.steps.length}
        done={done}
        onDone={() => undefined}
      />

      <div className="mt-4 flex items-center justify-between">
        <Button onClick={() => setIndex((i) => Math.max(0, i - 1))} disabled={index === 0}>
          ← Previous step
        </Button>
        {index < project.steps.length - 1 ? (
          <Button variant="primary" onClick={() => setIndex((i) => i + 1)}>
            Next step →
          </Button>
        ) : (
          <LinkButton href="/projects" variant="primary">
            Back to projects
          </LinkButton>
        )}
      </div>

      {completed === project.steps.length ? (
        <div className="card mt-5 border-success/35 bg-success-soft px-6 py-8 text-center">
          <p className="text-2xl" aria-hidden>
            ✓
          </p>
          <h2 className="mt-1.5 text-lg font-semibold text-text">Project complete</h2>
          <p className="mt-1 text-sm text-muted">
            You built {project.title} end to end. Try writing the whole thing from a blank editor next.
          </p>
          <div className="mt-4 flex flex-wrap justify-center gap-2">
            <LinkButton href="/memory" variant="primary">
              Write from memory
            </LinkButton>
            <LinkButton href="/projects">Next project</LinkButton>
          </div>
        </div>
      ) : null}
    </div>
  );
}
