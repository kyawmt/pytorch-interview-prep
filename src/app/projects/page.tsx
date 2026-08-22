"use client";

import Link from "next/link";
import { Badge, DifficultyBadge } from "@/components/common/Badge";
import { PageHeader } from "@/components/common/PageHeader";
import { ProgressBar } from "@/components/progress/ProgressBar";
import { PROJECTS } from "@/data/projects";
import { topicTitle } from "@/data/topics";
import { useAppState } from "@/hooks/useAppState";

export default function ProjectsPage() {
  const state = useAppState();

  return (
    <div>
      <PageHeader
        eyebrow="Mini projects"
        title="Build the whole thing, one line at a time"
        description="Nothing runs here — you fill in the lines that matter. Each project walks from data to a trained, saved model, and every step checks the exact syntax an interviewer would expect you to produce."
      />

      <div className="space-y-3">
        {PROJECTS.map((project, index) => {
          const progress = state.projects[project.slug];
          const completed = project.steps.filter((step) => progress?.steps[step.id]).length;
          const percentage = (completed / project.steps.length) * 100;

          return (
            <Link
              key={project.slug}
              href={`/projects/${project.slug}`}
              className="card block p-4 transition-colors hover:border-border-strong hover:bg-surface-hover"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-xs text-accent">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <h2 className="text-sm font-semibold text-text">{project.title}</h2>
                <DifficultyBadge difficulty={project.difficulty} />
                <span className="text-[11px] text-faint">~{project.estimatedMinutes} min</span>
                {completed === project.steps.length ? (
                  <Badge tone="success">Complete</Badge>
                ) : completed > 0 ? (
                  <Badge tone="warning">
                    {completed}/{project.steps.length}
                  </Badge>
                ) : null}
              </div>

              <p className="mt-1.5 text-sm text-muted">{project.tagline}</p>

              <div className="mt-3 flex flex-wrap items-center gap-1.5">
                {project.outline.map((stage, stageIndex) => (
                  <span key={stage} className="flex items-center gap-1.5">
                    <span className="rounded-md border border-border-base bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-muted">
                      {stage}
                    </span>
                    {stageIndex < project.outline.length - 1 ? (
                      <span aria-hidden className="text-faint">
                        →
                      </span>
                    ) : null}
                  </span>
                ))}
              </div>

              <div className="mt-3 flex items-center gap-3">
                <ProgressBar value={percentage} showValue={false} size="sm" className="flex-1" />
                <span className="shrink-0 text-[11px] text-faint">
                  {project.topics.slice(0, 3).map(topicTitle).join(" · ")}
                </span>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
