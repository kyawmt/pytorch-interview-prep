import Link from "next/link";
import { notFound } from "next/navigation";
import { DifficultyBadge } from "@/components/common/Badge";
import { PageHeader } from "@/components/common/PageHeader";
import { ProjectRunner } from "@/components/projects/ProjectRunner";
import { PROJECTS, projectBySlug } from "@/data/projects";
import { topicTitle } from "@/data/topics";

export function generateStaticParams() {
  return PROJECTS.map((project) => ({ project: project.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ project: string }> }) {
  const { project } = await params;
  const found = projectBySlug(project);
  return found ? { title: found.title, description: found.tagline } : {};
}

export default async function ProjectPage({ params }: { params: Promise<{ project: string }> }) {
  const { project: slug } = await params;
  const project = projectBySlug(slug);
  if (!project) notFound();

  return (
    <div>
      <nav aria-label="Breadcrumb" className="mb-3 text-xs text-muted">
        <Link href="/projects" className="hover:text-accent">
          Mini projects
        </Link>
        <span className="mx-1.5 text-faint">/</span>
        <span className="text-text">{project.title}</span>
      </nav>

      <PageHeader
        title={project.title}
        description={project.overview}
        actions={
          <span className="flex items-center gap-2">
            <DifficultyBadge difficulty={project.difficulty} />
            <span className="text-xs text-muted">~{project.estimatedMinutes} min</span>
          </span>
        }
      />

      <div className="mb-5 flex flex-wrap items-center gap-1.5 text-[11px] text-faint">
        Covers: {project.topics.map(topicTitle).join(" · ")}
      </div>

      <ProjectRunner project={project} />
    </div>
  );
}
