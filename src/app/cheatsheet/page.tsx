import Link from "next/link";
import { Badge } from "@/components/common/Badge";
import { PageHeader } from "@/components/common/PageHeader";
import { CHEATSHEET, cheatSheetByTopic } from "@/data/cheatsheet";
import { TOPICS } from "@/data/topics";

export const metadata = {
  title: "Cheat Sheet",
  description: "Searchable PyTorch syntax reference with interview notes.",
};

export default function CheatSheetIndexPage() {
  const topics = TOPICS.map((topic) => ({
    ...topic,
    count: cheatSheetByTopic(topic.id).length,
  })).filter((topic) => topic.count > 0);

  return (
    <div>
      <PageHeader
        eyebrow="Cheat sheet"
        title="PyTorch syntax, organised for recall"
        description={`${CHEATSHEET.length} cards covering the syntax an AI engineer is expected to write without looking it up. Every card has a short explanation, a runnable example and — where it matters — the detail an interviewer will probe. Turn on recall mode to hide the examples.`}
      />

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {topics.map((topic) => (
          <Link
            key={topic.id}
            href={`/cheatsheet/${topic.id}`}
            className="card group flex flex-col p-4 transition-colors hover:border-border-strong hover:bg-surface-hover"
          >
            <div className="flex items-center gap-2">
              <span aria-hidden className="text-accent">
                {topic.icon}
              </span>
              <h2 className="text-sm font-semibold text-text">{topic.title}</h2>
              {topic.importance === "high" ? <Badge tone="accent">🔥</Badge> : null}
              <span className="ml-auto font-mono text-xs text-faint">{topic.count}</span>
            </div>
            <p className="mt-2 flex-1 text-xs leading-relaxed text-muted">{topic.summary}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
