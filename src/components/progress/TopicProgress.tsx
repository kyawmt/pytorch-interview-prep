import Link from "next/link";
import { TOPIC_MAP } from "@/data/topics";
import type { TopicStats } from "@/lib/progress";
import { masteryBar } from "@/lib/utils";

export function TopicProgress({ stats, href }: { stats: TopicStats; href?: string }) {
  const target = href ?? `/practice/${stats.topic}`;
  const icon = TOPIC_MAP[stats.topic]?.icon ?? "▦";
  const bar = masteryBar(stats.mastery);

  return (
    <Link
      href={target}
      className="flex items-center gap-3 rounded-lg px-2 py-2 transition-colors hover:bg-surface-hover"
    >
      <span aria-hidden className="w-4 text-center text-xs text-faint">
        {icon}
      </span>
      <span className="w-36 shrink-0 truncate text-sm text-text sm:w-44">{stats.title}</span>
      <span
        aria-hidden
        className="hidden font-mono text-xs tracking-tighter sm:inline"
        title={`${stats.mastery}% mastery`}
      >
        <span className="text-accent">{bar.filled}</span>
        <span className="text-[var(--border-strong)]">{bar.empty}</span>
      </span>
      <span className="ml-auto flex items-center gap-3">
        <span className="hidden text-[11px] text-faint md:inline">
          {stats.correct}/{stats.total}
        </span>
        <span className="w-9 text-right font-mono text-xs tabular-nums text-muted">
          {stats.mastery}%
        </span>
      </span>
    </Link>
  );
}
