import type { Difficulty, Importance } from "@/lib/types";
import { cn } from "@/lib/utils";

type Tone = "neutral" | "accent" | "success" | "danger" | "warning" | "info";

const TONE_CLASS: Record<Tone, string> = {
  neutral: "bg-surface-2 text-muted border-border-base",
  accent: "bg-accent-soft text-accent border-accent/25",
  success: "bg-success-soft text-success border-success/25",
  danger: "bg-danger-soft text-danger border-danger/25",
  warning: "bg-warning-soft text-warning border-warning/25",
  info: "bg-info-soft text-info border-info/25",
};

export function Badge({
  children,
  tone = "neutral",
  className,
  title,
}: {
  children: React.ReactNode;
  tone?: Tone;
  className?: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium leading-none",
        TONE_CLASS[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

const DIFFICULTY_TONE: Record<Difficulty, Tone> = {
  easy: "success",
  medium: "warning",
  hard: "danger",
};

export function DifficultyBadge({ difficulty }: { difficulty: Difficulty }) {
  return (
    <Badge tone={DIFFICULTY_TONE[difficulty]} className="capitalize">
      {difficulty}
    </Badge>
  );
}

export function ImportanceBadge({ importance }: { importance: Importance }) {
  if (importance === "high") {
    return (
      <Badge tone="accent" title="Frequently asked in AI engineer interviews">
        🔥 Interview essential
      </Badge>
    );
  }
  if (importance === "medium") {
    return <Badge tone="neutral">Medium priority</Badge>;
  }
  return <Badge tone="neutral">Low priority</Badge>;
}

export function TypeBadge({ type }: { type: string }) {
  const label: Record<string, string> = {
    code: "Write code",
    "fill-blank": "Fill the blank",
    "multiple-choice": "Multiple choice",
    shape: "Predict the shape",
    debug: "Spot the bug",
    ordering: "Order the steps",
    memory: "From memory",
  };
  return <Badge tone="info">{label[type] ?? type}</Badge>;
}
