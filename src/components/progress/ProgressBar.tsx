import { cn } from "@/lib/utils";

export function ProgressBar({
  value,
  label,
  showValue = true,
  tone = "accent",
  size = "md",
  className,
}: {
  value: number;
  label?: string;
  showValue?: boolean;
  tone?: "accent" | "success" | "muted";
  size?: "sm" | "md";
  className?: string;
}) {
  const clamped = Math.max(0, Math.min(100, Math.round(value)));
  const fill =
    tone === "success" ? "bg-success" : tone === "muted" ? "bg-[var(--border-strong)]" : "bg-accent";

  return (
    <div className={className}>
      {label || showValue ? (
        <div className="mb-1 flex items-baseline justify-between gap-3 text-xs">
          {label ? <span className="truncate text-muted">{label}</span> : <span />}
          {showValue ? <span className="font-mono tabular-nums text-text">{clamped}%</span> : null}
        </div>
      ) : null}
      <div
        className={cn("overflow-hidden rounded-full bg-surface-2", size === "sm" ? "h-1" : "h-1.5")}
        role="progressbar"
        aria-valuenow={clamped}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label ? `${label} progress` : "Progress"}
      >
        <div
          className={cn("h-full origin-left rounded-full transition-[width] duration-500", fill)}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}
