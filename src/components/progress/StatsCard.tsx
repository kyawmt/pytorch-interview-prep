import Link from "next/link";
import { cn } from "@/lib/utils";

export function StatsCard({
  label,
  value,
  hint,
  href,
  tone = "default",
}: {
  label: string;
  value: string | number;
  hint?: string;
  href?: string;
  tone?: "default" | "accent" | "danger" | "success";
}) {
  const valueTone =
    tone === "accent"
      ? "text-accent"
      : tone === "danger"
        ? "text-danger"
        : tone === "success"
          ? "text-success"
          : "text-text";

  const body = (
    <>
      <p className="text-[11px] font-medium uppercase tracking-[0.07em] text-faint">{label}</p>
      <p className={cn("mt-1.5 font-mono text-2xl font-semibold tabular-nums", valueTone)}>{value}</p>
      {hint ? <p className="mt-0.5 text-[11px] text-muted">{hint}</p> : null}
    </>
  );

  const className = cn(
    "card block p-3.5 transition-colors",
    href ? "hover:border-border-strong hover:bg-surface-hover" : "",
  );

  return href ? (
    <Link href={href} className={className}>
      {body}
    </Link>
  ) : (
    <div className={className}>{body}</div>
  );
}
