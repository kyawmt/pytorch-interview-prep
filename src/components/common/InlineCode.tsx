import { Fragment } from "react";
import { cn } from "@/lib/utils";

/**
 * Renders `backtick`-delimited spans as inline code. Content is authored as
 * plain strings, so this is the only "markdown" the app needs — a full parser
 * would be a dependency for one feature.
 */
export function InlineText({ children, className }: { children: string; className?: string }) {
  const parts = children.split("`");
  return (
    <span className={className}>
      {parts.map((part, index) =>
        index % 2 === 1 ? (
          <code
            key={index}
            className="rounded border border-border-base bg-surface-2 px-1 py-px font-mono text-[0.9em] text-accent"
          >
            {part}
          </code>
        ) : (
          <Fragment key={index}>{part}</Fragment>
        ),
      )}
    </span>
  );
}

/** Paragraph variant: splits on blank lines and renders inline code inside. */
export function InlineParagraphs({ children, className }: { children: string; className?: string }) {
  const paragraphs = children.split(/\n{2,}/);
  return (
    <div className={cn("prose-tight", className)}>
      {paragraphs.map((paragraph, index) => (
        <p key={index} className="whitespace-pre-line">
          <InlineText>{paragraph}</InlineText>
        </p>
      ))}
    </div>
  );
}
