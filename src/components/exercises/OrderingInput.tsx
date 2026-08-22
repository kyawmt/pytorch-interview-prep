"use client";

import { HighlightedCode } from "@/components/common/CodeBlock";
import { cn } from "@/lib/utils";

interface OrderingInputProps {
  blocks: string[];
  /** Indices into `blocks`, in the order the learner has chosen. */
  order: number[];
  onChange: (order: number[]) => void;
  disabled?: boolean;
}

/**
 * Click-to-build ordering. Deliberately not drag-and-drop: clicking works
 * identically with a keyboard, a mouse and a touch screen.
 */
export function OrderingInput({ blocks, order, onChange, disabled }: OrderingInputProps) {
  const available = blocks.map((_, index) => index).filter((index) => !order.includes(index));

  const add = (index: number) => onChange([...order, index]);
  const remove = (position: number) => onChange(order.filter((_, i) => i !== position));
  const move = (position: number, delta: number) => {
    const target = position + delta;
    if (target < 0 || target >= order.length) return;
    const next = [...order];
    [next[position], next[target]] = [next[target], next[position]];
    onChange(next);
  };

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <div>
        <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-faint">
          Available blocks
        </p>
        <ul className="space-y-1.5">
          {available.length === 0 ? (
            <li className="rounded-lg border border-dashed border-border-base px-3 py-3 text-center text-xs text-faint">
              All blocks used
            </li>
          ) : (
            available.map((index) => (
              <li key={index}>
                <button
                  type="button"
                  disabled={disabled}
                  onClick={() => add(index)}
                  className="w-full overflow-x-auto rounded-lg border border-border-base bg-surface-2 px-3 py-2 text-left font-mono text-xs transition-colors hover:border-accent disabled:cursor-not-allowed"
                >
                  <HighlightedCode code={blocks[index]} />
                </button>
              </li>
            ))
          )}
        </ul>
      </div>

      <div>
        <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-faint">
          Your order
        </p>
        <ol className="space-y-1.5">
          {order.length === 0 ? (
            <li className="rounded-lg border border-dashed border-border-base px-3 py-3 text-center text-xs text-faint">
              Click blocks on the left to add them
            </li>
          ) : (
            order.map((blockIndex, position) => (
              <li
                key={`${blockIndex}-${position}`}
                className="flex items-center gap-1.5 rounded-lg border border-border-base bg-surface px-2 py-1.5"
              >
                <span className="w-4 shrink-0 text-center font-mono text-[11px] text-faint">
                  {position + 1}
                </span>
                <span className="min-w-0 flex-1 overflow-x-auto font-mono text-xs">
                  <HighlightedCode code={blocks[blockIndex]} />
                </span>
                <span className="flex shrink-0 items-center">
                  <button
                    type="button"
                    disabled={disabled || position === 0}
                    onClick={() => move(position, -1)}
                    aria-label={`Move step ${position + 1} up`}
                    className={cn(
                      "rounded px-1 text-xs text-faint hover:text-text",
                      (disabled || position === 0) && "invisible",
                    )}
                  >
                    ↑
                  </button>
                  <button
                    type="button"
                    disabled={disabled || position === order.length - 1}
                    onClick={() => move(position, 1)}
                    aria-label={`Move step ${position + 1} down`}
                    className={cn(
                      "rounded px-1 text-xs text-faint hover:text-text",
                      (disabled || position === order.length - 1) && "invisible",
                    )}
                  >
                    ↓
                  </button>
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() => remove(position)}
                    aria-label={`Remove step ${position + 1}`}
                    className="rounded px-1 text-xs text-faint hover:text-danger"
                  >
                    ✕
                  </button>
                </span>
              </li>
            ))
          )}
        </ol>
      </div>
    </div>
  );
}
