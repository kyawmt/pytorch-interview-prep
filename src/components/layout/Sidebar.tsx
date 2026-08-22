"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAppState } from "@/hooks/useAppState";
import { overallStats } from "@/lib/progress";
import { cn } from "@/lib/utils";
import { NAV, isActive } from "./nav";

interface SidebarProps {
  /** Called after a navigation click, so the mobile drawer can close itself. */
  onNavigate?: () => void;
}

export function Sidebar({ onNavigate }: SidebarProps) {
  const pathname = usePathname();
  const state = useAppState();
  const stats = overallStats(state);

  return (
    <nav className="flex h-full flex-col" aria-label="Main">
      <div className="flex-1 overflow-y-auto px-3 py-4">
        {NAV.map((group) => (
          <div key={group.title} className="mb-5">
            <p className="px-2.5 pb-1.5 text-[10px] font-semibold uppercase tracking-[0.09em] text-faint">
              {group.title}
            </p>
            <ul className="space-y-0.5">
              {group.items.map((item) => {
                const active = isActive(pathname, item);
                const badge =
                  item.href === "/review" && stats.needingReview > 0 ? stats.needingReview : null;
                return (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      onClick={onNavigate}
                      aria-current={active ? "page" : undefined}
                      className={cn(
                        "group flex items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-sm transition-colors",
                        active
                          ? "bg-accent-soft font-medium text-accent"
                          : "text-muted hover:bg-surface-hover hover:text-text",
                      )}
                    >
                      <span
                        aria-hidden
                        className={cn("w-4 text-center text-xs", active ? "text-accent" : "text-faint")}
                      >
                        {item.icon}
                      </span>
                      <span className="flex-1 truncate">{item.label}</span>
                      {badge ? (
                        <span className="rounded-full bg-danger-soft px-1.5 py-0.5 text-[10px] font-semibold text-danger">
                          {badge}
                        </span>
                      ) : null}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>

      <div className="border-t border-border-base px-4 py-3">
        <div className="flex items-baseline justify-between text-xs">
          <span className="text-muted">Overall</span>
          <span className="font-mono font-semibold text-text">{stats.overallProgress}%</span>
        </div>
        <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-surface-2">
          <div
            className="h-full rounded-full bg-accent transition-[width] duration-500"
            style={{ width: `${stats.overallProgress}%` }}
          />
        </div>
        <p className="mt-1.5 text-[11px] text-faint">
          {stats.solved} / {stats.totalExercises} exercises
          {stats.streak > 0 ? ` · ${stats.streak} day streak` : ""}
        </p>
      </div>
    </nav>
  );
}
