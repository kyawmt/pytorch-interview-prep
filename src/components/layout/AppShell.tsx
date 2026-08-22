"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { ALL_NAV_ITEMS, isActive } from "./nav";
import { SearchDialog } from "./SearchDialog";
import { Sidebar } from "./Sidebar";
import { ThemeToggle } from "./ThemeToggle";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const pathname = usePathname();

  useKeyboardShortcuts([
    { key: "k", meta: true, allowInInput: true, handler: () => setSearchOpen((open) => !open), description: "Search" },
    { key: "/", handler: () => setSearchOpen(true), description: "Search" },
  ]);

  const current = ALL_NAV_ITEMS.find((item) => isActive(pathname, item));

  return (
    <div className="min-h-screen bg-bg">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-[60] focus:rounded-md focus:bg-accent focus:px-3 focus:py-2 focus:text-sm focus:text-accent-text"
      >
        Skip to content
      </a>

      {/* Desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-60 flex-col border-r border-border-base bg-bg-subtle lg:flex">
        <div className="flex h-14 items-center gap-2 border-b border-border-base px-4">
          <Link href="/" className="flex items-center gap-2 text-sm font-semibold text-text">
            <span
              aria-hidden
              className="flex h-6 w-6 items-center justify-center rounded-md bg-accent text-[13px] font-bold text-accent-text"
            >
              ⚡
            </span>
            PyTorch Prep
          </Link>
        </div>
        <Sidebar />
      </aside>

      {/* Mobile drawer */}
      {menuOpen ? (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setMenuOpen(false)}
            aria-hidden
          />
          <div className="absolute inset-y-0 left-0 flex w-72 flex-col border-r border-border-base bg-bg-subtle animate-fade-in-up">
            <div className="flex h-14 items-center justify-between border-b border-border-base px-4">
              <span className="text-sm font-semibold">PyTorch Prep</span>
              <button
                type="button"
                onClick={() => setMenuOpen(false)}
                aria-label="Close menu"
                className="rounded-md p-1.5 text-muted hover:bg-surface-hover hover:text-text"
              >
                ✕
              </button>
            </div>
            <Sidebar onNavigate={() => setMenuOpen(false)} />
          </div>
        </div>
      ) : null}

      <div className="lg:pl-60">
        <header className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b border-border-base bg-bg/85 px-4 backdrop-blur-md sm:px-6">
          <button
            type="button"
            onClick={() => setMenuOpen(true)}
            aria-label="Open menu"
            aria-expanded={menuOpen}
            className="rounded-md p-1.5 text-muted hover:bg-surface-hover hover:text-text lg:hidden"
          >
            ☰
          </button>

          <span className="truncate text-sm font-medium text-text lg:hidden">PyTorch Prep</span>
          <span className="hidden truncate text-sm text-muted lg:block">
            {current?.label ?? "PyTorch Prep"}
          </span>

          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick={() => setSearchOpen(true)}
              className="flex items-center gap-2 rounded-lg border border-border-base bg-surface-2 px-2.5 py-1.5 text-xs text-muted transition-colors hover:border-border-strong hover:text-text"
              aria-label="Search (Cmd+K)"
            >
              <span aria-hidden>⌕</span>
              <span className="hidden sm:inline">Search</span>
              <kbd className="hidden rounded border border-border-base bg-surface px-1 py-0.5 font-mono text-[10px] sm:inline">
                ⌘K
              </kbd>
            </button>
            <ThemeToggle />
          </div>
        </header>

        <main id="main" className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6 sm:py-8">
          {children}
        </main>
      </div>

      {/* Mounted only while open so it always starts from a clean state. */}
      {searchOpen ? <SearchDialog onClose={() => setSearchOpen(false)} /> : null}
    </div>
  );
}
