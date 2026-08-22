import type { Metadata, Viewport } from "next";
import "./globals.css";
import { AppShell } from "@/components/layout/AppShell";
import { ThemeScript } from "@/components/layout/ThemeScript";

export const metadata: Metadata = {
  title: {
    default: "PyTorch Prep — interview practice for AI engineers",
    template: "%s · PyTorch Prep",
  },
  description:
    "Active-recall PyTorch practice for AI / ML engineer interviews: cheat sheets, coding exercises, interview questions and guided mini projects.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0c0f" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeScript />
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
