import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Interview Challenge",
  description: "20 questions, 30 minutes, with a per-topic breakdown.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
