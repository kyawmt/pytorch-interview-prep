import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Interview Questions",
  description: "AI engineer interview questions with spoken and detailed answers.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
