import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Progress",
  description: "Topic mastery, learning path and challenge history.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
