import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Daily Practice",
  description: "Today's ten-question active-recall session.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
