import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Shape Playground",
  description: "Predict tensor shapes without running any code.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
