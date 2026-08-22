import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Write From Memory",
  description: "Reproduce the core PyTorch patterns from a blank editor.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
