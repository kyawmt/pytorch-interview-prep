import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Mini Projects",
  description: "Guided PyTorch mini projects you complete line by line.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
