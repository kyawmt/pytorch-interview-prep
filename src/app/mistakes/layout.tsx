import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Common Mistakes",
  description: "The PyTorch bugs that cost interviews, with symptoms and fixes.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
