import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Practice",
  description: "Filterable PyTorch exercise browser with instant answer checking.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
