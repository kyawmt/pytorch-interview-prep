import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Training Loop Builder",
  description: "Assemble the PyTorch training loop from its parts.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
