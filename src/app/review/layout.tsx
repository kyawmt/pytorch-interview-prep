import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Review",
  description: "Every exercise you got wrong, hinted or revealed.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
