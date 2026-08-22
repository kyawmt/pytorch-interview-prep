import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "15-Minute Review",
  description: "The concepts to refresh in the hour before an interview.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
