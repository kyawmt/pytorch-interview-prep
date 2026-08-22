export interface NavItem {
  href: string;
  label: string;
  icon: string;
  /** Match sub-routes too (e.g. /cheatsheet/tensors). */
  prefix?: boolean;
  description?: string;
}

export interface NavGroup {
  title: string;
  items: NavItem[];
}

export const NAV: NavGroup[] = [
  {
    title: "Learn",
    items: [
      { href: "/", label: "Dashboard", icon: "◈", description: "Progress, weak areas, daily practice" },
      { href: "/cheatsheet", label: "Cheat Sheet", icon: "▤", prefix: true, description: "Searchable PyTorch syntax reference" },
      { href: "/practice", label: "Practice", icon: "▶", prefix: true, description: "180 exercises by topic and level" },
      { href: "/interview", label: "Interview Questions", icon: "◎", description: "Spoken answers you can rehearse" },
      { href: "/projects", label: "Mini Projects", icon: "▦", prefix: true, description: "Guided end-to-end builds" },
    ],
  },
  {
    title: "Drill",
    items: [
      { href: "/shapes", label: "Shape Playground", icon: "⤢", description: "Predict tensor shapes" },
      { href: "/builder", label: "Training Loop Builder", icon: "↻", description: "Order the steps correctly" },
      { href: "/memory", label: "Write From Memory", icon: "✎", description: "Reproduce the key patterns" },
      { href: "/challenge", label: "Interview Challenge", icon: "⏱", description: "20 questions, 30 minutes" },
      { href: "/mistakes", label: "Common Mistakes", icon: "⚠", description: "Bugs that cost interviews" },
      { href: "/rapid-review", label: "15-Minute Review", icon: "⚡", description: "Read this before the interview" },
    ],
  },
  {
    title: "Track",
    items: [
      { href: "/review", label: "Review", icon: "↺", description: "Everything you got wrong" },
      { href: "/progress", label: "Progress", icon: "▥", description: "Mastery, path, statistics" },
    ],
  },
];

export const ALL_NAV_ITEMS = NAV.flatMap((group) => group.items);

export function isActive(pathname: string, item: NavItem): boolean {
  if (item.href === "/") return pathname === "/";
  if (item.prefix) return pathname === item.href || pathname.startsWith(`${item.href}/`);
  return pathname === item.href;
}
