import { notFound } from "next/navigation";
import { TOPICS, TOPIC_MAP } from "@/data/topics";
import { exercisesByTopic } from "@/data/exercises";
import type { TopicId } from "@/lib/types";
import { TopicPracticeClient } from "./TopicPracticeClient";

const PRACTICE_TOPICS = TOPICS.filter((topic) => exercisesByTopic(topic.id).length > 0);

export function generateStaticParams() {
  return PRACTICE_TOPICS.map((topic) => ({ topic: topic.id }));
}

export async function generateMetadata({ params }: { params: Promise<{ topic: string }> }) {
  const { topic } = await params;
  const meta = TOPIC_MAP[topic as TopicId];
  return meta ? { title: `${meta.title} — Practice`, description: meta.summary } : {};
}

export default async function PracticeTopicPage({
  params,
}: {
  params: Promise<{ topic: string }>;
}) {
  const { topic } = await params;
  const meta = TOPIC_MAP[topic as TopicId];
  if (!meta || exercisesByTopic(topic as TopicId).length === 0) notFound();
  return <TopicPracticeClient topic={topic as TopicId} />;
}
