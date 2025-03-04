import { useStreamingStore } from "@/stores/StreamingResponseStore";

type FilteredMessageType = {
  title: string | null;
  summary: string | null;
};

export const useFilteredMessages = () => {
  const { messages } = useStreamingStore();

  const finalAnswerIndex = messages.findIndex((msg) =>
    msg.output.includes('"title": '),
  );

  if (finalAnswerIndex === -1) return { title: null, summary: null };

  const filteredMessage = messages
    .slice(finalAnswerIndex)
    .map((msg) => msg.output)[0];

  const finalResult: FilteredMessageType = JSON.parse(filteredMessage);

  return finalResult;
};
