import { useStreamingStore } from "@/stores/StreamingResponseStore";

export const useFilteredMessages = () => {
  const { messages } = useStreamingStore();
  const finalAnswerIndex = messages.findIndex((msg) =>
    msg.content.includes('"title": '),
  );

  if (finalAnswerIndex === -1) return { finalResult: {} };

  const filteredMessages = messages
    .slice(finalAnswerIndex)
    .map((msg) => msg.content);

  const extractToObject = (array: string[]) => {
    const result: { [key: string]: string } = {};

    array.forEach((str) => {
      // Find key and value using regex
      const matches = str.match(/"([^"]+)":\s*"([^"]+)"/);
      if (matches) {
        const [_, key, value] = matches;
        result[key] = value;
      }
    });

    return result;
  };

  const finalResult = extractToObject(filteredMessages);

  return { finalResult };
};
