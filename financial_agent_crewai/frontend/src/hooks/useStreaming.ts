import { useStreamingStore } from "@/stores/StreamingResponseStore";

// Custom hook for easier usage in components
export const useStreaming = () => {
  const { messages, isStreaming, error, startStream, clearMessages } =
    useStreamingStore();

  return {
    messages,
    isStreaming,
    error,
    startStream,
    clearMessages,
  };
};

// Optional: Type-safe selector hooks for specific state values
export const useStreamingMessages = () =>
  useStreamingStore((state) => state.messages);
export const useStreamingStatus = () =>
  useStreamingStore((state) => ({
    isStreaming: state.isStreaming,
    error: state.error,
  }));
