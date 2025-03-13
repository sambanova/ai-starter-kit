import { create } from "zustand";

import { BASE_URL } from "@/Constants";

export type RequestSourcesType = {
  source_generic_search: boolean;
  source_sec_filings: boolean;
  source_yfinance_news: boolean;
  source_yfinance_stocks: boolean;
};

export interface StreamMessage {
  id: string;
  timestamp: string;
  task_name: string;
  task: string;
  agent: string;
  status: string;
  output: string;
}

interface StreamingState {
  messages: StreamMessage[];
  isStreaming: boolean;
  isFinished: boolean;
  error: string | null;
  startStream: (
    query: string,
    selectedSources: RequestSourcesType,
  ) => Promise<void>;
  clearMessages: () => void;
  setError: (error: string | null) => void;
  setIsStreaming: (isStreaming: boolean) => void;
}

export const useStreamingStore = create<StreamingState>((set) => ({
  messages: [],
  isStreaming: false,
  isFinished: false,
  error: null,

  clearMessages: () => set({ messages: [] }),

  setError: (error: string | null) => set({ error }),

  setIsStreaming: (isStreaming: boolean) => set({ isStreaming }),

  startStream: async (query: string, selectedSources: RequestSourcesType) => {
    set({ isStreaming: true, isFinished: false, error: null });

    try {
      const response = await fetch(`${BASE_URL}/flow/run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({
          user_query: query,
          source_generic_search: selectedSources.source_generic_search,
          source_sec_filings: selectedSources.source_sec_filings,
          source_yfinance_news: selectedSources.source_yfinance_news,
          source_yfinance_stocks: selectedSources.source_yfinance_stocks,
        }),
      });

      if (!response.ok) {
        console.error(response);
        const errorData = await response.json();
        throw new Error(errorData.detail || response.statusText);
      }

      if (!response.body) {
        throw new Error("No response body");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const chunks = text.split("\n").filter((chunk) => chunk.trim());

        chunks.forEach((chunk) => {
          set((state) => ({
            messages: [
              ...state.messages,
              {
                id: crypto.randomUUID(),
                ...JSON.parse(chunk),
              },
            ],
          }));
        });
      }
    } catch (err) {
      set({ error: err instanceof Error ? err.message : "An error occurred" });
    } finally {
      set({ isStreaming: false, isFinished: true });
    }
  },
}));
