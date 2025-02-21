import { create } from "zustand";

import { BASE_URL } from "../Constants";

export type APIResponseType = {
  title: string | null;
  summary: string | null;
};

export type SourcesType = {
  source_generic_search: boolean;
  source_sec_filings: boolean;
  source_yfinance_news: boolean;
  source_yfinance_stocks: boolean;
};

interface ResponseState {
  response: APIResponseType;
  isLoading: boolean;
  error: unknown;
  sendRequest: (query: string, selectedSources: SourcesType) => void;
}

export const useResponseStore = create<ResponseState>((set) => ({
  response: { title: null, summary: null },
  isLoading: false,
  error: null,
  sendRequest: async (query: string, selectedSources: SourcesType) => {
    set({ isLoading: true });
    const response = await fetch(`${BASE_URL}/agent/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_query: query,
        source_generic_search: selectedSources.source_generic_search,
        source_sec_filings: selectedSources.source_sec_filings,
        source_yfinance_news: selectedSources.source_yfinance_news,
        source_yfinance_stocks: selectedSources.source_yfinance_stocks,
      }),
    }).then((res) => res.json());
    set({ response });
    set({ isLoading: false });
  },
}));
