import { useState } from "react";
import { BASE_URL } from "../Constants";

type ResponseType = {
  title: string;
  summary: string;
};

export type SourcesType = {
  source_generic_search: boolean;
  source_sec_filings: boolean;
  source_yfinance_news: boolean;
  source_yfinance_stocks: boolean;
};

const useSearch = () => {
  const [response, setResponse] = useState<ResponseType | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const sendRequest = async (query: string, sources: SourcesType) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetch(`${BASE_URL}/financial_agent`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_query: query,
          source_generic_search: sources.source_generic_search,
          source_sec_filings: sources.source_sec_filings,
          source_yfinance_news: sources.source_yfinance_news,
          source_yfinance_stocks: sources.source_yfinance_stocks,
        }),
      }).then((res) => res.json());
      setResponse(data);
    } catch (error) {
      setError(error as Error);
      console.error(error);
    }
    setIsLoading(false);
  };

  return { response, isLoading, error, sendRequest };
};

export default useSearch;
