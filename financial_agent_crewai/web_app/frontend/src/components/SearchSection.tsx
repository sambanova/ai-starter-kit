import { useState } from "react";

import { LoaderCircle } from "lucide-react";

import { useAPIKeysStore } from "@/stores/APIKeysStore";
import { SourcesType } from "@/stores/StreamingResponseStore";
import { useStreamingStore } from "@/stores/StreamingResponseStore";

import MultiSelectDropdown from "./utils/MultiSelectDropdown";

const SearchSection = () => {
  const [searchQuery, setSearchQuery] = useState<string | null>(null);
  const [selectedSources, setSelectedSources] = useState<SourcesType>({
    source_generic_search: false,
    source_sec_filings: false,
    source_yfinance_news: false,
    source_yfinance_stocks: false,
  });

  const { isStreaming, startStream, clearMessages } = useStreamingStore();
  const { apiKeys } = useAPIKeysStore();

  const sources = [
    {
      id: "source_generic_search",
      label: "Generic Google Search",
    },
    { id: "source_sec_filings", label: "SEC Edgar Filings" },
    { id: "source_yfinance_news", label: "Yahoo Finance News" },
    { id: "source_yfinance_stocks", label: "Yahoo Finance Stocks" },
  ];
  const missingSambaNovaKey = Object.keys(apiKeys)
    .filter((key) => apiKeys[key as keyof typeof apiKeys] === null)
    .includes("SambaNova");
  const searchButtonIsDisabled =
    isStreaming ||
    missingSambaNovaKey || // Disable if any API key is missing
    !Object.values(selectedSources).some((value) => value) || // Disable if no source is selected
    !searchQuery?.trim(); // Disable if search query is empty

  const performSearch = () => {
    if (searchQuery) {
      clearMessages();

      startStream(searchQuery, {
        source_generic_search: selectedSources.source_generic_search,
        source_sec_filings: selectedSources.source_sec_filings,
        source_yfinance_news: selectedSources.source_yfinance_news,
        source_yfinance_stocks: selectedSources.source_yfinance_stocks,
      });

      setSearchQuery(null);
    }
  };

  const handleSelectedSources = (selectedSources: typeof sources) => {
    const updatedSources = selectedSources.reduce(
      (acc, curr) => {
        acc[curr.id as keyof SourcesType] = true;
        return acc;
      },
      {
        source_generic_search: false,
        source_sec_filings: false,
        source_yfinance_news: false,
        source_yfinance_stocks: false,
      },
    );
    setSelectedSources(updatedSources);
  };

  return (
    <div className="sn-border-shadowed p-6 mb-6">
      <h2 className="text-lg font-bold mb-2 ml-1">User query</h2>

      <div className="flex items-center space-x-4">
        <div className="relative flex-1">
          {/* Search Input */}
          <input
            type="search"
            className="w-full p-3 rounded-lg border sn-input-text truncate"
            value={searchQuery || ""}
            disabled={isStreaming}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="E.g. What is the research and development trend for Google in 2024?"
          />
        </div>

        {/* Sources dropdown */}
        <div className="flex-none">
          <MultiSelectDropdown
            options={sources}
            onChange={handleSelectedSources}
            placeholder="Select sources"
            optionName="source"
          />
        </div>

        {/* Search button */}
        <button
          type="button"
          onClick={performSearch}
          disabled={searchButtonIsDisabled}
          className="flex items-center justify-center w-25 sn-button"
        >
          {isStreaming ? <LoaderCircle className="animate-spin" /> : "Search"}
        </button>
      </div>
    </div>
  );
};

export default SearchSection;
