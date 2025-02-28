import { useState } from "react";
import { LoaderCircle } from "lucide-react";

import MultiSelectDropdown from "./utils/MultiSelectDropdown";
import { SourcesType } from "@/stores/ResponseStore";
import { useAPIKeysStore } from "@/stores/APIKeysStore";
import { useStreamingStore } from "@/stores/StreamingResponseStore";

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

  const sources = {
    source_generic_search: "Generic Google Search",
    source_sec_filings: "SEC Edgar Filings",
    source_yfinance_news: "Yahoo Finance News",
    source_yfinance_stocks: "Yahoo Finance Stocks",
  };
  // const sources = {
  //   source_generic_search: { name: "Generic Google Search", checked: false },
  //   source_sec_filings: { name: "SEC Edgar Filings", checked: false },
  //   source_yfinance_news: { name: "Yahoo Finance News", checked: false },
  //   source_yfinance_stocks: { name: "Yahoo Finance Stocks", checked: false },
  // };
  const missingKeys = Object.keys(apiKeys).filter(
    (key) => apiKeys[key as keyof typeof apiKeys] === null,
  );
  const searchButtonIsDisabled =
    isStreaming ||
    missingKeys.length > 0 || // Disable if any API key is missing
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

  const handleSelectedSources = (source: string, value: boolean) => {
    setSelectedSources((prevState) => ({ ...prevState, [source]: value }));
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
            placeholder="Select sources"
            handleSelectedItems={handleSelectedSources}
          />
        </div>

        {/* Search button */}
        <button
          type="button"
          onClick={performSearch}
          disabled={searchButtonIsDisabled}
          className="cursor-pointer disabled:cursor-default flex items-center justify-center w-25 px-6 py-3 sn-button rounded-lg hover:bg-primary-700 disabled:opacity-50"
        >
          {isStreaming ? <LoaderCircle className="animate-spin" /> : "Search"}
        </button>
      </div>
    </div>
  );
};

export default SearchSection;
