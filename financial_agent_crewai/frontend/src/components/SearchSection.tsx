import { useState } from "react";

import MultiSelectDropdown from "./utils/MultiSelectDropdown";
import { SourcesType } from "@/stores/ResponseStore";
import { useResponseStore } from "@/stores/ResponseStore";
import { useAPIKeysStore } from "@/stores/APIKeysStore";
import { useStreaming } from "@/hooks/useStreaming";
import { LoaderCircle, Mic } from "lucide-react";

const SearchSection = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [searchQuery, setSearchQuery] = useState<string | null>(null);
  const [selectedSources, setSelectedSources] = useState<SourcesType>({
    source_generic_search: false,
    source_sec_filings: false,
    source_yfinance_news: false,
    source_yfinance_stocks: false,
  });

  const { isLoading, sendRequest } = useResponseStore();
  const { startStream, clearMessages } = useStreaming();
  const { apiKeys } = useAPIKeysStore();
  const missingKeys = Object.keys(apiKeys).filter(
    (key) => apiKeys[key as keyof typeof apiKeys] === null
  );

  const sources = {
    source_generic_search: "Generic Google Search",
    source_sec_filings: "SEC Edgar Filings",
    source_yfinance_news: "Yahoo Finance News",
    source_yfinance_stocks: "Yahoo Finance Stocks",
  };
  const searchButtonIsDisabled =
    isLoading ||
    missingKeys.length > 0 || // Disable if any API key is missing
    !Object.values(selectedSources).some((value) => value) || // Disable if no source is selected
    !searchQuery?.trim(); // Disable if search query is empty

  const toggleRecording = () => {
    setIsRecording((prevState) => !prevState);
  };

  const performSearch = () => {
    if (searchQuery) {
      clearMessages();

      sendRequest(searchQuery, {
        source_generic_search: selectedSources.source_generic_search,
        source_sec_filings: selectedSources.source_sec_filings,
        source_yfinance_news: selectedSources.source_yfinance_news,
        source_yfinance_stocks: selectedSources.source_yfinance_stocks,
      });

      startStream(searchQuery, {
        source_generic_search: selectedSources.source_generic_search,
        source_sec_filings: selectedSources.source_sec_filings,
        source_yfinance_news: selectedSources.source_yfinance_news,
        source_yfinance_stocks: selectedSources.source_yfinance_stocks,
      });
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
            className="w-full p-3 pr-12 rounded-lg border sn-input-text truncate"
            value={searchQuery || ""}
            disabled={isLoading}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="E.g. What is the research and development trend for Google in 2024?"
          />

          {/* TODO: Voice Input Button */}
          <button
            onClick={toggleRecording}
            disabled={isLoading}
            className="hidden cursor-pointer absolute right-4 top-1/2 transform -translate-y-1/2 p-2 sn-icon-button rounded-full transition-colors"
            title="Voice Search"
          >
            <Mic className={`${isRecording && "animate-pulse"}`} />
          </button>
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
          {isLoading ? <LoaderCircle className="animate-spin" /> : "Search"}
        </button>
      </div>

      {isRecording && (
        <div className="mt-2 text-sm sn-text-primary flex items-center space-x-2">
          <span className="inline-block w-2 h-2 bg-red-500 rounded-full animate-pulse" />
          <span>Recording... Click microphone again to stop</span>
        </div>
      )}
    </div>
  );
};

export default SearchSection;
