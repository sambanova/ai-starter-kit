import { useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import MultiSelectDropdown from "./utils/MultiSelectDropdown";

const SearchSection = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [searchQuery, setSearchQuery] = useState<string | null>(null);

  const sources = [
    "Generic Google Search",
    "SEC Edgar Filings",
    "Yahoo Finance News",
    "Yahoo Finance Stocks",
  ];

  const toggleRecording = () => {
    setIsRecording((prevState) => !prevState);
  };

  const performSearch = () => {
    setIsLoading(true);
    console.log(searchQuery);

    setTimeout(() => {
      setIsLoading(false);
      setSearchQuery("");
    }, 1000);
  };

  return (
    <>
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
            <FontAwesomeIcon icon={["fas", "microphone"]} beat={isRecording} />
          </button>
        </div>

        {/* Sources dropdown */}
        <div className="flex-none">
          <MultiSelectDropdown options={sources} placeholder="Select sources" />
        </div>

        {/* Search button */}
        <button
          type="button"
          onClick={performSearch}
          disabled={isLoading || !searchQuery?.trim()}
          className="cursor-pointer disabled:cursor-default w-25 px-6 py-3 sn-button rounded-lg hover:bg-primary-700 disabled:opacity-50"
        >
          <span>
            {isLoading ? (
              <FontAwesomeIcon icon={["fas", "spinner"]} spin />
            ) : (
              "Search"
            )}
          </span>
        </button>
      </div>

      {isRecording && (
        <div className="mt-2 text-sm sn-text-primary flex items-center space-x-2">
          <span className="inline-block w-2 h-2 bg-red-500 rounded-full animate-pulse" />
          <span>Recording... Click microphone again to stop</span>
        </div>
      )}
    </>
  );
};

export default SearchSection;
