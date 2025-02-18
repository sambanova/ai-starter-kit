import { useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

const SearchSection = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

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
      <div className="flex items-center space-x-4">
        <div className="relative flex-1">
          {/* Search Input */}
          <input
            type="search"
            className="w-full p-3 pr-12 rounded-lg border border-gray-300 focus:ring-2 focus:ring-primary-500 focus:border-transparent truncate"
            value={searchQuery}
            disabled={isLoading}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="E.g. What is the research and development trend for Google in 2024?"
          />

          {/* Voice Input Button */}
          <button
            onClick={toggleRecording}
            disabled={isLoading}
            className="cursor-pointer absolute right-4 top-1/2 transform -translate-y-1/2 p-2 hover:bg-gray-100 rounded-full transition-colors"
            title="Voice Search"
          >
            <FontAwesomeIcon
              icon={["fas", "microphone"]}
              beat={isRecording}
              className="text-orange-500"
            />
          </button>
        </div>

        {/* Search button */}
        <button
          type="button"
          onClick={performSearch}
          disabled={isLoading || isRecording || !searchQuery.trim()}
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
        <div className="mt-2 text-sm text-gray-600 flex items-center space-x-2">
          <span className="inline-block w-2 h-2 bg-red-500 rounded-full animate-pulse" />
          <span>Recording... Click microphone again to stop</span>
        </div>
      )}
    </>
  );
};

export default SearchSection;
