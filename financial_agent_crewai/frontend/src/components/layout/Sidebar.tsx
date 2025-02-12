import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { useState } from "react";
import MultiSelectDropdown from "../utils/MultiSelectDropdown";

const Sidebar = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const sources = [
    "Generic Google Search",
    "SEC Edgar Filings",
    "Yahoo Finance News",
    "Yahoo Finance Stocks",
  ];

  return (
    <div className="relative h-full">
      <button
        className="cursor-pointer absolute -right-3 top-4 z-20 p-1.5 bg-white rounded-full shadow-md hover:bg-gray-50 transition"
        onClick={() => setIsCollapsed((prevState) => !prevState)}
      >
        <FontAwesomeIcon
          icon={["fas", `chevron-${isCollapsed ? "right" : "left"}`]}
          className="h-4 w-4 text-gray-600 transition-transform duration-300"
        />
      </button>

      <div
        className={`h-full bg-white shadow-lg transition-all duration-300 overflow-hidden' ${
          isCollapsed ? "w-18" : "w-64"
        }`}
      >
        <div className="p-4 border-b border-gray-200 space-y-3">
          {/* Saved Searches title and filter on a new line */}
          <div className="space-y-2">
            <h2
              className={`font-semibold text-gray-900 whitespace-nowrap flex items-center ${
                isCollapsed ? "justify-center text-lg h-8" : ""
              }`}
            >
              <FontAwesomeIcon
                icon={["fas", "newspaper"]}
                className="w-5 h-5 mr-2"
              />
              {!isCollapsed && <span>Sources</span>}
            </h2>
          </div>

          {!isCollapsed ? <MultiSelectDropdown options={sources} /> : null}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
