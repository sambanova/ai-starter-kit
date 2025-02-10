import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { useState } from "react";

const Sidebar = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const exportAllReports = () => {};
  const clearAllConfirm = () => {};

  return (
    <div className="relative h-full">
      <button
        className="cursor-pointer absolute -right-3 top-4 z-20 p-1.5 bg-white rounded-full shadow-md hover:bg-gray-50 transition"
        onClick={() => setIsCollapsed((prevState) => !prevState)}
      >
        {isCollapsed ? (
          <FontAwesomeIcon
            icon={["fas", "chevron-right"]}
            className="h-4 w-4 text-gray-600 transition-transform duration-300"
          />
        ) : (
          <FontAwesomeIcon
            icon={["fas", "chevron-left"]}
            className="h-4 w-4 text-gray-600 transition-transform duration-300"
          />
        )}
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
                icon={["fas", "magnifying-glass"]}
                className="w-5 h-5 mr-2"
              />
              {!isCollapsed && <span>Saved Searches</span>}
            </h2>

            {!isCollapsed && (
              <div className="flex items-center space-x-2">
                <label htmlFor="filterType" className="text-sm text-gray-600">
                  Filter:
                </label>
                <select
                  id="filterType"
                  className="border border-gray-300 rounded-md p-1 text-sm focus:outline-none focus:ring-1 focus:ring-primary-500"
                >
                  <option value="all">All</option>
                  <option value="educational_content">Research</option>
                  <option value="sales_leads">Sales Leads</option>
                  <option value="financial_analysis">Financial Analysis</option>
                </select>
              </div>
            )}
          </div>

          {!isCollapsed && (
            <div className="flex items-center justify-between">
              <button
                onClick={exportAllReports}
                className="flex items-center space-x-1 text-sm text-gray-700 hover:underline focus:outline-none"
              >
                <FontAwesomeIcon
                  icon={["fas", "file-arrow-down"]}
                  className="w-4 h-4"
                />
                <span>Export All (JSON)</span>
              </button>
              <button
                onClick={clearAllConfirm}
                className="flex items-center space-x-1 text-sm text-red-600 hover:underline focus:outline-none"
              >
                <FontAwesomeIcon icon={["fas", "trash"]} className="w-4 h-4" />
                <span>Clear All</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
