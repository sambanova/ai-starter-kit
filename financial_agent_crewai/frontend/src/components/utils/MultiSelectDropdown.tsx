import { useEffect, useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

interface IMultiSelectDropdownProps {
  options?: string[];
  placeholder?: string;
}

const MultiSelectDropdown = ({
  options = [],
  placeholder = "Select items...",
}: IMultiSelectDropdownProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedItems, setSelectedItems] = useState<string[]>([]);

  const toggleItem = (item: string) => {
    setSelectedItems((prev) =>
      prev.includes(item) ? prev.filter((i) => i !== item) : [...prev, item]
    );
  };

  const removeItem = (itemToRemove: string) => {
    setSelectedItems((prev) => prev.filter((item) => item !== itemToRemove));
  };

  const isItemSelected = (item: string) => selectedItems.includes(item);

  useEffect(() => {
    console.log("Mounted");
  }, []);

  return (
    <div className="relative w-full max-w-md">
      {/* Dropdown Button */}
      <div
        onClick={() => setIsOpen(!isOpen)}
        className="w-full min-h-10 bg-white border border-gray-300 rounded-lg px-4 py-2 flex items-center justify-between cursor-pointer hover:border-gray-400"
      >
        <div className="flex flex-wrap gap-2">
          {selectedItems.length === 0 ? (
            <span className="text-gray-500">{placeholder}</span>
          ) : (
            selectedItems.map((item) => (
              <>
                <span
                  key={item}
                  className="bg-blue-100 text-blue-800 px-2 py-1 rounded-md flex items-center gap-1 text-sm"
                >
                  <FontAwesomeIcon
                    icon={["fab", "google"]}
                    className="size-14"
                  />
                  {/* {item} */}
                  {/* <FontAwesomeIcon
                    icon={["fas", "times"]}
                    className="cursor-pointer hover:text-blue-600 size-14"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeItem(item);
                    }}
                  /> */}
                </span>
              </>
            ))
          )}
        </div>

        <FontAwesomeIcon
          icon={["fas", `chevron-${isOpen ? "up" : "down"}`]}
          className="text-gray-500 transition-transform size-20 w-5"
        />
      </div>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-auto">
          {options.map((option) => (
            <div
              key={option}
              onClick={() => toggleItem(option)}
              className={`px-4 py-2 hover:bg-gray-100 ${
                isItemSelected(option) && "bg-gray-200"
              } cursor-pointer flex items-center justify-between`}
            >
              <span>{option}</span>
              {isItemSelected(option) && (
                <FontAwesomeIcon
                  icon={["fas", "check"]}
                  className="text-blue-600 size-16"
                />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default MultiSelectDropdown;
