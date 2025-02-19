import { useState } from "react";
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

  const isItemSelected = (item: string) => selectedItems.includes(item);

  return (
    <>
      {/* Dropdown Button */}
      <div
        onClick={() => setIsOpen(!isOpen)}
        className="w-full h-full min-w-50 sn-background border border-gray-300 rounded-lg px-4 py-3 flex items-center justify-between hover:border-gray-400"
      >
        <div className="flex flex-wrap gap-2">
          {selectedItems.length === 0 ? (
            <span className="sn-text-secondary">{placeholder}</span>
          ) : (
            <span>{selectedItems.length} sources selected</span>
          )}
        </div>

        <FontAwesomeIcon
          icon={["fas", `chevron-${isOpen ? "up" : "down"}`]}
          className="sn-icon-button-secondary transition-transform size-20 w-5"
        />
      </div>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="fixed z-10 w-1/4 mt-1 bg-background-main border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-auto">
          {options.map((option) => (
            <div
              key={option}
              onClick={() => toggleItem(option)}
              className={`px-4 py-2 hover:bg-background-secondary/70 ${
                isItemSelected(option) && "bg-background-secondary"
              } cursor-pointer flex items-center justify-between`}
            >
              <span>{option}</span>
              {isItemSelected(option) && (
                <FontAwesomeIcon
                  icon={["fas", "check"]}
                  className="text-orange-500 size-6"
                />
              )}
            </div>
          ))}
        </div>
      )}
    </>
  );
};

export default MultiSelectDropdown;
