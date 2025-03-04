import { useState } from "react";

import { Check, ChevronDown, ChevronUp } from "lucide-react";

interface IMultiSelectDropdownProps {
  options: { [key: string]: string };
  handleSelectedItems: (source: string, value: boolean) => void;
  placeholder?: string;
}

const MultiSelectDropdown = ({
  options,
  handleSelectedItems,
  placeholder = "Select items...",
}: IMultiSelectDropdownProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedItems, setSelectedItems] = useState<string[]>([]);

  const toggleItem = (item: string) => {
    setSelectedItems((prev) =>
      prev.includes(item) ? prev.filter((i) => i !== item) : [...prev, item],
    );
    handleSelectedItems(item, !selectedItems.includes(item));
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
            <span className="sn-text-primary">
              {selectedItems.length} source{selectedItems.length > 1 ? "s" : ""}{" "}
              selected
            </span>
          )}
        </div>

        {isOpen ? <ChevronUp /> : <ChevronDown />}
      </div>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="fixed z-10 w-1/5 mt-1 sn-background border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-auto">
          {Object.keys(options).map((option) => (
            <div
              key={option}
              onClick={() => toggleItem(option)}
              className={`px-4 py-2 border-b border-gray-300 ${
                isItemSelected(option)
                  ? "sn-dropdown-selected-background"
                  : "sn-dropdown-background"
              } cursor-pointer flex items-center justify-between`}
            >
              {options[option]}
              <span className="mx-1">
                {isItemSelected(option) && (
                  <Check className="text-orange-500 h-5" />
                )}
              </span>
            </div>
          ))}
        </div>
      )}
    </>
  );
};

export default MultiSelectDropdown;
