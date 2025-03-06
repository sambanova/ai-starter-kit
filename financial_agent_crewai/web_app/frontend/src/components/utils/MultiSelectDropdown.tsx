import { useEffect, useRef, useState } from "react";

import { Check, ChevronDown, ChevronUp } from "lucide-react";

type OptionType = {
  id: string;
  label: string;
};

interface MultiSelectProps {
  options: OptionType[];
  placeholder?: string;
  optionName?: string;
  onChange?: (selectedOptions: OptionType[]) => void;
  maxHeight?: string;
  disabled?: boolean;
}

const MultiSelect = ({
  options,
  placeholder = "Select options",
  optionName = "option",
  onChange,
  maxHeight = "max-h-80",
  disabled = false,
}: MultiSelectProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selected, setSelected] = useState<OptionType[]>([]);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Handle click outside to close the dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    // Add event listener when dropdown is open
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    // Clean up event listener
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  const toggleOption = (option: OptionType) => {
    const isSelected = selected.some((item) => item.id === option.id);
    let updatedSelection: OptionType[];

    if (isSelected) {
      updatedSelection = selected.filter((item) => item.id !== option.id);
    } else {
      updatedSelection = [...selected, option];
    }

    setSelected(updatedSelection);
    onChange?.(updatedSelection);
  };

  const toggleDropdown = () => {
    if (!disabled) {
      setIsOpen(!isOpen);
    }
  };

  // Create a display string for selected items
  const getSelectionDisplay = () => {
    if (selected.length === 0) {
      return <span className="text-gray-500">{placeholder}</span>;
    }

    return (
      <span className="sn-text-primary truncate">
        {selected.length} {optionName}
        {selected.length !== 1 ? "s" : ""} selected
      </span>
    );
  };

  return (
    <div className="relative w-full min-w-50" ref={dropdownRef}>
      <div
        className={`flex items-center justify-between p-3 border rounded-md cursor-pointer ${
          isOpen ? "ring-2 ring-orange-500" : "border-gray-300"
        } ${disabled ? "bg-gray-100 cursor-not-allowed" : "sn-background"}`}
        onClick={toggleDropdown}
      >
        <div className="flex-1 truncate">{getSelectionDisplay()}</div>

        <div className="ml-2 flex-shrink-0 text-gray-600">
          {isOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </div>
      </div>

      {isOpen && (
        <div
          className={`absolute mt-1 w-full rounded-md sn-background border border-gray-300 shadow-lg z-10 ${maxHeight} overflow-y-auto`}
        >
          <ul>
            {options.map((option) => {
              const isOptionSelected = selected.some(
                (item) => item.id === option.id,
              );

              return (
                <li
                  key={option.id}
                  className={`px-3 py-2 cursor-pointer flex items-center justify-between ${
                    isOptionSelected
                      ? "sn-dropdown-selected-background"
                      : "sn-dropdown-background"
                  }`}
                  onClick={() => toggleOption(option)}
                >
                  <span>{option.label}</span>
                  {isOptionSelected && (
                    <Check size={16} className="text-orange-500" />
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
};

export default MultiSelect;
