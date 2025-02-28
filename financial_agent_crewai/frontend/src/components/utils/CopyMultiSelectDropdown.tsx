import {
  DropdownMenuCheckboxItemProps,
  DropdownMenuSeparator,
} from "@radix-ui/react-dropdown-menu";
import { useState } from "react";

import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "../shadcn/dropdown-menu";

type CheckedType = DropdownMenuCheckboxItemProps["checked"];

interface IMultiSelectDropdownProps {
  options: { [key: string]: { name: string; checked: boolean } };
  // handleSelectedItems: (source: string, value: boolean) => void;
  placeholder?: string;
}

const MultiSelectDropdown = ({
  options,
  // handleSelectedItems,
  placeholder = "Select items...",
}: IMultiSelectDropdownProps) => {
  const [showStatusBar, setShowStatusBar] = useState<CheckedType>(true);
  const [showActivityBar, setShowActivityBar] = useState<CheckedType>(false);
  const [showPanel, setShowPanel] = useState<CheckedType>(false);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button className="border border-gray-400 dark:border-gray-100/50 rounded-lg my-4 p-3">
          {placeholder}
        </button>
      </DropdownMenuTrigger>

      <DropdownMenuContent>
        {Object.entries(options).map(([key, value]) => (
          <>
            <DropdownMenuCheckboxItem
              key={key}
              checked={value.checked}
              onCheckedChange={setShowStatusBar}
            >
              {value.name}
              <span></span>
            </DropdownMenuCheckboxItem>
            <DropdownMenuSeparator />
          </>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default MultiSelectDropdown;
