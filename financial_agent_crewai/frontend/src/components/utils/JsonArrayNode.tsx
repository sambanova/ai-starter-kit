import { useState } from "react";

import JsonNode from "./JsonNode";
import { JsonArrayType } from "./Types";
import { ChevronDown, ChevronRight } from "lucide-react";

interface JsonArrayNodeProps {
  array: JsonArrayType;
  initialExpanded: boolean;
  depth: number;
}

const JsonArrayNode = ({
  array,
  initialExpanded,
  depth,
}: JsonArrayNodeProps) => {
  const [isExpanded, setIsExpanded] = useState(initialExpanded);

  // Empty array
  if (array.length === 0) {
    return <span className="text-gray-500">[]</span>;
  }

  const paddingLeft = `${depth * 1.5}rem`;

  return (
    <div className="font-mono">
      <div
        className="flex items-center cursor-pointer hover:bg-gray-200 dark:hover:bg-background-secondary/50 rounded-md p-1"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {isExpanded ? (
          <ChevronDown className="h-4 w-4 text-gray-500" />
        ) : (
          <ChevronRight className="h-4 w-4 text-gray-500" />
        )}

        <span className="mr-2 sn-text-primary">{"["}</span>

        {!isExpanded && (
          <span className="text-gray-500 italic">
            {array.length} {array.length === 1 ? "item" : "items"}
          </span>
        )}

        {!isExpanded && <span className="sn-text-primary ml-2">{"]"}</span>}
      </div>

      {isExpanded && (
        <div className="ml-4" style={{ paddingLeft }}>
          {array.map((item, index) => (
            <div key={index} className="flex items-start my-1">
              <span className="text-gray-400 mr-2">{index}:</span>

              <JsonNode
                value={item}
                initialExpanded={depth < 1}
                depth={depth + 1}
              />

              {index < array.length - 1 && <span>,</span>}
            </div>
          ))}

          <div style={{ paddingLeft: `${depth * 1.5 - 1.5}rem` }}>{"]"}</div>
        </div>
      )}
    </div>
  );
};

export default JsonArrayNode;
