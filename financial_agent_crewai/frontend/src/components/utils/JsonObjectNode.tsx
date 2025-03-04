import { useState } from "react";

import JsonNode from "./JsonNode";
import { JsonObjectType } from "./Types";
import { ChevronDown, ChevronRight } from "lucide-react";

interface JsonObjectNodeProps {
  object: JsonObjectType;
  initialExpanded: boolean;
  depth: number;
}

const JsonObjectNode = ({
  object,
  initialExpanded,
  depth,
}: JsonObjectNodeProps) => {
  const [isExpanded, setIsExpanded] = useState(initialExpanded);

  const keys = Object.keys(object);

  // Empty object
  if (keys.length === 0) {
    return <span className="text-gray-500">{"{}"}</span>;
  }

  return (
    <div className="font-mono">
      <div
        className="flex items-center cursor-pointer hover:sn-background-secondary rounded-md p-1"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {isExpanded ? (
          <ChevronDown className="h-4 w-4 sn-icon-button" />
        ) : (
          <ChevronRight className="h-4 w-4 sn-icon-button" />
        )}

        <span className="mr-2 sn-text-primary">{"{"} </span>

        {!isExpanded && (
          <span className="sn-text-tertiary italic">
            {keys.length} {keys.length === 1 ? "property" : "properties"}
          </span>
        )}

        {!isExpanded && <span className="sn-text-primary ml-2">{" }"}</span>}
      </div>

      {isExpanded && (
        <div className="ml-8">
          {keys.map((key, index) => (
            <div key={key} className="flex flex-wrap items-start my-1">
              <span className="sn-text-tertiary mr-1">"{key}"</span>

              <span className="mr-1">:</span>

              <JsonNode
                value={object[key]}
                initialExpanded={depth < 1}
                depth={depth + 1}
              />

              {index < keys.length - 1 && <span>,</span>}
            </div>
          ))}

          <div>{"}"}</div>
        </div>
      )}
    </div>
  );
};

export default JsonObjectNode;
