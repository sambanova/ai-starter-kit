import JsonArrayNode from "./JsonArrayNode";
import JsonObjectNode from "./JsonObjectNode";
import { JsonValueType } from "./Types";

// Props for a JSON node (could be any type)
interface JsonNodeProps {
  value: JsonValueType;
  initialExpanded: boolean;
  depth: number;
}

// Component to render a single JSON node of any type
const JsonNode = ({ value, initialExpanded, depth }: JsonNodeProps) => {
  if (value === null) return <span className="text-gray-500">null</span>;

  if (typeof value === "string")
    return <span className="sn-text-primary">"{value}"</span>;

  if (typeof value === "number")
    return <span className="text-blue-600">{value}</span>;

  if (typeof value === "boolean")
    return <span className="text-purple-600">{value.toString()}</span>;

  if (Array.isArray(value)) {
    return (
      <JsonArrayNode
        array={value}
        initialExpanded={initialExpanded}
        depth={depth}
      />
    );
  }

  // Must be an object
  return (
    <JsonObjectNode
      object={value}
      initialExpanded={initialExpanded}
      depth={depth}
    />
  );
};

export default JsonNode;
