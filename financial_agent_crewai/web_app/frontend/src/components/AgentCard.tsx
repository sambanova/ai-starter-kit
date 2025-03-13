import { useEffect, useState } from "react";

import { CheckCircle, ChevronDown, ChevronRight, Clock } from "lucide-react";

import { StreamMessage } from "@/stores/StreamingResponseStore";

import JsonNode from "./utils/JsonNode";
import { JsonValueType } from "./utils/Types";

interface AgentCardProps {
  task: StreamMessage;
  initialExpanded: boolean;
}

const AgentCard = ({ task, initialExpanded }: AgentCardProps) => {
  const [isExpanded, setIsExpanded] = useState(initialExpanded);
  const [parsedOutput, setParsedOutput] = useState<JsonValueType | null>(null);

  const AgentTasksColors: { [key: string]: string } = {
    context_analysis_task: "bg-blue-100 text-blue-800",
    reformulation_task: "bg-green-100 text-green-800",
    research_task: "bg-yellow-100 text-yellow-800",
    rag_research_task: "bg-red-100 text-red-800",
    reporting_task: "bg-purple-100 text-purple-800",
    sec_research_task: "bg-blue-100 text-blue-800",
    extraction_task: "bg-green-100 text-green-800",
    summarization: "bg-green-100 text-green-800",
    yahoo_finance_research_task: "bg-yellow-100 text-yellow-800",
    yfinance_stock_analysis_task: "bg-red-100 text-red-800",
    default: "bg-gray-100 text-gray-800",
  };

  // Parse the output string to JSON if possible
  const parseStringToJson = (str: string) => {
    try {
      if (str) {
        setParsedOutput(JSON.parse(str));
      }
    } catch (error) {
      console.error("Error parsing output JSON:", error);
      setParsedOutput(str);
    }
  };

  // Format the timestamp to be more readable
  const formatTimestamp = (timestamp: string) => {
    try {
      const utcTimestamp = timestamp.replace(" ", "T") + "Z";
      const date = new Date(utcTimestamp);

      return date.toLocaleTimeString();
    } catch (e) {
      console.error(e);

      return timestamp;
    }
  };

  useEffect(() => {
    parseStringToJson(task.output);
  }, [task.output]);

  return (
    <div className="sn-border sn-background overflow-hidden">
      {/* Card header */}
      <div className="p-4 sn-background-secondary border-b flex flex-col sm:flex-row sm:justify-between sm:items-center">
        <div className="flex items-center space-x-2 justify-between">
          <span
            className={`px-2 py-1 rounded text-sm font-medium ${
              AgentTasksColors[task.task_name] || AgentTasksColors.default
            }`}
          >
            {task.task_name}
          </span>

          <span className="sn-text-agent-name font-medium text-right">
            {task.agent}
          </span>
        </div>

        {/* Task status and timestamp */}
        <div className="flex items-center space-x-3 text-sm text-gray-500  justify-between">
          <span className="flex items-center">
            <Clock className="h-4 w-4 mr-1" />
            {formatTimestamp(task.timestamp)}
          </span>

          {task.status === "completed" && (
            <span className="flex items-center text-green-600">
              <CheckCircle className="h-4 w-4 mr-1" />
              {task.status}
            </span>
          )}
        </div>
      </div>

      {/* Card content */}
      <div className="p-4">
        <div className="mb-3">
          <h3 className="text-sm font-semibold sn-text-tertiary mb-1">Task:</h3>
          <p className="sn-text-primary">{task.task}</p>
        </div>

        {/* Output */}
        <div>
          <div
            className="flex items-center cursor-pointer sn-background hover:bg-gray-100 dark:hover:bg-background-secondary rounded-md p-1 mb-2"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? (
              <ChevronDown className="h-4 w-4 sn-text-tertiary" />
            ) : (
              <ChevronRight className="h-4 w-4 sn-text-tertiary" />
            )}
            <h3 className="text-sm font-semibold sn-text-tertiary ml-1">
              Output:
            </h3>
          </div>

          {isExpanded && parsedOutput && (
            <div className="bg-gray-50 dark:bg-background-secondary rounded-md p-3">
              <JsonNode value={parsedOutput} initialExpanded={true} depth={0} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AgentCard;
