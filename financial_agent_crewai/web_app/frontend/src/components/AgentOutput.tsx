import { useStreamingStore } from "@/stores/StreamingResponseStore";

import AgentCard from "./AgentCard";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "./shadcn/Accordion";

interface AgentOutputProps {
  initialExpanded?: boolean;
}

const AgentOutput: React.FC<AgentOutputProps> = ({
  initialExpanded = false,
}: AgentOutputProps) => {
  const { messages, isStreaming } = useStreamingStore();
  const messagesArray = Array.isArray(messages) ? messages : [messages];

  return (
    <div>
      <h2 className="text-xl font-bold ml-1">Agent Output</h2>

      <div className="sn-border-shadowed px-6 rounded-lg flex flex-col justify-center space-y-8">
        <Accordion type="single" collapsible>
          <AccordionItem value="item-1">
            <AccordionTrigger>
              Details
              {isStreaming && (
                <span className="text-md sn-text-secondary animate-pulse">
                  [Streaming messages]
                </span>
              )}
            </AccordionTrigger>

            <AccordionContent>
              <div className="p-4 max-w-4xl mx-auto">
                <div className="space-y-4">
                  {messagesArray.map((task, index) => (
                    <AgentCard
                      key={index}
                      task={task}
                      initialExpanded={initialExpanded}
                    />
                  ))}
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </div>
  );
};

export default AgentOutput;
