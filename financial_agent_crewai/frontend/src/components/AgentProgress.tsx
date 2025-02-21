import { useStreaming } from "@/hooks/useStreaming";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "./shadcn/Accordion";
import { ScrollArea, ScrollBar } from "./shadcn/scroll-area";
import { Alert, AlertDescription, AlertTitle } from "./shadcn/alert";
import { AlertCircle } from "lucide-react";

const AgentProgress = () => {
  const { messages, error, isStreaming } = useStreaming();
  console.log(messages);

  return (
    <div>
      <h2 className="text-lg font-bold mb-2 ml-1">Agent Output</h2>

      {error ? (
        <Alert variant="error" className="py-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : (
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
                <ScrollArea
                  className={`${messages.length > 0 && "h-100"} pr-6`}
                  type="always"
                >
                  <div className="pr-6 text-justify">
                    {messages.map((message) => (
                      <p key={message.id}>{message.content}</p>
                    ))}
                  </div>
                  <ScrollBar id="hello" />
                </ScrollArea>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>
      )}
    </div>
  );
};

export default AgentProgress;
