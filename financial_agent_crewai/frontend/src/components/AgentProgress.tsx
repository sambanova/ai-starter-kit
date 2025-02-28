import { useEffect, useRef } from "react";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "./shadcn/Accordion";
import { ScrollArea, ScrollBar } from "./shadcn/scroll-area";
import { useStreamingStore } from "@/stores/StreamingResponseStore";

const AgentProgress = () => {
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const { messages, isStreaming } = useStreamingStore();

  const scrollToBottom = () => {
    const scrollContainer = scrollAreaRef.current;

    if (scrollContainer) {
      // Scroll to the bottom whenever streamingContent changes
      const scrollableElement = scrollContainer.querySelector(
        "[data-radix-scroll-area-viewport]",
      );
      if (scrollableElement) {
        scrollableElement.scrollTop = scrollableElement.scrollHeight;
      }
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div>
      <h2 className="text-lg font-bold ml-1">Agent Output</h2>

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
                ref={scrollAreaRef}
                type="always"
              >
                <div className="pr-6 text-justify">
                  {messages.map((message) => (
                    <p key={message.id}>{message.content}</p>
                  ))}
                </div>
                <ScrollBar />
              </ScrollArea>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </div>
  );
};

export default AgentProgress;
