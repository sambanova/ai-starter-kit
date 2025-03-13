import { AlertCircle } from "lucide-react";

import {
  StreamMessage,
  useStreamingStore,
} from "@/stores/StreamingResponseStore";

import { Alert, AlertDescription, AlertTitle } from "./shadcn/alert";

type FilteredMessageType = {
  title: string | null;
  summary: string | null;
};

const FinancialReport = () => {
  const { messages } = useStreamingStore();

  const filterFinalMessage = (messages: StreamMessage[]) => {
    const finalAnswerIndex = messages.findIndex((msg) =>
      msg.output.includes('"title": '),
    );

    if (finalAnswerIndex === -1) return { title: null, summary: null };

    const filteredMessage = messages
      .slice(finalAnswerIndex)
      .map((msg) => msg.output)[0];

    const finalResult: FilteredMessageType = JSON.parse(filteredMessage);

    return finalResult;
  };

  const finalResult = filterFinalMessage(messages);

  return (
    <>
      {Object.values(finalResult).some((value) => value) ? (
        <div className="h-full my-6 text-md">
          <h2 className="text-lg font-bold ml-1 mb-4">Financial Report</h2>

          <div className="sn-border-shadowed p-6 rounded-md flex flex-col text-justify space-y-8">
            <h2 className="font-bold">{finalResult.title} summary</h2>

            <p>{finalResult.summary}</p>
          </div>
        </div>
      ) : (
        <Alert variant="error" className="py-4 my-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>
            An error ocurred while retrieving the response's summary. Please try
            again.
          </AlertDescription>
        </Alert>
      )}
    </>
  );
};

export default FinancialReport;
