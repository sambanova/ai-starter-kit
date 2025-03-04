import { Alert, AlertDescription, AlertTitle } from "./shadcn/alert";
import { AlertCircle } from "lucide-react";

import { useFilteredMessages } from "@/utils/useFilteredMessages";

const FinancialReport = () => {
  const finalResult = useFilteredMessages();

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
