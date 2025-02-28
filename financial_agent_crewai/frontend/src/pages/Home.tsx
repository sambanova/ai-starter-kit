import { AlertCircle } from "lucide-react";
import { toast } from "sonner";
import { useEffect } from "react";

import AgentProgress from "@/components/AgentProgress";
import FinancialReport from "@/components/FinancialReport";
import SearchSection from "@/components/SearchSection";
import { Alert, AlertTitle, AlertDescription } from "@/components/shadcn/alert";
import WarningMessage from "@/components/WarningMessage";
import { useFilteredMessages } from "@/hooks/useFilteredMessages";
import FilePreviews from "@/components/FilePreviews";
import { useStreamingStore } from "@/stores/StreamingResponseStore";

const Home = () => {
  const { isStreaming, isFinished, error } = useStreamingStore();

  const { finalResult } = useFilteredMessages();

  useEffect(() => {
    if (isFinished) {
      toast.success("Finished search");
    }
  }, [isFinished]);

  return (
    <div className="h-full mb-6">
      <WarningMessage />

      <SearchSection />

      {/* Results */}
      {error ? (
        <Alert variant="error" className="py-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>
            An error ocurred while retrieving your response. Details: {error}
          </AlertDescription>
        </Alert>
      ) : (
        <>
          {(isStreaming || isFinished) && <AgentProgress />}

          {isFinished && (
            <>
              <div
                className={`${isStreaming && "transition duration-100 blur-xs"}`}
              >
                <FinancialReport result={finalResult} />
              </div>

              <FilePreviews isFinished={isFinished} />
            </>
          )}
        </>
      )}
    </div>
  );
};

export default Home;
