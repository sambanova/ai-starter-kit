import { useEffect } from "react";

import { AlertCircle } from "lucide-react";
import { toast } from "sonner";

import AgentOutput from "@/components/AgentOutput";
import FilePreviews from "@/components/FilePreviews";
import FinancialReport from "@/components/FinancialReport";
import SearchSection from "@/components/SearchSection";
import WarningMessage from "@/components/WarningMessage";
import LoadingSpinner from "@/components/layout/LoadingSpinner";
import { Alert, AlertDescription, AlertTitle } from "@/components/shadcn/alert";

import { useStreamingStore } from "@/stores/StreamingResponseStore";

const Home = () => {
  const { isStreaming, isFinished, error } = useStreamingStore();

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
          {(isStreaming || isFinished) && <AgentOutput />}

          {isStreaming ? (
            <LoadingSpinner message="Getting results..." />
          ) : (
            isFinished && (
              <>
                <div
                  className={`${isStreaming && "transition duration-100 blur-xs"}`}
                >
                  <FinancialReport />
                </div>

                <FilePreviews isFinished={isFinished} />
              </>
            )
          )}
        </>
      )}
    </div>
  );
};

export default Home;
