import AgentProgress from "@/components/AgentProgress";
import FinancialReport from "@/components/FinancialReport";
import SearchSection from "@/components/SearchSection";
import WarningMessage from "@/components/WarningMessage";
import { useFilteredMessages } from "@/hooks/useFilteredMessages";
import { useStreaming } from "@/hooks/useStreaming";

const Home = () => {
  const { isStreaming } = useStreaming();

  const { finalResult } = useFilteredMessages();

  return (
    <div className="h-full mb-6">
      <WarningMessage />

      <SearchSection />

      {/* Results */}
      <AgentProgress />

      {Object.values(finalResult).some((value) => value) && (
        <div className={`${isStreaming && "transition duration-100 blur-xs"}`}>
          <FinancialReport result={finalResult} />
        </div>
      )}
    </div>
  );
};

export default Home;
