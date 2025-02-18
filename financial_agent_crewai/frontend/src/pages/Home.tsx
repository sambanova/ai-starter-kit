import AgentProgress from "../components/AgentProgress";
import FinancialReport from "../components/FinancialReport";
import SearchSection from "../components/SearchSection";
import WarningMessage from "../components/WarningMessage";

const Home = () => {
  const results: string[] = [];

  return (
    <div className="h-full mb-6">
      <WarningMessage />

      <div className="rounded-xl shadow-md border border-gray-100 p-6 mb-6">
        <SearchSection />
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="grid grid-cols-2 gap-6 h-full mb-6 text-center text-sm">
          <AgentProgress />

          <FinancialReport />
        </div>
      )}
    </div>
  );
};

export default Home;
