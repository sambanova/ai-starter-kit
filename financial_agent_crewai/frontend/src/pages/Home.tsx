import AgentProgress from "../components/AgentProgress";
import FinancialReport from "../components/FinancialReport";
import SearchSection from "../components/SearchSection";
import WarningMessage from "../components/WarningMessage";
import { useResponseStore } from "../stores/ResponseStore";

const Home = () => {
  const { response, isLoading } = useResponseStore();

  return (
    <div className="h-full mb-6">
      <WarningMessage />

      <SearchSection />

      {/* Results */}
      {Object.values(response).some((value) => value) && (
        <div className={`${isLoading && "transition duration-100 blur-xs"}`}>
          <AgentProgress />

          <div className="h-full my-6 text-md">
            <FinancialReport result={response} />
          </div>
        </div>
      )}
    </div>
  );
};

export default Home;
