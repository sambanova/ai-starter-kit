import { useState } from "react";
import AgentProgress from "../components/AgentProgress";
import FinancialReport from "../components/FinancialReport";
import SearchSection from "../components/SearchSection";
import WarningMessage from "../components/WarningMessage";

const Home = () => {
  const [missingKeys, setMissingKeys] = useState<string[]>([]);

  const toggleMissingKeys = () => {
    if (missingKeys.length > 0) {
      setMissingKeys([]);
    } else {
      setMissingKeys(["OpenAI", "AlphaVantage"]);
    }
  };

  return (
    <div className="h-full mb-6">
      <button onClick={toggleMissingKeys} className="cursor-pointer">
        Toggle Missing keys
      </button>
      {missingKeys.length > 0 && <WarningMessage missingKeys={missingKeys} />}
      <div className="bg-white rounded-xl shadow-md border border-gray-100 p-6 mb-6">
        <SearchSection />
      </div>

      {/* Results */}
      <div className="grid grid-cols-2 gap-6 h-full mb-6 text-center text-sm">
        <AgentProgress />

        <FinancialReport />
      </div>
    </div>
  );
};

export default Home;
