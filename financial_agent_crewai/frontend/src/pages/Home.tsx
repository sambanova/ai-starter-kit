import SearchSection from "../components/SearchSection";
import WarningMessage from "../components/WarningMessage";

const Home = () => {
  return (
    <div className="h-full mb-6">
      {/* TODO: Missing keys message */}
      <div>
        <WarningMessage />
      </div>

      <div className="mb-4">
        <SearchSection />
      </div>

      {/* Results */}
      <div className="grid grid-cols-2 gap-6 h-full">
        <div className="border-black-500 border-2 rounded-md">Agent side</div>

        <div className="border-black-500 border-2 rounded-md">Output side</div>
      </div>
    </div>
  );
};

export default Home;
