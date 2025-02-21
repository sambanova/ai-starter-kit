import { BASE_URL } from "@/Constants";

interface IFinancialReport {
  result: {
    [key: string]: string;
  };
}

const FinancialReport = ({ result }: IFinancialReport) => {
  const downloadMarkdown = async () => {
    try {
      const response = await fetch(`${BASE_URL}/report/md`);

      // if (!response.body) {
      //   throw new Error("No response body");
      // }
      console.log(response.body);
      return response.body;
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="h-full my-6 text-md">
      <h2 className="text-lg font-bold mb-2 ml-1">Financial Report</h2>

      <div className="sn-border-shadowed p-4 mb-4 rounded-md flex flex-col text-justify space-y-8">
        <h2 className="font-bold">{result.title}</h2>

        <p>{result.summary}</p>
      </div>

      <div className="flex space-x-4">
        <button
          type="button"
          onClick={downloadMarkdown}
          // disabled={searchButtonIsDisabled}
          className="cursor-pointer disabled:cursor-default flex items-center justify-center sn-button bg-blue-400"
        >
          Download Markdown
        </button>

        <button
          type="button"
          onClick={() => console.log("")}
          // disabled={searchButtonIsDisabled}
          className="cursor-pointer disabled:cursor-default flex items-center justify-center sn-button bg-red-400"
        >
          Download PDF
        </button>
      </div>
    </div>
  );
};

export default FinancialReport;
