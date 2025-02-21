import { APIResponseType } from "../stores/ResponseStore";

type IFInancialReport = {
  result: APIResponseType;
};

const FinancialReport = ({ result }: IFInancialReport) => {
  return (
    <>
      <h2 className="text-lg font-bold mb-2 ml-1">Financial Report</h2>

      <div className="sn-border-shadowed p-4 rounded-md flex flex-col text-justify space-y-8">
        <h2 className="font-bold">{result.title}</h2>
        <p>{result.summary}</p>
      </div>
    </>
  );
};

export default FinancialReport;
