interface IFinancialReport {
  result: {
    [key: string]: string;
  };
}

const FinancialReport = ({ result }: IFinancialReport) => {
  return (
    <div className="h-full my-6 text-md">
      <h2 className="text-lg font-bold ml-1 mb-4">Financial Report</h2>

      <div className="sn-border-shadowed p-6 rounded-md flex flex-col text-justify space-y-8">
        <h2 className="font-bold">{result.title} summary</h2>

        <p>{result.summary}</p>
      </div>
    </div>
  );
};

export default FinancialReport;
