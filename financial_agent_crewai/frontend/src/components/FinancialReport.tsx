import { toast } from "sonner";
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

      if (!response.ok) {
        throw new Error(response.statusText);
      }

      // Create a link and click it to trigger the browser's download
      const url = window.URL.createObjectURL(await response.blob());
      const link = document.createElement("a");

      // Get filename from headers if available
      const contentDisposition = response.headers.get("Content-Disposition");
      let filename = "report.md";

      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename=([^;]+)/);

        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }

      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();

      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error(error);

      if (error instanceof Error) {
        toast.error("Failed to download report: " + error.message);
      } else {
        toast.error(
          "Failed to download report. Check the console for more details.",
        );
      }
    }
  };

  const downloadPDF = async () => {
    try {
      const response = await fetch(`${BASE_URL}/report/pdf`);

      if (!response.ok) {
        throw new Error(response.statusText);
      }

      // Create a link and click it to trigger the browser's download
      const url = window.URL.createObjectURL(await response.blob());
      const link = document.createElement("a");

      // Get filename from headers if available
      const contentDisposition = response.headers.get("Content-Disposition");
      let filename = "report.pdf";

      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename=([^;]+)/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }

      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();

      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error(error);

      if (error instanceof Error) {
        toast.error("Failed to download report: " + error.message);
      } else {
        toast.error(
          "Failed to download report. Check the console for more details.",
        );
      }
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
          className="cursor-pointer flex items-center justify-center sn-button bg-blue-400 hover:bg-blue-500"
        >
          Download Markdown
        </button>

        <button
          type="button"
          onClick={downloadPDF}
          className="cursor-pointer flex items-center justify-center sn-button bg-red-400 hover:bg-red-500"
        >
          Download PDF
        </button>
      </div>
    </div>
  );
};

export default FinancialReport;
