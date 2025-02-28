import { toast } from "sonner";
import { useEffect, useState } from "react";

import { BASE_URL } from "@/Constants";

interface IFilePreviews {
  isFinished: boolean;
}

type FileType = {
  name: string;
  url: string;
};

const FilePreviews = ({ isFinished }: IFilePreviews) => {
  const [pdfFile, setPdfFile] = useState<FileType | null>(null);
  const [mdFile, setMdFile] = useState<FileType | null>(null);

  const getFile = async (fileType: "md" | "pdf") => {
    try {
      const response = await fetch(`${BASE_URL}/report/${fileType}`);

      if (!response.ok) {
        throw new Error(response.statusText);
      }

      const url = window.URL.createObjectURL(await response.blob());
      const contentDisposition = response.headers.get("Content-Disposition");
      let filename = `report.${fileType}`;

      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename=([^;]+)/);

        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }
      fileType === "md"
        ? setMdFile({
            name: filename,
            url: url,
          })
        : setPdfFile({
            name: filename,
            url: url,
          });
    } catch (error) {
      console.error(error);

      if (error instanceof Error) {
        toast.error("Failed to retrieve: " + error.message);
      } else {
        toast.error(
          "Failed to download report. Check the console for more details.",
        );
      }
    }
  };

  const downloadMarkdown = async () => {
    try {
      if (mdFile) {
        const link = document.createElement("a");

        link.href = mdFile.url;
        link.download = mdFile.name;
        document.body.appendChild(link);
        link.click();

        // Cleanup
        document.body.removeChild(link);
        window.URL.revokeObjectURL(mdFile.url);
      } else {
        throw Error();
      }
    } catch (error) {
      toast.error("Failed to download Markdwon report.");
    }
  };

  const downloadPDF = async () => {
    try {
      if (pdfFile) {
        const link = document.createElement("a");

        link.href = pdfFile.url;
        link.download = pdfFile.name;
        document.body.appendChild(link);
        link.click();

        // Cleanup
        document.body.removeChild(link);
        window.URL.revokeObjectURL(pdfFile.url);
      } else {
        throw Error();
      }
    } catch (error) {
      toast.error("Failed to download PDF report.");
    }
  };

  useEffect(() => {
    if (isFinished) {
      getFile("md");
      getFile("pdf");
    }
  }, [isFinished]);

  return (
    <div>
      <div className="flex space-x-4 items-center mb-4">
        <h2 className="text-lg font-bold ml-1">Files</h2>

        <div className="flex space-x-4">
          <button
            type="button"
            onClick={downloadMarkdown}
            className="cursor-pointer flex items-center justify-center sn-button bg-blue-400 hover:bg-blue-500 p-3"
          >
            Download Markdown
          </button>

          <button
            type="button"
            onClick={downloadPDF}
            className="cursor-pointer flex items-center justify-center sn-button bg-red-400 hover:bg-red-500 px-4"
          >
            Download PDF
          </button>
        </div>
      </div>

      {/* PREVIEWS */}
      <div className="flex flex-col lg:flex-row justify-center space-x-4 space-y-8 lg:space-y-0 mx-auto h-300 lg:h-150  py-4 sn-border-shadowed">
        {mdFile && (
          <object
            type="text/markdown"
            data={mdFile.url}
            className="mx-auto w-[90%] lg:w-[48%] h-[48%] lg:h-auto"
          >
            <p>
              It appears your browser doesn't support embedded Markdowns. Click{" "}
              <a
                href={mdFile.url}
                download={mdFile.name}
                className="underline text-blue-500"
              >
                here
              </a>{" "}
              to download the Markdown.
            </p>
          </object>
        )}

        {pdfFile && (
          <object
            type="application/pdf"
            title="Report"
            data={pdfFile.url}
            className="mx-auto w-[90%] lg:w-[48%] h-[48%] lg:h-auto"
          >
            <p>
              It appears your browser doesn't support embedded PDFs. Click{" "}
              <a
                href={pdfFile.url}
                download={pdfFile.name}
                className="underline text-blue-500"
              >
                here
              </a>{" "}
              to download the PDF.
            </p>
          </object>
        )}
      </div>
    </div>
  );
};

export default FilePreviews;
