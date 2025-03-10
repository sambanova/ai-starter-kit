import { useEffect, useState } from "react";

import { AlertCircle } from "lucide-react";
import { toast } from "sonner";

import { BASE_URL } from "@/Constants";

import { Alert, AlertDescription, AlertTitle } from "./shadcn/alert";

interface IFilePreviews {
  isFinished: boolean;
}

type FileType = {
  name: string;
  url: string;
};

type MdFileType = FileType & {
  content: string;
};

const FilePreviews = ({ isFinished }: IFilePreviews) => {
  const [pdfFile, setPdfFile] = useState<FileType | null>(null);
  const [mdFile, setMdFile] = useState<MdFileType | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const getFile = async (fileType: "md" | "pdf") => {
    try {
      const response = await fetch(`${BASE_URL}/report/${fileType}`, {
        method: "GET",
        credentials: "include",
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || response.statusText);
      }

      const content = await response.blob();
      const url = window.URL.createObjectURL(content);
      const contentDisposition = response.headers.get("Content-Disposition");
      let filename = `report.${fileType}`;

      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename=([^;]+)/);

        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }
      if (fileType === "md") {
        setMdFile({
          name: filename,
          url: url,
          content: await content.text(),
        });
      } else if (fileType === "pdf") {
        setPdfFile({
          name: filename,
          url: url,
        });
      }
    } catch (er) {
      console.error(er);
      setError(er as Error);

      const typeName = fileType === "md" ? "Markdown" : "PDF";
      const errorMessage = `"Failed to retrieve ${typeName} report.`;

      if (er instanceof Error) {
        toast.error(`${errorMessage} Details: ${er.message}`);
      } else {
        toast.error(`${errorMessage} Check the console for more details.`);
      }
    }
  };

  const downloadFile = async (file: FileType | null) => {
    try {
      if (file) {
        const link = document.createElement("a");

        link.href = file.url;
        link.download = file.name;
        document.body.appendChild(link);
        link.click();

        // Cleanup
        document.body.removeChild(link);
        window.URL.revokeObjectURL(file.url);
      } else {
        throw Error();
      }
    } catch (er) {
      console.error(er);
      toast.error("Failed to download report.");
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
        <h2 className="text-lg font-bold ml-1">File previews</h2>

        {!error && (
          <div className="flex space-x-4">
            <button
              type="button"
              onClick={() => downloadFile(mdFile)}
              className="flex items-center justify-center sn-button bg-blue-400 hover:bg-blue-500 p-3"
            >
              Download Markdown
            </button>

            <button
              type="button"
              onClick={() => downloadFile(pdfFile)}
              className="flex items-center justify-center sn-button bg-red-400 hover:bg-red-500 px-4"
            >
              Download PDF
            </button>
          </div>
        )}
      </div>

      {/* PREVIEWS */}
      {error ? (
        <Alert variant="error" className="py-4 my-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>
            An error ocurred while retrieving the file previews. Details:{" "}
            {error.message}
          </AlertDescription>
        </Alert>
      ) : (
        <>
          <div className="flex flex-col lg:flex-row justify-center space-x-4 space-y-8 lg:space-y-0 mx-auto h-300 lg:h-200  py-4 sn-border-shadowed">
            {mdFile && (
              <object
                type="text/markdown"
                data={mdFile.url}
                className="mx-auto w-[90%] lg:w-[48%] h-[48%] lg:h-auto"
              >
                <p>
                  It appears your browser doesn't support embedded Markdowns.
                  Click{" "}
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
        </>
      )}
    </div>
  );
};

export default FilePreviews;
