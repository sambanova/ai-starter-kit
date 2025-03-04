import { Settings, TriangleAlert } from "lucide-react";

import { useAPIKeysStore } from "../stores/APIKeysStore";
import { Alert, AlertDescription, AlertTitle } from "./shadcn/alert";

const WarningMessage = () => {
  const { apiKeys } = useAPIKeysStore();
  const missingKeys = Object.keys(apiKeys).filter(
    (key) => apiKeys[key as keyof typeof apiKeys] === null,
  );

  return (
    <div className="mb-4">
      {missingKeys.length > 0 && (
        <Alert variant="warning" className="py-4 rounded-xl">
          <TriangleAlert className="h-5 w-5" />
          <AlertTitle>Missing API keys</AlertTitle>
          <AlertDescription>
            Please set up your{" "}
            <span className="font-bold">{missingKeys.join(", ")}</span> API key
            {missingKeys.length > 1 ? "s" : ""} in the Settings{" "}
            <span className="inline-flex items-center align-middle w-5">
              <Settings className="relative left-0 bottom-0.5" />
            </span>{" "}
            button at the top of the page.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default WarningMessage;
