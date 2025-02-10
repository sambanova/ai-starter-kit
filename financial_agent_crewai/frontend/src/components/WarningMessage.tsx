import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { useState } from "react";

const WarningMessage = () => {
  const [missingKeys, setMissingKeys] = useState<string[]>([]);

  const openSettings = () => {};

  return (
    <div className="mb-4 p-4 rounded-lg bg-yellow-50 border border-yellow-200">
      <div className="flex">
        <FontAwesomeIcon
          icon={["fas", "exclamation-triangle"]}
          className="text-yellow-700"
        />
        <div>
          <p className="text-yellow-700">
            Please set up your {missingKeys.join(", ")} API key
            {missingKeys.length > 1 ? "s" : ""} in the{" "}
            <button
              onClick={openSettings}
              className="text-yellow-800 underline hover:text-yellow-900 font-medium"
            >
              settings
            </button>{" "}
            . button
          </p>
        </div>
      </div>
    </div>
  );
};

export default WarningMessage;
