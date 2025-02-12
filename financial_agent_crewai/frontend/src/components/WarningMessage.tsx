import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

interface IWarningMessageProps {
  missingKeys: string[];
}

const WarningMessage = ({ missingKeys }: IWarningMessageProps) => {
  return (
    <div className="mb-4 p-4 rounded-lg bg-yellow-50 border border-yellow-200">
      <div className="flex items-center space-x-2">
        <FontAwesomeIcon
          icon={["fas", "exclamation-triangle"]}
          className="text-yellow-700"
        />
        <div>
          <p className="text-yellow-700">
            Please set up your{" "}
            <span className="font-bold">{missingKeys.join(", ")}</span> API key
            {missingKeys.length > 1 ? "s" : ""} in the Settings{" "}
            <span className="text-yellow-800 font-medium">
              <FontAwesomeIcon icon={["fas", "gear"]} />
            </span>{" "}
            button at the top of the page.
          </p>
        </div>
      </div>
    </div>
  );
};

export default WarningMessage;
