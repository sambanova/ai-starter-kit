import { useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

import { useAPIKeysStore } from "../stores/APIKeysStore";

interface ISettingsModal {
  setIsSettingsModalOpen: (isOpen: React.SetStateAction<boolean>) => void;
}

type keyType = "sambanova" | "serper";

const SettingsModal = ({ setIsSettingsModalOpen }: ISettingsModal) => {
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [isSambanovaKeyVisible, setIsSambanovaKeyVisible] = useState(false);
  const [isSerperKeyVisible, setIsSerperKeyVisible] = useState(false);

  const { apiKeys, addApiKey, updateApiKey } = useAPIKeysStore();
  const [sambanovaKey, setSambanovaKey] = useState<string | null>(
    apiKeys.SambaNova
  );
  const [serperKey, setSerperKey] = useState<string | null>(apiKeys.Serper);

  const toggleKeyVisibility = (key: keyType) => {
    if (key === "sambanova") {
      setIsSambanovaKeyVisible((prev) => !prev);
    } else if (key === "serper") {
      setIsSerperKeyVisible((prev) => !prev);
    }
  };

  const clearKey = (key: keyType) => {
    localStorage.removeItem(`${key}_key`);

    if (key === "sambanova") {
      updateApiKey("SambaNova", null);
      setSuccessMessage("SambaNova API key cleared successfully!");
    } else if (key === "serper") {
      updateApiKey("Serper", null);
      setSuccessMessage("Serper API key cleared successfully!");
    }

    clearMessagesAfterDelay();
  };

  const saveKey = (key: keyType) => {
    if (key === "sambanova") {
      if (!sambanovaKey) {
        setErrorMessage("SambaNova API Key cannot be empty");
      } else {
        addApiKey("SambaNova", sambanovaKey);
        setSuccessMessage("SambaNova API key saved successfully!");
      }
    } else if (key === "serper") {
      if (!serperKey) {
        setErrorMessage("Serper API Key cannot be empty");
      } else {
        addApiKey("Serper", serperKey);
        setSuccessMessage("Serper API key saved successfully!");
      }
    }

    clearMessagesAfterDelay();
  };

  const clearMessagesAfterDelay = () => {
    setTimeout(() => {
      setErrorMessage("");
      setSuccessMessage("");
    }, 3000);
  };

  const closeModal = () => setIsSettingsModalOpen((prevState) => !prevState);

  return (
    <div v-if="isOpen" className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-center justify-center p-4">
        {/* Backdrop */}
        <div
          className="fixed inset-0 bg-black opacity-40"
          onClick={closeModal}
        ></div>

        {/* Modal */}
        <div className="relative w-full max-w-lg sn-background-secondary sn-text-primary rounded-xl shadow-lg p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold">API Settings</h2>
            <button
              onClick={closeModal}
              className="cursor-pointer sn-icon-button"
            >
              <FontAwesomeIcon icon={["fas", "times"]} size="lg" />
            </button>
          </div>

          {/* TODO: convert this into a toast at the top */}
          {errorMessage && (
            <div className="mb-4 p-3 bg-red-100 text-red-700 rounded">
              {errorMessage}
            </div>
          )}

          {successMessage && (
            <div className="mb-4 p-3 bg-green-100 text-green-700 rounded">
              {successMessage}
            </div>
          )}

          <div className="space-y-6 bg-grape">
            {/* SambaNova API Key */}
            <div>
              <label className="block text-sm font-medium mb-1">
                SambaNova API Key
                <a
                  href="https://cloud.sambanova.ai/"
                  target="_blank"
                  className="text-orange-600 hover:text-orange-700 ml-2 text-sm"
                >
                  Get Key →
                </a>
              </label>
              <div className="relative">
                <input
                  value={sambanovaKey || ""}
                  onChange={(e) => setSambanovaKey(e.target.value)}
                  type={`${isSambanovaKeyVisible ? "text" : "password"}`}
                  placeholder="Enter your SambaNova API Key"
                  className="block w-full border pr-10 sn-input-text"
                />
                <button
                  onClick={() => toggleKeyVisibility("sambanova")}
                  className="absolute inset-y-0 right-0 px-3 flex items-center sn-icon-button-secondary"
                >
                  {/* Eye icon */}
                  <FontAwesomeIcon
                    icon={[
                      "fas",
                      `eye${isSambanovaKeyVisible ? "-slash" : ""}`,
                    ]}
                  />
                </button>
              </div>
              {/* Save and Clear Buttons */}
              <div className="flex justify-end space-x-2 mt-2">
                <button
                  onClick={() => clearKey("sambanova")}
                  className="px-3 py-1 text-sm bg-red-500 text-white rounded-md hover:bg-red-600 focus:outline-none"
                >
                  Clear Key
                </button>
                <button
                  onClick={() => saveKey("sambanova")}
                  className="px-3 py-1 text-sm bg-orange-500 text-white rounded-md hover:bg-orange-600 focus:outline-none"
                >
                  Save Key
                </button>
              </div>
            </div>

            {/* Serper API Key */}
            <div>
              <label className="block text-sm font-medium mb-1">
                Serper API Key
                <a
                  href="https://serper.dev/"
                  target="_blank"
                  className="text-orange-500 hover:text-orange-600 ml-2 text-sm"
                >
                  Get Key →
                </a>
              </label>
              <div className="relative">
                <input
                  value={serperKey || ""}
                  onChange={(e) => setSerperKey(e.target.value)}
                  type={`${isSerperKeyVisible ? "text" : "password"}`}
                  placeholder="Enter your Serper API Key"
                  className="block w-full border pr-10 sn-input-text"
                />
                <button
                  onClick={() => toggleKeyVisibility("serper")}
                  className="absolute inset-y-0 right-0 px-3 flex items-center sn-icon-button-secondary"
                >
                  {/* Eye icon */}
                  <FontAwesomeIcon
                    icon={["fas", `eye${isSerperKeyVisible ? "-slash" : ""}`]}
                  />
                </button>
              </div>
              {/* Add Save and Clear Buttons for Serper */}
              <div className="flex justify-end space-x-2 mt-2">
                <button
                  onClick={() => clearKey("serper")}
                  className="px-3 py-1 text-sm bg-red-500 text-white rounded-md hover:bg-red-600 focus:outline-none"
                >
                  Clear Key
                </button>
                <button
                  onClick={() => saveKey("serper")}
                  className="px-3 py-1 text-sm bg-orange-600 text-white rounded-md hover:bg-orange-700 focus:outline-none"
                >
                  Save Key
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
