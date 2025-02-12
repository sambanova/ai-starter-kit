import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { useState, useEffect } from "react";

interface ISettingsModal {
  setIsSettingsModalOpen: (isOpen: React.SetStateAction<boolean>) => void;
}

type keyType = "sambanova" | "exa" | "serper";

const SettingsModal = ({ setIsSettingsModalOpen }: ISettingsModal) => {
  const [sambanovaKey, setSambanovaKey] = useState("");
  const [serperKey, setSerperKey] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [isSambanovaKeyVisible, setIsSambanovaKeyVisible] = useState(false);
  const [isSerperKeyVisible, setIsSerperKeyVisible] = useState(false);
  const userId = "";

  const toggleKeyVisibility = (key: keyType) => {
    if (key === "sambanova") {
      setIsSambanovaKeyVisible((prev) => !prev);
    } else if (key === "serper") {
      setIsSerperKeyVisible((prev) => !prev);
    }
  };

  const clearKey = (key: keyType) => {
    localStorage.removeItem(`${key}_key_${userId}`);

    if (key === "sambanova") {
      // Clear SambaNova API Key
      setSambanovaKey("");
      setSuccessMessage("SambaNova API key cleared successfully!");
    } else if (key === "serper") {
      // Clear Serper API Key
      setSerperKey("");
      setSuccessMessage("Serper API key cleared successfully!");
    }

    clearMessagesAfterDelay();
  };

  const saveKey = (key: keyType) => {
    if (key === "sambanova") {
      if (!sambanovaKey) {
        setErrorMessage("SambaNova API Key cannot be empty");
      } else {
        localStorage.setItem(`${key}_key_${userId}`, sambanovaKey);
        setSuccessMessage("SambaNova API key saved successfully!");
      }
    } else if (key === "serper") {
      if (!sambanovaKey) {
        setErrorMessage("Serper API Key cannot be empty");
      } else {
        localStorage.setItem(`${key}_key_${userId}`, serperKey);
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

  useEffect(() => {
    setSambanovaKey(localStorage.getItem(`sambanova_key_${userId}`) || "");
    setSerperKey(localStorage.getItem(`serper_key_${userId}`) || "");
  }, []);

  return (
    <div v-if="isOpen" className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-center justify-center p-4">
        {/* Backdrop */}
        <div
          className="fixed inset-0 bg-black opacity-30"
          onClick={closeModal}
        ></div>

        {/* Modal */}
        <div className="relative w-full max-w-lg bg-white rounded-xl shadow-lg p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-gray-900">API Settings</h2>
            <button
              onClick={closeModal}
              className="text-gray-500 hover:text-gray-700"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
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

          <div className="space-y-6">
            {/* SambaNova API Key */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
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
                  value={sambanovaKey}
                  onChange={(e) => setSambanovaKey(e.target.value)}
                  type={`${isSambanovaKeyVisible ? "text" : "password"}`}
                  placeholder="Enter your SambaNova API Key"
                  className="block w-full border border-gray-300 rounded-md shadow-sm p-2 focus:outline-none focus:ring-orange-500 focus:border-orange-500 pr-10"
                />
                <button
                  onClick={() => toggleKeyVisibility("sambanova")}
                  className="absolute inset-y-0 right-0 px-3 flex items-center text-gray-500 hover:text-gray-700"
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
                  className="px-3 py-1 text-sm bg-orange-600 text-white rounded-md hover:bg-orange-700 focus:outline-none"
                >
                  Save Key
                </button>
              </div>
            </div>

            {/* Serper API Key */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Serper API Key
                <a
                  href="https://serper.dev/"
                  target="_blank"
                  className="text-orange-600 hover:text-orange-700 ml-2 text-sm"
                >
                  Get Key →
                </a>
              </label>
              <div className="relative">
                <input
                  value={serperKey}
                  onChange={(e) => setSerperKey(e.target.value)}
                  type={`${isSerperKeyVisible ? "text" : "password"}`}
                  placeholder="Enter your Serper API Key"
                  className="block w-full border border-gray-300 rounded-md shadow-sm p-2 focus:outline-none focus:ring-orange-500 focus:border-orange-500 pr-10"
                />
                <button
                  onClick={() => toggleKeyVisibility("serper")}
                  className="absolute inset-y-0 right-0 px-3 flex items-center text-gray-500 hover:text-gray-700"
                >
                  {/* Eye icon */}
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
