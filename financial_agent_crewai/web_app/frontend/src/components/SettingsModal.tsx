import { useState } from "react";

import { Eye, EyeOff, Info, X } from "lucide-react";
import { toast } from "sonner";

import { useAPIKeysStore } from "@/stores/APIKeysStore";

import Tooltip from "./utils/Tooltip";

import { BASE_URL } from "@/Constants";

interface ISettingsModal {
  setIsSettingsModalOpen: (isOpen: React.SetStateAction<boolean>) => void;
}

type keyType = "sambanova" | "serper";

const SettingsModal = ({ setIsSettingsModalOpen }: ISettingsModal) => {
  const [isSambanovaKeyVisible, setIsSambanovaKeyVisible] = useState(false);
  const [isSerperKeyVisible, setIsSerperKeyVisible] = useState(false);

  const { apiKeys, addApiKey, updateApiKey } = useAPIKeysStore();
  const [sambanovaKey, setSambanovaKey] = useState<string | null>(
    apiKeys.SambaNova,
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
    if (key === "sambanova") {
      updateApiKey("SambaNova", null);
      setSambanovaKey(null);
      toast.success("SambaNova API key cleared successfully");
    } else if (key === "serper") {
      updateApiKey("Serper", null);
      setSerperKey(null);
      toast.success("Serper API key cleared successfully");
    }
  };

  const saveKeys = async () => {
    if (!sambanovaKey) {
      toast.error("SambaNova API Key cannot be empty.");
    } else {
      try {
        const response = await fetch(`${BASE_URL}/keys`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-sambanova-key": sambanovaKey,
            "x-serper-key": serperKey || "",
          },
          credentials: "include", // TODO: if backend and frontend are going to be on the same domain, change to "same-origin"
        });

        if (!response.ok)
          throw new Error(`${response.status} - ${response.statusText}`);
      } catch (error) {
        console.error(error);

        if (error instanceof Error) {
          toast.error(`Failed to save API keys. Detail: ${error.message}`);
        }
      }

      addApiKey("SambaNova", sambanovaKey);
      if (serperKey) addApiKey("Serper", serperKey);
      toast.success("API keys saved successfully!");
    }
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
              <X />
            </button>
          </div>

          <div className="space-y-6 bg-grape">
            {/* SambaNova API Key */}
            <div>
              <label className="block text-sm font-medium mb-1">
                SambaNova API Key <span className="text-red-400">*</span>
                <a
                  href="https://cloud.sambanova.ai/"
                  target="_blank"
                  className="text-orange-500 hover:text-orange-600 ml-2 text-sm"
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
                  {isSambanovaKeyVisible ? <EyeOff /> : <Eye />}
                </button>
              </div>
              {/* Clear Button */}
              <div className="flex justify-end space-x-2 mt-2">
                <button
                  onClick={() => clearKey("sambanova")}
                  className="sn-clear-button"
                >
                  Clear Key
                </button>
              </div>
            </div>

            {/* Serper API Key */}
            <div>
              <label className="inline-block align-middle text-sm font-medium mb-1 space-x-1">
                Serper API Key <span className="font-bold">(Optional)</span>{" "}
                <span className="inline-flex items-center align-middle w-5">
                  <Tooltip text="This API key is only required if you want to make a query using the 'Generic Google Search' source.">
                    <Info />
                  </Tooltip>
                </span>
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
                  type={`${isSerperKeyVisible ? "search" : "password"}`}
                  placeholder="Enter your Serper API Key"
                  className="block w-full border pr-10 sn-input-text"
                />
                <button
                  onClick={() => toggleKeyVisibility("serper")}
                  className="absolute inset-y-0 right-0 px-3 flex items-center sn-icon-button-secondary"
                >
                  {isSerperKeyVisible ? <EyeOff /> : <Eye />}
                </button>
              </div>
              {/* Clear Button */}
              <div className="flex justify-end space-x-2 mt-2">
                <button
                  onClick={() => clearKey("serper")}
                  className="sn-clear-button"
                >
                  Clear Key
                </button>
              </div>
            </div>

            <div className="flex justify-center space-x-2 mt-2">
              <button
                onClick={saveKeys}
                className="cursor-pointer focus:outline-none px-4 py-2 text-sm sn-button"
              >
                Save Keys
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
