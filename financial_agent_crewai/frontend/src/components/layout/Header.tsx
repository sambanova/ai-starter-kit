import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import React, { useState } from "react";

interface IHeader {
  setIsSettingsModalOpen: (isOpen: React.SetStateAction<boolean>) => void;
}

type PageModeType = "light" | "dark";

const Header = ({ setIsSettingsModalOpen }: IHeader) => {
  const [pageMode, setPageMode] = useState<PageModeType>("dark");

  const handleModeChange = () =>
    setPageMode((prevMode) => (prevMode === "light" ? "dark" : "light"));

  const openSettingsModal = () =>
    setIsSettingsModalOpen((prevState) => !prevState);

  return (
    <header className="shadow-md">
      <div className="h-16 mx-auto px-4 sm:px-6 flex items-center justify-between">
        {/* Left: Brand */}
        <div className="flex items-center space-x-2 sm:space-x-4">
          <div className="flex-shrink-0 grid grid-cols-2 gap-1">
            <img
              src="https://sambanova.ai/hubfs/logotype_sambanova_orange.png"
              alt="Samba Sales Co-Pilot Logo"
              className="h-8 block md:hidden"
            />

            <img
              src="https://sambanova.ai/hubfs/sambanova-logo-black.png"
              alt="Samba Sales Co-Pilot Logo"
              className="hidden md:h-8 md:block"
            />

            <img
              src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Ulg1BjUIxIdmOw63J5gF1Q.png"
              alt="CrewAI Logo"
              className="h-8 block"
            />
          </div>
        </div>

        {/* Right: App name & Actions */}
        <div className="flex items-center space-x-2">
          <h1 className="text-lg sm:text-2xl font-bold text-gray-900 tracking-tight text-center">
            SambaNova Financial Agent
          </h1>

          <div className="flex items-center space-x-2">
            <button
              className="cursor-pointer m-1 w-8 h-8 rounded-full hover:bg-gray-100 transition"
              onClick={handleModeChange}
            >
              <div className="text-xl">
                {pageMode === "dark" ? (
                  <FontAwesomeIcon icon={["fas", "sun"]} />
                ) : (
                  <FontAwesomeIcon icon={["fas", "moon"]} />
                )}
              </div>
            </button>

            <button
              className="cursor-pointer m-1 w-8 h-8 rounded-full hover:bg-gray-100 transition"
              onClick={openSettingsModal}
            >
              <div className="text-xl">
                <FontAwesomeIcon icon={["fas", "gear"]} />
              </div>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
