import React from "react";

import { Moon, Settings, Sun } from "lucide-react";

import { useThemeStore } from "../../stores/ThemeStore";

interface IHeader {
  setIsSettingsModalOpen: (isOpen: React.SetStateAction<boolean>) => void;
}

const Header = ({ setIsSettingsModalOpen }: IHeader) => {
  const theme = useThemeStore((state) => state.theme);
  const toggleTheme = useThemeStore((state) => state.toggleTheme);

  const openSettingsModal = () =>
    setIsSettingsModalOpen((prevState) => !prevState);

  return (
    <header className="sn-header">
      <div className="h-16 mx-auto px-4 sm:px-6 flex items-center justify-between">
        {/* Left: Brand */}
        <div className="flex items-center text-left space-x-2 sm:space-x-4">
          <img
            src="https://sambanova.ai/hubfs/logotype_sambanova_orange.png"
            alt="SambaNova Logo"
            className="h-8 block md:hidden"
          />

          {theme === "dark" ? (
            <img
              src="https://sambanova.ai/hubfs/sambanova-logo-white.png"
              alt="SambaNova Logo"
              className="hidden md:h-8 md:block"
            />
          ) : (
            <img
              src="https://sambanova.ai/hubfs/sambanova-logo-black.png"
              alt="SambaNova Logo"
              className="hidden md:h-8 md:block"
            />
          )}

          <img
            src="https://cdn.prod.website-files.com/66cf2bfc3ed15b02da0ca770/66d07240057721394308addd_Logo%20(1).svg"
            alt="CrewAI Logo"
            className="h-8 block"
          />
        </div>

        {/* Right: App name & Actions */}
        <div className="flex items-center space-x-4">
          <h1 className="text-md sm:text-2xl font-bold tracking-tight text-center">
            SambaNova Financial Agent
          </h1>

          <div className="flex items-center">
            <button
              className="cursor-pointer m-1 w-6 h-6 rounded-full sn-icon-button"
              onClick={toggleTheme}
            >
              {theme === "dark" ? <Sun /> : <Moon />}
            </button>

            <button
              className="cursor-pointer m-1 w-6 h-6 rounded-full sn-icon-button"
              onClick={openSettingsModal}
            >
              <Settings />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
