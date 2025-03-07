import { useState } from "react";

import { Outlet } from "react-router-dom";

import { useThemeStore } from "../../stores/ThemeStore";
import SettingsModal from "../SettingsModal";
import Header from "./Header";

const MainLayout = () => {
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);

  const theme = useThemeStore((state) => state.theme);

  return (
    <div className="min-h-screen flex" data-theme={theme}>
      <div className="sn-backdrop sn-text-primary flex-1 flex flex-col h-screen overflow-hidden">
        <Header setIsSettingsModalOpen={setIsSettingsModalOpen} />

        <main className="flex-grow flex flex-col p-4 space-y-4 overflow-y-auto">
          <Outlet />
        </main>

        {isSettingsModalOpen && (
          <SettingsModal setIsSettingsModalOpen={setIsSettingsModalOpen} />
        )}
      </div>
    </div>
  );
};

export default MainLayout;
