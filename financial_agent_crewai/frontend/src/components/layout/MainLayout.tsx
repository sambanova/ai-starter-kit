import { Outlet } from "react-router-dom";
import { useState } from "react";

import Header from "./Header";
import SettingsModal from "../SettingsModal";
import { useThemeStore } from "../../stores/ThemeStore";

const MainLayout = () => {
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);

  const theme = useThemeStore((state) => state.theme);

  return (
    <div className="min-h-screen flex" data-theme={theme}>
      <div className="sn-backdrop flex-1 flex flex-col h-screen overflow-hidden">
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
