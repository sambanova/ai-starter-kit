import { Outlet } from "react-router-dom";
import Header from "./Header";
import Sidebar from "./Sidebar";
import SettingsModal from "../SettingsModal";
import { useState } from "react";

const MainLayout = () => {
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);

  return (
    <div className="min-h-screen flex">
      <Sidebar />

      <div className="flex-1 flex flex-col h-screen overflow-hidden">
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
