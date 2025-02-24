import { Outlet } from "react-router-dom";
import Header from "./Header";
import Sidebar from "./Sidebar";

const MainLayout = () => {
  return (
    <div>
      <Header />
      <Sidebar />
      Main Layout
      <main>
        <Outlet />
      </main>
    </div>
  );
};

export default MainLayout;
