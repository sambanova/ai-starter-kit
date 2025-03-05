import { useState } from "react";
import { Outlet } from "react-router-dom";

import { Box } from "@mui/material";

import Header from "./Header";
import Sidebar from "./Sidebar";

const MainLayout = () => {
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  return (
    <Box sx={{ display: "flex" }}>
      <Header onDrawerToggle={handleDrawerToggle} />
      <Sidebar mobileOpen={mobileOpen} onDrawerToggle={handleDrawerToggle} />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: {
            sm: `calc(100% - 240px)`,
            xs: "100%",
          },
          marginTop: "64px", // Adjust based on header height
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
};

export default MainLayout;
