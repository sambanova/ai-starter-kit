import { Route, Routes } from "react-router-dom";

import { CssBaseline, ThemeProvider, createTheme } from "@mui/material";

import MainLayout from "./components/layout/MainLayout";

import Checkpoint from "./pages/Checkpoint";
import Dataset from "./pages/Dataset";
import Deployment from "./pages/Deployment";
import FineTuning from "./pages/FineTuning";
import Home from "./pages/Home";

function App() {
  const theme = createTheme();

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<Home />} />
          <Route path="/dataset" element={<Dataset />} />
          <Route path="/checkpoint" element={<Checkpoint />} />
          <Route path="/fine_tuning" element={<FineTuning />} />
          <Route path="/deployment" element={<Deployment />} />
        </Route>
      </Routes>
    </ThemeProvider>
  );
}

export default App;
