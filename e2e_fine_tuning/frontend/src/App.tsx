import { Route, Routes } from "react-router-dom";

import Home from "./pages/Home";
import MainLayout from "./components/layout/MainLayout";
import Dataset from "./pages/Dataset";

function App() {
  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route index element={<Home />} />
        <Route path="/dataset" element={<Dataset />} />
      </Route>
    </Routes>
  );
}

export default App;
