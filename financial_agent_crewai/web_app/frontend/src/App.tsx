import { Route, Routes } from "react-router-dom";

import MainLayout from "./components/layout/MainLayout";
import Home from "./pages/Home";

function App() {
  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route index element={<Home />} />
      </Route>
    </Routes>
  );
}

export default App;
