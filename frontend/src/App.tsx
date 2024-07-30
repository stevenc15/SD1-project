import { Box, CssBaseline, ThemeProvider } from "@mui/material";
import { createTheme } from "@mui/material/styles";
import { useMemo } from "react";
import { themeSettings } from "./theme";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Navbar from "@/scenes/navbar";
import Dashboard from "@/scenes/dashboard";
import Predictions from "@/scenes/predictions/predictions.jsx";
import Collection from "@/scenes/collection/collection.jsx";
import About from "@/scenes/about/about.jsx";
// import { DataProvider } from "./context/DataContext.jsx";

function App() {
  const theme = useMemo(() => createTheme(themeSettings), []);
  return (
    <div className="app">
      <BrowserRouter>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          {/* <DataProvider> */}
          <Box width="100%" height="100%" padding="1rem 2rem 4rem 2rem">
            <Navbar />
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/collection" element={<Collection />} />
              <Route path="/about" element={<About />} />
            </Routes>
          </Box>
          {/* </DataProvider> */}
        </ThemeProvider>
      </BrowserRouter>
    </div>
  );
}

export default App;
