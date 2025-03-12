import { createTheme } from "@mui/material/styles";

const lightTheme = createTheme({
  palette: {
    primary: {
      main: "#0d3660",
      light: "#0071b2",
    },
    secondary: {
      main: "#00ffff",
      light: "#00ffc5",
      dark: "#0043ff",
    },
    info: {
      main: "#0071b2",
    },
    text: {
      primary: "#151515",
      secondary: "#fafafa",
    },
    background: {
      default: "#f5f5f5",
      paper: "#fff",
    },
    error: {
      main: "#A83128",
    },
  },
  typography: {
    fontFamily: "Roboto, sans-serif",
  },
});

export { lightTheme };
