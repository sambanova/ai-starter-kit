'use client';

import { createTheme } from '@mui/material/styles';

export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#622B86',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#737075',
      contrastText: '#FFFFFF',
    },
    background: {
      default: '#fcf9fe',
      paper: '#FFFFFF',
    },
  },
  typography: {
    fontFamily: 'Inter, system-ui, -apple-system, var(--font-geist-sans), sans-serif',
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#622B86',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#FFFFFF',
        },
      },
    },
  },
});
