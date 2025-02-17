import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{vue,js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // primary: {
        //   50: "#fef7f1",
        //   100: "#feeee4",
        //   200: "#fcd9c4",
        //   300: "#fabb93",
        //   400: "#f79362",
        //   500: "#f17439",
        //   600: "#e45a1a",
        //   700: "#bd4614",
        //   800: "#983a15",
        //   900: "#7c3114",
        // },
        // primary: "var(--color-primary)",
        // secondary: "var(--color-secondary)",
        grape: "rgb(var(--grape))",
      },
    },
  },
  plugins: [],
};

export default config;
