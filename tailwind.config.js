/** @type {import('tailwindcss').Config} */
//import colors, { green, slate } from "tailwindcss/colors";
const colors = require("tailwindcss/colors");
export const content = [
  "./templates/*.html",
  "./static/*.js",
  "./templates/*.css",
];
export const theme = {
  extend: {},
  colors: {
    transparent: "transparent",
    current: "currentColor",
    black: colors.black,
    white: colors.white,
    gray: colors.gray,
    emerald: colors.emerald,
    indigo: colors.indigo,
    yellow: colors.yellow,
    orange: colors.orange,
    slate: colors.slate,
    green: colors.green,
    teal: colors.teal,
    blue: colors.blue,
    red: colors.red,
    brown: {
      50: "#fdf8f6",
      100: "#f2e8e5",
      200: "#eaddd7",
      300: "#e0cec7",
      400: "#d2bab0",
      500: "#bfa094",
      600: "#a18072",
      700: "#977669",
      800: "#846358",
      900: "#43302b",
    },
  },
};
export const plugins = [];
