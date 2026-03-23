/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#101418",
        mist: "#eff6f2",
        mint: "#9ad3bc",
        sand: "#f6e7c1",
        ember: "#d96f4a"
      },
      boxShadow: {
        soft: "0 18px 50px rgba(16, 20, 24, 0.12)"
      }
    }
  },
  plugins: []
};
