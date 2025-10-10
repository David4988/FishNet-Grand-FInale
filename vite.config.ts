import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import fs from "fs";                    // 👈 add this
import { componentTagger } from "lovable-tagger";

export default defineConfig(({ mode }) => ({
  server: {
    host: "0.0.0.0",                    // lets other devices connect
    port: 5000,
    // strictPort: true,
    allowedHosts: ["*"],
    // https: {                           // 👈 add this block
    //   key: fs.readFileSync("./192.168.0.5-key.pem"),
    //   cert: fs.readFileSync("./192.168.0.5.pem")
    // }
    // no need for the old HMR section unless you really want Replit hot reload
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  optimizeDeps: {
    exclude: ["sql.js"],
  },
  define: {
    global: "globalThis",
  },
}));
