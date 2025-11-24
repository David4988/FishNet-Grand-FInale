import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import "./i18n";

import * as tf from '@tensorflow/tfjs'; 

// � 2. THE CRITICAL SHIM: Expose NPM-imported TF using legal ES Module syntax.
if (typeof window !== 'undefined') {
  (window as any).tf = tf;
}

createRoot(document.getElementById("root")!).render(<App />);
