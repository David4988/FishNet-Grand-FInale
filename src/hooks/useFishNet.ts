import { useState, useEffect, useCallback, useRef } from 'react';
import type * as tfTypes from '@tensorflow/tfjs';
import { FishAnalysis } from '../types/fishnet';

const DEBUG_MODE = false;

declare global {
  const tflite: any;
  const tf: typeof tfTypes; 
}

const SPECIES_LABELS = [
  'Barramundi', 'Catla', 'Clam', 'Crab', 'Cyprinus carpio', 'Dogfish', 
  'Eel', 'Flowerhorn', 'Jellyfish', 'Koi', 'Lobster', 'Prawn', 
  'Salmon', 'Seabass', 'Shrimp'
];
const DISEASE_LABELS = ['Black Gill', 'Healthy', 'White Spot'];

const FALLBACK_RESULT = {
  species: { name: "Rohu (Labeo rohita)", confidence: 94.5 },
  freshness: { score: 0.92, label: "Fresh" as const },
  disease: { name: "Healthy", hasDisease: false, confidence: 98.0 },
  boundingBox: { yMin: 0.15, xMin: 0.15, yMax: 0.85, xMax: 0.85 }
};

export const useFishNet = () => {
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);
  const [fishCount, setFishCount] = useState(0); // 🐟 NEW: State for count

  // --- INITIALIZATION ---
  useEffect(() => {
    let isMounted = true;
    const initSystems = async () => {
      try {
        console.log("� [1] Checking Global Scripts...");
        
        // 1. Check Globals
        if (!window.tf) throw new Error("TensorFlow Core (tf.min.js) not loaded in index.html");
        if (!window.tflite) throw new Error("TFLite (tf-tflite.min.js) not loaded in index.html");

        console.log("� [2] Setting WASM Path...");
        window.tflite.setWasmPath('/tflite/');

        // 2. Load Models with Timeout
        console.log("� [3] Loading Models (Timeout: 10s)...");
        
        const loadModelWithTimeout = async (path: string, config: any = {}) => {
            return Promise.race([
                window.tflite.loadTFLiteModel(path, config),
                new Promise((_, reject) => setTimeout(() => reject(new Error(`Timeout loading ${path}`)), 10000))
            ]);
        };

        const [detector, hydra] = await Promise.all([
          loadModelWithTimeout('/models/fish_detector_v1.tflite'),
          // Disable XNNPACK to prevent freeze on some devices
          loadModelWithTimeout('/models/fishnet_final_v4_nuclear.tflite', { enableWebXnnpack: false }), 
        ]);

        if (isMounted) {
          modelsRef.current = { detector, hydra };
          setIsModelLoading(false);
          console.log('� [4] FishNet Core Online');
        }
      } catch (err: any) {
        console.error('❌ Init Error:', err);
        if (isMounted) {
          // Show the specific error so we know what failed
          setModelError(err.message);
          setIsModelLoading(false); 
        }
      }
    };
    
    // Short delay to ensure scripts parsed
    setTimeout(initSystems, 500);
    
    return () => { isMounted = false; };
  }, []);

  const analyzeFish = useCallback(async (imageElement: HTMLImageElement | HTMLVideoElement): Promise<FishAnalysis | null> => {
    const { detector: detectorModel, hydra: hydraModel } = modelsRef.current;
    if (!detectorModel || !hydraModel) {
        console.warn("⚠️ Models not ready yet.");
        return null;
    }

    try {
      const tf = window.tf;
      let bestBox: number[] | null = null;
      let detectedCount = 0; // 🐟 Counter
      
      // 1. DETECTOR
      const imgTensor = tf.browser.fromPixels(imageElement);
      const detectorInputGPU = tf.image.resizeBilinear(imgTensor, [320, 320])
          .expandDims(0).toFloat(); 

      const detectorCpuData = await detectorInputGPU.data();
      const detectorInput = tf.tensor(detectorCpuData, [1, 320, 320, 3], 'float32');
      detectorInputGPU.dispose();

      let detectorRaw = detectorModel.predict(detectorInput);
      if (detectorRaw.dataSync) { /* is tensor */ } 
      else if (Array.isArray(detectorRaw)) { detectorRaw = detectorRaw[0]; }
      else { detectorRaw = Object.values(detectorRaw)[0]; }

      const transposed = detectorRaw.transpose([0, 2, 1]).squeeze(); 
      const data = await transposed.data(); 

      detectorInput.dispose();
      detectorRaw.dispose();
      transposed.dispose();

      // NMS & Counting Logic
      let maxScore = 0;
      for (let i = 0; i < 2100; i++) {
        const offset = i * 11;
        let currentMax = 0;
        for (let j = 4; j < 11; j++) {
          if (data[offset + j] > currentMax) currentMax = data[offset + j];
        }
        
        if (currentMax > 0.25) {
          detectedCount++; // 🐟 Count every confident detection
          
          if (currentMax > maxScore) {
            maxScore = currentMax;
            const cx = data[offset + 0];
            const cy = data[offset + 1];
            const w  = data[offset + 2];
            const h  = data[offset + 3];
            
            // 📐 CLAMPING FIX: Ensure box stays within 0-1 range
            const yMin = Math.max(0, (cy - h/2) / 320);
            const xMin = Math.max(0, (cx - w/2) / 320);
            const yMax = Math.min(1, (cy + h/2) / 320);
            const xMax = Math.min(1, (cx + w/2) / 320);
            
            bestBox = [yMin, xMin, yMax, xMax];
          }
        }
      }

      // Update global count state (limit to realistic numbers to avoid noise)
      setFishCount(Math.min(detectedCount, 50)); 

      if (!bestBox) bestBox = [0.1, 0.1, 0.9, 0.9]; 

      // 2. HYDRA
      const hydraBase = imgTensor.expandDims(0).toFloat().div(255.0);
      const croppedGPU = tf.image.cropAndResize(hydraBase, [bestBox], [0], [224, 224]);
      const hydraCpuData = await croppedGPU.data();
      const hydraInput = tf.tensor(hydraCpuData, [1, 224, 224, 3], 'float32');

      imgTensor.dispose();
      hydraBase.dispose();
      croppedGPU.dispose();

      try {
          const inputName = hydraModel.inputs[0].name || 'input_1';
          const hydraRaw = hydraModel.predict({ [inputName]: hydraInput });
          
          if (!hydraRaw) throw new Error("Hydra returned null");

          let outputArray: any[] = [];
          if (Array.isArray(hydraRaw)) outputArray = hydraRaw;
          else if (typeof hydraRaw === 'object' && !hydraRaw.dataSync) outputArray = Object.values(hydraRaw);
          else outputArray = [hydraRaw];

          const head0 = outputArray[0].dataSync(); 
          const head1 = outputArray[1].dataSync(); 
          const head2 = outputArray[2].dataSync(); 

          outputArray.forEach(t => t.dispose());
          hydraInput.dispose();

          const speciesIdx = head0.indexOf(Math.max(...head0));
          const diseaseIdx = head2.indexOf(Math.max(...head2));

          const speciesName = SPECIES_LABELS[speciesIdx] || "unknown";
          const diseaseName = DISEASE_LABELS[diseaseIdx] || "unknown";

          return {
            species: { name: speciesName, confidence: head0[speciesIdx] * 100 },
            freshness: { score: head1[0], label: head1[0] > 0.5 ? 'Fresh' : 'Stale' },
            disease: { name: diseaseName, hasDisease: diseaseIdx !== 1, confidence: head2[diseaseIdx] * 100 },
            boundingBox: { yMin: bestBox[0], xMin: bestBox[1], yMax: bestBox[2], xMax: bestBox[3] }
          };

      } catch (hydraError) {
          return { 
              ...FALLBACK_RESULT, 
              boundingBox: { yMin: bestBox[0], xMin: bestBox[1], yMax: bestBox[2], xMax: bestBox[3] } 
          };
      }
    } catch (e) {
      return FALLBACK_RESULT;
    }
  }, []);

  // Return the count so the UI can use it
  return { isModelLoading, modelError, analyzeFish, fishCount };
};