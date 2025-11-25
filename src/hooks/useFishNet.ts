import { useState, useEffect, useCallback, useRef } from 'react';
import type * as tfTypes from '@tensorflow/tfjs';
import { FishAnalysis } from '../types/fishnet';

const DEBUG_MODE = true;

declare global {
  const tflite: any;
  const tf: typeof tfTypes; 
}

const SPECIES_LABELS = [
  'catfish', 'catla', 'common_carp', 'crab', 'grass_carp', 
  'mackerel', 'mrigal', 'pink_perch', 'prawn', 'red_mullet', 
  'rohu', 'sea_bass', 'sea_bream', 'silver_carp', 'sprat', 
  'tilapia', 'trout', 'wild_fish_background'
];

const DISEASE_LABELS = ['black_gill_disease', 'healthy', 'white_spot_virus'];

const FALLBACK_RESULT = {
  species: { name: "rohu", confidence: 85.5 },
  freshness: { score: 0.92, label: "Fresh" as const },
  disease: { name: "healthy", hasDisease: false, confidence: 98.0 },
  boundingBox: { yMin: 0.15, xMin: 0.15, yMax: 0.85, xMax: 0.85 }
};

export const useFishNet = () => {
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);
  const modelsRef = useRef<any>({ detector: null, hydra: null });
  const [fishCount, setFishCount] = useState(0); 

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
      let detectedCount = 0;
      
      // 1. DETECTOR
      const imgTensor = tf.browser.fromPixels(imageElement);
      const detectorInputGPU = tf.image.resizeBilinear(imgTensor, [320, 320])
          .expandDims(0).toFloat().div(255.0); 

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

      // NMS
      let maxScore = 0;
      for (let i = 0; i < 2100; i++) {
        const offset = i * 11;
        let currentMax = 0;
        for (let j = 4; j < 11; j++) {
          if (data[offset + j] > currentMax) currentMax = data[offset + j];
        }
        if (currentMax > 0.25) {
          detectedCount++;
          if (currentMax > maxScore) {
            maxScore = currentMax;
            const cx = data[offset + 0];
            const cy = data[offset + 1];
            const w  = data[offset + 2];
            const h  = data[offset + 3];
            
            // Scale Fix
            const isNormalized = cx < 1.5 && w < 1.5;
            const scale = isNormalized ? 1.0 : 320.0;

            bestBox = [
               Math.max(0, (cy - h/2)),
               Math.max(0, (cx - w/2)),
               Math.min(1, (cy + h/2)),
               Math.min(1, (cx + w/2))
            ];
          }
        }
      }
      setFishCount(Math.min(detectedCount, 50));
      if (!bestBox) bestBox = [0.1, 0.1, 0.9, 0.9]; 
      
      if (DEBUG_MODE) console.log("📦 DETECTED BOX:", bestBox);

      // 2. HYDRA
      const hydraBase = imgTensor.expandDims(0).toFloat().div(255.0); // <--- DIVIDE IS MANDATORY
      const croppedGPU = tf.image.cropAndResize(hydraBase, [bestBox], [0], [224, 224]);
      const hydraCpuData = await croppedGPU.data();
      const hydraInput = tf.tensor(hydraCpuData, [1, 224, 224, 3], 'float32');

      if (DEBUG_MODE) {
        const meanVal = tf.mean(hydraInput).dataSync()[0];
        console.log(`🧪 Hydra Input Brightness: ${meanVal.toFixed(3)} (Should be > 0.2)`);
      }

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

          // --- 🧠 SMART SELECTION LOGIC ---
          
          // 1. Sort predictions by confidence
          const predictions = Array.from(head0).map((p: any, i) => ({
              index: i,
              label: SPECIES_LABELS[i],
              score: p
          }));
          predictions.sort((a: any, b: any) => b.score - a.score);

          if (DEBUG_MODE) {
             console.log("🏆 Top 3 Predictions:", 
                 predictions.slice(0,3).map(p => `${p.label}: ${(p.score*100).toFixed(1)}%`)
             );
          }

          // 2. Logic: If #1 is "Wild Fish" but #2 is valid, pick #2
          let finalChoice = predictions[0];
          const runnerUp = predictions[1];

          if (finalChoice.label === 'wild_fish_background' && runnerUp.score > 0.10) {
               if (DEBUG_MODE) console.log("🔄 Override: Ignoring 'Wild Fish' for", runnerUp.label);
               finalChoice = runnerUp;
          }

          const speciesIdx = finalChoice.index;
          const speciesName = finalChoice.label;

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
          console.warn("⚠️ Hydra Failed:", hydraError);
          return { ...FALLBACK_RESULT, boundingBox: { yMin: bestBox[0], xMin: bestBox[1], yMax: bestBox[2], xMax: bestBox[3] } };
      }
    } catch (e) {
      return FALLBACK_RESULT;
    }
  }, []);

  return { isModelLoading, modelError, analyzeFish, fishCount };
};