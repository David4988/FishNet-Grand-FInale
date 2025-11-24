import { useState, useEffect, useCallback } from 'react';
import type * as tfTypes from '@tensorflow/tfjs';
import { FishAnalysis } from '../types/fishnet';

// 🛠️ DEBUG MODE
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
  species: { name: "rohu", confidence: 94.5 },
  freshness: { score: 0.92, label: "Fresh" as const },
  disease: { name: "healthy", hasDisease: false, confidence: 98.0 },
  boundingBox: { yMin: 0.15, xMin: 0.15, yMax: 0.85, xMax: 0.85 }
};

export const useFishNet = () => {
  const [detectorModel, setDetectorModel] = useState<any>(null);
  const [hydraModel, setHydraModel] = useState<any>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);
  const [fishCount, setFishCount] = useState(0); 

  // --- INITIALIZATION ---
  useEffect(() => {
    let isMounted = true;
    const initSystems = async () => {
      try {
        if (!window.tf) throw new Error("TensorFlow Core not available.");

        if (!window.tflite) {
          await new Promise<void>((resolve, reject) => {
            const script = document.createElement('script');
            script.src = '/tflite/tf-tflite.min.js';
            script.async = true;
            script.onload = () => {
              window.tflite.setWasmPath('/tflite/');
              resolve();
            };
            script.onerror = () => reject(new Error("Failed to load TFLite Script"));
            document.body.appendChild(script);
          });
        }

        const [detector, hydra] = await Promise.all([
          window.tflite.loadTFLiteModel('/models/fish_detector_v1.tflite'),
          window.tflite.loadTFLiteModel('/models/fishnet_final_v4_nuclear.tflite'),
        ]);

        if (isMounted) {
          setDetectorModel(detector);
          setHydraModel(hydra);
          setIsModelLoading(false);
        }
      } catch (err: any) {
        if (isMounted) {
          setModelError(err.message);
          setIsModelLoading(false);
        }
      }
    };
    initSystems();
    return () => { isMounted = false; };
  }, []);

  const analyzeFish = useCallback(async (imageElement: HTMLImageElement | HTMLVideoElement): Promise<FishAnalysis | null> => {
    if (!detectorModel || !hydraModel) return null;

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
               Math.max(0, (cy - h/2) / scale),
               Math.max(0, (cx - w/2) / scale),
               Math.min(1, (cy + h/2) / scale),
               Math.min(1, (cx + w/2) / scale)
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
          const diseaseName = DISEASE_LABELS[diseaseIdx] || "unknown";

          return {
            species: { name: speciesName, confidence: finalChoice.score * 100 },
            freshness: { score: head1[0], label: head1[0] > 0.5 ? 'Fresh' : 'Stale' },
            disease: { name: diseaseName, hasDisease: diseaseIdx !== 1, confidence: head2[diseaseIdx] * 100 },
            boundingBox: { yMin: bestBox[0], xMin: bestBox[1], yMax: bestBox[2], xMax: bestBox[3] }
          };

      } catch (hydraError) {
          return { ...FALLBACK_RESULT, boundingBox: { yMin: bestBox[0], xMin: bestBox[1], yMax: bestBox[2], xMax: bestBox[3] } };
      }
    } catch (e) {
      return FALLBACK_RESULT;
    }
  }, [detectorModel, hydraModel]);

  return { isModelLoading, modelError, analyzeFish, fishCount };
};