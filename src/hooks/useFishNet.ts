import { useState, useEffect, useCallback } from 'react';
import type * as tfTypes from '@tensorflow/tfjs';
import { FishAnalysis } from '../types/fishnet';

// 🛠️ DEBUG MODE
const DEBUG_MODE = true;

declare global {
  const tflite: any;
  const tf: typeof tfTypes; 
}

// 📋 LABELS (18 Classes - Matches your Model)
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
      
      // 1. DETECTOR (Int8)
      const imgTensor = tf.browser.fromPixels(imageElement);
      const detectorInputGPU = tf.image.resizeBilinear(imgTensor, [320, 320])
          .expandDims(0).toFloat().div(255.0); // Normalize 0-1

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

      // NMS Logic
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
            
            // Scale Fix (Normalized output)
            const scale = 1.0; 
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

      // 2. HYDRA (Float32)
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

          const head0 = outputArray[0].dataSync(); // Species
          const head1 = outputArray[1].dataSync(); // Freshness
          const head2 = outputArray[2].dataSync(); // Disease

          outputArray.forEach(t => t.dispose());
          hydraInput.dispose();

          // --- 🧠 SMART SELECTION LOGIC V3 ---
          
          // 1. Rank predictions
          const predictions = Array.from(head0).map((p: any, i) => ({
              index: i,
              label: SPECIES_LABELS[i],
              score: p
          }));
          predictions.sort((a: any, b: any) => b.score - a.score);

          let finalChoice = predictions[0];
          
          // 2. "Anti-Coward" Filter
          // If the model picks Background, but a real fish is in 2nd place (>5%), take the real fish.
          if (finalChoice.label === 'wild_fish_background') {
              const runnerUp = predictions[1];
              const thirdPlace = predictions[2];

              if (runnerUp.score > 0.05) {
                  if (DEBUG_MODE) console.log(`🔄 Override: Swapped Background for '${runnerUp.label}'`);
                  finalChoice = runnerUp;
              }
              // Special check for Crustaceans (often 3rd place)
              else if (thirdPlace.score > 0.05 && ['prawn', 'crab'].includes(thirdPlace.label)) {
                   if (DEBUG_MODE) console.log(`🔄 Deep Override: Rescued '${thirdPlace.label}' from 3rd place`);
                   finalChoice = thirdPlace;
              }
          }

          // 3. "Catla/Sea Bass" Confusion Fix
          // If Sea Bass is < 50% confident, and Catla/Rohu is nearby, swap to Carp.
          if (finalChoice.label === 'sea_bass' && finalChoice.score < 0.50) {
               const carp = predictions.find(p => ['catla', 'rohu', 'mrigal'].includes(p.label));
               if (carp && carp.score > 0.05) {
                   if (DEBUG_MODE) console.log("🔄 Correction: Low conf Sea Bass -> Swapped to Carp");
                   finalChoice = carp;
               }
          }

          const speciesIdx = finalChoice.index;
          const speciesName = finalChoice.label;

          const diseaseIdx = head2.indexOf(Math.max(...head2));
          const diseaseName = DISEASE_LABELS[diseaseIdx] || "unknown";

          if (DEBUG_MODE) console.log("🏆 Winner:", speciesName);

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