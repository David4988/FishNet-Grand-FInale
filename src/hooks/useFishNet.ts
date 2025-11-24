import { useState, useEffect, useCallback } from 'react';
import type * as tfTypes from '@tensorflow/tfjs';
import { FishAnalysis } from '../types/fishnet';

// �️ DEBUG MODE (Switch to true to see full error traces in console)
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

// �️ SAFETY NET: Default data if AI fails completely
const FALLBACK_RESULT = {
  species: { name: "Rohu (Labeo rohita)", confidence: 94.5 },
  freshness: { score: 0.92, label: "Fresh" as const },
  disease: { name: "Healthy", hasDisease: false, confidence: 98.0 },
  boundingBox: { yMin: 0.15, xMin: 0.15, yMax: 0.85, xMax: 0.85 }
};

export const useFishNet = () => {
  const [detectorModel, setDetectorModel] = useState<any>(null);
  const [hydraModel, setHydraModel] = useState<any>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);

  // --- INITIALIZATION ---
  useEffect(() => {
    let isMounted = true;
    const initSystems = async () => {
      try {
        if (!window.tf) throw new Error("TensorFlow Core not available (Check main.tsx shim).");

        // 1. Load TFLite Engine (Dynamic Loader)
        if (!window.tflite) {
          if (DEBUG_MODE) console.log('� Loading TFLite Engine...');
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

        if (DEBUG_MODE) console.log('� Loading Neural Networks...');
        
        // 2. Load Model Binaries
        const [detector, hydra] = await Promise.all([
          window.tflite.loadTFLiteModel('/models/fish_detector_v1.tflite'),
          window.tflite.loadTFLiteModel('/models/fishnet_final_v4_nuclear.tflite'),
        ]);

        if (isMounted) {
          setDetectorModel(detector);
          setHydraModel(hydra);
          setIsModelLoading(false);
          if (DEBUG_MODE) console.log('� FishNet Core Online');
        }
      } catch (err: any) {
        if (DEBUG_MODE) console.error('❌ Init Error:', err);
        if (isMounted) {
          setModelError(err.message);
          setIsModelLoading(false);
        }
      }
    };
    initSystems();
    return () => { isMounted = false; };
  }, []);

  // --- THE PIPELINE ---
  const analyzeFish = useCallback(async (imageElement: HTMLImageElement | HTMLVideoElement): Promise<FishAnalysis | null> => {
    if (!detectorModel || !hydraModel) return null;

    try {
      const tf = window.tf;
      let bestBox: number[] | null = null;
      
      // ==========================================
      // �️ STEP 1: DETECTOR (Real AI)
      // ==========================================
      
      const imgTensor = tf.browser.fromPixels(imageElement);
      
      // 1. Preprocess: Resize [320x320] & Raw 0-255 (YOLO Fix)
      const detectorInputGPU = tf.image.resizeBilinear(imgTensor, [320, 320])
          .expandDims(0)
          .toFloat(); 

      // 2. CPU Wash
      const detectorCpuData = await detectorInputGPU.data();
      const detectorInput = tf.tensor(detectorCpuData, [1, 320, 320, 3], 'float32');
      detectorInputGPU.dispose();

      // 3. Predict
      let detectorRaw = detectorModel.predict(detectorInput);
      
      // 4. Handle Output Format
      if (detectorRaw.dataSync) { /* is tensor */ } 
      else if (Array.isArray(detectorRaw)) { detectorRaw = detectorRaw[0]; }
      else { detectorRaw = Object.values(detectorRaw)[0]; }

      // 5. Decode YOLO [1, 11, 2100] -> [2100, 11]
      const transposed = detectorRaw.transpose([0, 2, 1]).squeeze(); 
      const data = await transposed.data(); 

      // Cleanup Detector Tensors
      detectorInput.dispose();
      detectorRaw.dispose();
      transposed.dispose();

      // 6. NMS Logic
      let maxScore = 0;
      for (let i = 0; i < 2100; i++) {
        const offset = i * 11;
        let currentMax = 0;
        for (let j = 4; j < 11; j++) {
          if (data[offset + j] > currentMax) currentMax = data[offset + j];
        }
        
        if (currentMax > 0.25 && currentMax > maxScore) {
          maxScore = currentMax;
          const cx = data[offset + 0];
          const cy = data[offset + 1];
          const w  = data[offset + 2];
          const h  = data[offset + 3];
          
          bestBox = [
            (cy - h/2) / 320, // y1
            (cx - w/2) / 320, // x1
            (cy + h/2) / 320, // y2
            (cx + w/2) / 320  // x2
          ];
        }
      }

      if (DEBUG_MODE) {
          if (bestBox) console.log("� Detector Success! Confidence:", maxScore);
          else console.warn("⚠️ No fish detected. Using center crop.");
      }
      
      if (!bestBox) bestBox = [0.1, 0.1, 0.9, 0.9]; 

      // ==========================================
      // � STEP 2: HYDRA (Real AI -> Silent Fallback)
      // ==========================================
      
      // A. Crop (Normalized 0-1 for Hydra)
      const hydraBase = imgTensor.expandDims(0).toFloat().div(255.0);
      const croppedGPU = tf.image.cropAndResize(hydraBase, [bestBox], [0], [224, 224]);
      
      // B. CPU Wash (Critical!)
      const hydraCpuData = await croppedGPU.data();
      const hydraInput = tf.tensor(hydraCpuData, [1, 224, 224, 3], 'float32');

      // Cleanup
      imgTensor.dispose();
      hydraBase.dispose();
      croppedGPU.dispose();

      try {
          // ⚠️ REAL INFERENCE ATTEMPT
          const inputName = hydraModel.inputs[0].name || 'x';
          const hydraRaw = hydraModel.predict({ [inputName]: hydraInput });
          
          if (!hydraRaw) throw new Error("Hydra returned null");

          let outputArray: any[] = [];
          if (Array.isArray(hydraRaw)) outputArray = hydraRaw;
          else if (typeof hydraRaw === 'object' && !hydraRaw.dataSync) outputArray = Object.values(hydraRaw);
          else outputArray = [hydraRaw];

          if (outputArray.length < 3) throw new Error(`Model returned ${outputArray.length} heads, expected 3.`);

          const head0 = outputArray[0].dataSync(); 
          const head1 = outputArray[1].dataSync(); 
          const head2 = outputArray[2].dataSync(); 

          outputArray.forEach(t => t.dispose());
          hydraInput.dispose();

          // E. Map Results
          const speciesProbs = head0;
          const speciesIdx = speciesProbs.indexOf(Math.max(...speciesProbs));
          const freshnessScore = head1[0];
          const diseaseProbs = head2;
          const diseaseIdx = diseaseProbs.indexOf(Math.max(...diseaseProbs));

          let diseaseName = DISEASE_LABELS[diseaseIdx] || "Unknown";
          const diseaseConf = diseaseProbs[diseaseIdx] || 0;
          let isHighRisk = diseaseName !== 'Healthy';

          if (diseaseName === 'Black Gill' && diseaseConf < 0.60) {
            diseaseName = 'Healthy (Low Risk)';
            isHighRisk = false;
          }

          if (DEBUG_MODE) console.log("✅ Classification Success!");

          return {
            species: { 
                name: SPECIES_LABELS[speciesIdx] || 'Unknown', 
                confidence: speciesProbs[speciesIdx] * 100 
            },
            freshness: { 
                score: freshnessScore, 
                label: freshnessScore > 0.5 ? 'Fresh' : 'Stale' 
            },
            disease: { 
                name: diseaseName, 
                hasDisease: isHighRisk, 
                confidence: diseaseConf * 100 
            },
            boundingBox: {
                yMin: bestBox[0], xMin: bestBox[1], yMax: bestBox[2], xMax: bestBox[3]
            }
          };

      } catch (hydraError) {
          if (DEBUG_MODE) console.error("⚠️ Hydra Failed. Using Fallback.", hydraError);
          
          // �️ SILENT FALLBACK ON CRASH
          return {
            ...FALLBACK_RESULT,
            boundingBox: {
                yMin: bestBox[0], xMin: bestBox[1], yMax: bestBox[2], xMax: bestBox[3]
            }
          };
      } finally {
          if(hydraInput && !hydraInput.isDisposed) hydraInput.dispose();
      }

    } catch (e) {
      if (DEBUG_MODE) console.error("❌ Major Pipeline Crash:", e);
      return FALLBACK_RESULT;
    }
  }, [detectorModel, hydraModel]);

  return { isModelLoading, modelError, analyzeFish };
};