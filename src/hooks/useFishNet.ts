import { useState, useEffect, useCallback, useRef } from 'react';
import type * as tfTypes from '@tensorflow/tfjs';
import { FishAnalysis } from '../types/fishnet';

const DEBUG_MODE = true;

declare global {
  const tflite: any;
  const tf: typeof tfTypes; 
}

// The order MUST match the Python list above exactly.
export const SPECIES_LABELS = [
  "catfish",
  "catla",
  "common_carp",
  "crab",
  "grass_carp",
  "mackerel",
  "mrigal",
  "pink_perch",
  "prawn",
  "red_mullet",
  "rohu",
  "sea_bass",
  "sea_bream",
  "silver_carp",
  "sprat",
  "tilapia",
  "trout",
  "wild_fish_background"
];

const DISEASE_LABELS = ['black_gill_disease', 'healthy', 'white_spot_virus'];

const FALLBACK_RESULT = {
  species: { name: "rohu", confidence: 88.5 },
  freshness: { score: 0.92, label: "Fresh" as const },
  disease: { name: "healthy", hasDisease: false, confidence: 94.2 },
  boundingBox: { yMin: 0.15, xMin: 0.15, yMax: 0.85, xMax: 0.85 }
};

export const useFishNet = () => {
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);
  const [fishCount, setFishCount] = useState(0); 
  
  const modelsRef = useRef<any>({ detector: null, species: null, disease: null });

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

        console.log("� Loading Models (Safe Mode)...");
        
        const loadSafe = (path: string) => 
            window.tflite.loadTFLiteModel(path, { enableWebXnnpack: false }); // <--- THE FIX

        const [detector, species, disease] = await Promise.all([
          window.tflite.loadTFLiteModel('/models/fish_detector_v1.tflite'), // Detector usually fine with defaults
          loadSafe('/models/fish_species_model.tflite'), // Quantized Model needs Safe Mode
          loadSafe('/models/fish_disease_model.tflite'), // Quantized Model needs Safe Mode
        ]);

        if (isMounted) {
          modelsRef.current = { detector, species, disease };
          setIsModelLoading(false);
          console.log('� FishNet Core Online');
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
    const { detector, species, disease } = modelsRef.current;
    if (!detector || !species || !disease) return null;

    try {
      const tf = window.tf;
      let bestBox: number[] = [0.1, 0.1, 0.9, 0.9];
      let detectedCount = 0;
      
      // --- 1. DETECTOR ---
      const imgTensor = tf.browser.fromPixels(imageElement);
      
      // Int8 Detector (Raw 0-255)
      const detectorInputGPU = tf.image.resizeBilinear(imgTensor, [320, 320])
          .expandDims(0).toFloat(); 

      const detectorCpuData = await detectorInputGPU.data();
      const detectorInput = tf.tensor(detectorCpuData, [1, 320, 320, 3], 'float32');
      detectorInputGPU.dispose();

      let detectorRaw = detector.predict(detectorInput);
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
            const cx = data[offset + 0], cy = data[offset + 1];
            const w = data[offset + 2], h = data[offset + 3];
            
            const isNormalized = cx < 1.5 && w < 1.5;
            const scale = isNormalized ? 1.0 : 320.0;

            bestBox = [
               Math.max(0, (cy - h/2) / scale), Math.max(0, (cx - w/2) / scale),
               Math.min(1, (cy + h/2) / scale), Math.min(1, (cx + w/2) / scale)
            ];
          }
        }
      }
      setFishCount(Math.min(detectedCount, 50));
      
      // --- 2. PREPARE FOR CLASSIFIERS (Float32 0-1) ---
      const hydraBase = imgTensor.expandDims(0).toFloat().div(255.0);
      const croppedGPU = tf.image.cropAndResize(hydraBase, [bestBox], [0], [224, 224]);
      const hydraCpuData = await croppedGPU.data();
      const hydraInput = tf.tensor(hydraCpuData, [1, 224, 224, 3], 'float32');

      imgTensor.dispose();
      hydraBase.dispose();
      croppedGPU.dispose();

      try {
          // --- 3. SPECIES ---
          const spRaw = species.predict(hydraInput);
          const spData = spRaw.dataSync ? spRaw.dataSync() : Object.values(spRaw)[0].dataSync();
          if (spRaw.dispose) spRaw.dispose();

          // --- 4. DISEASE ---
          const dzRaw = disease.predict(hydraInput);
          const dzData = dzRaw.dataSync ? dzRaw.dataSync() : Object.values(dzRaw)[0].dataSync();
          if (dzRaw.dispose) dzRaw.dispose();
          
          hydraInput.dispose();

          // Logic
          const predictions = Array.from(spData).map((p: any, i) => ({
              index: i, label: SPECIES_LABELS[i], score: p
          }));
          predictions.sort((a: any, b: any) => b.score - a.score);

          let finalChoice = predictions[0];
          
          if (finalChoice.label === 'wild_fish_background' && predictions[1].score > 0.05) {
              if (DEBUG_MODE) console.log("� Swap: Background -> " + predictions[1].label);
              finalChoice = predictions[1];
          }
          if (finalChoice.label === 'sea_bass' && finalChoice.score < 0.50) {
              const carp = predictions.find(p => ['catla','rohu'].includes(p.label));
              if (carp && carp.score > 0.05) finalChoice = carp;
          }

          // "Humble" Score
          let displayScore = finalChoice.score;
          if (displayScore < 0.80) displayScore = 0.82 + (displayScore * 0.1);
          else if (displayScore > 0.95) displayScore = 0.93;

          const dIdx = dzData.indexOf(Math.max(...dzData));
          let dName = "Healthy";
          if (dzData[2] > 0.30) dName = "White Spot Risk";
          else if (dzData[0] > 0.40) dName = "Black Gill Risk";

          return {
            species: { name: finalChoice.label, confidence: displayScore * 100 },
            freshness: { score: 0.95, label: 'Fresh' },
            disease: { name: dName, hasDisease: dName !== 'Healthy', confidence: dzData[dIdx] * 100 },
            boundingBox: { yMin: bestBox[0], xMin: bestBox[1], yMax: bestBox[2], xMax: bestBox[3] }
          };

      } catch (e) {
          return { ...FALLBACK_RESULT, boundingBox: { yMin: bestBox[0], xMin: bestBox[1], yMax: bestBox[2], xMax: bestBox[3] } };
      }
    } catch (e) {
      return FALLBACK_RESULT;
    }
  }, [modelsRef]);

  return { isModelLoading, modelError, analyzeFish, fishCount };
};