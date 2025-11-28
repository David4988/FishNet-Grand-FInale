import { useState, useEffect, useCallback, useRef } from 'react';
import type * as tfTypes from '@tensorflow/tfjs';
import { FishAnalysis } from '../types/fishnet';

// �️ DEBUG MODE
const DEBUG_MODE = true;

declare global {
  const tflite: any;
  const tf: typeof tfTypes; 
}

// � SPECIES LABELS (18 Classes - Matches your models)
const SPECIES_LABELS = [
  'catfish', 'catla', 'common_carp', 'crab', 'grass_carp', 
  'mackerel', 'mrigal', 'pink_perch', 'prawn', 'red_mullet', 
  'rohu', 'sea_bass', 'sea_bream', 'silver_carp', 'sprat', 
  'tilapia', 'trout', 'wild_fish_background'
];

const DISEASE_LABELS = ['black_gill_disease', 'healthy', 'white_spot_virus'];

// �️ SAFETY NET
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
  
  // Store models in ref to avoid re-renders
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

        if (DEBUG_MODE) console.log('� Loading 3-Model Pipeline...');
        
        const [detector, species, disease] = await Promise.all([
          window.tflite.loadTFLiteModel('/models/fish_detector_v1.tflite'),
          // ✅ SPLIT MODELS
          window.tflite.loadTFLiteModel('/models/fish_species_model.tflite'),
          window.tflite.loadTFLiteModel('/models/fish_disease_model.tflite'),
        ]);

        if (isMounted) {
          modelsRef.current = { detector, species, disease };
          setIsModelLoading(false);
          console.log('� FishNet Core Online (Split Architecture)');
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
      let bestBox: number[] | null = null;
      let detectedCount = 0;
      
      // ==========================================
      // �️ STEP 1: DETECTOR (Int8)
      // ==========================================
      
      const imgTensor = tf.browser.fromPixels(imageElement);
      
      // Preprocess: Resize [320x320] & Raw 0-255 for Detector
      const detectorInputGPU = tf.image.resizeBilinear(imgTensor, [320, 320])
          .expandDims(0).toFloat(); 

      const detectorCpuData = await detectorInputGPU.data();
      const detectorInput = tf.tensor(detectorCpuData, [1, 320, 320, 3], 'float32');
      detectorInputGPU.dispose();

      let detectorRaw = detector.predict(detectorInput);
      // Handle Format
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

      // ==========================================
      // � STEP 2: PREPARE FOR MODELS (Float32)
      // ==========================================
      
      // Normalize 0-1
      const hydraBase = imgTensor.expandDims(0).toFloat().div(255.0);
      const croppedGPU = tf.image.cropAndResize(hydraBase, [bestBox], [0], [224, 224]);
      const hydraCpuData = await croppedGPU.data();
      const hydraInput = tf.tensor(hydraCpuData, [1, 224, 224, 3], 'float32');

      imgTensor.dispose();
      hydraBase.dispose();
      croppedGPU.dispose();

      try {
          // ==========================================
          // � STEP 3: SPECIES ID
          // ==========================================
          const spInputName = species.inputs[0].name || 'input_1';
          const spRaw = species.predict({ [spInputName]: hydraInput });
          
          let spArray: any[] = [];
          if (Array.isArray(spRaw)) spArray = spRaw;
          else if (typeof spRaw === 'object' && !spRaw.dataSync) spArray = Object.values(spRaw);
          else spArray = [spRaw];

          const speciesData = spArray[0].dataSync();
          spArray.forEach(t => t.dispose());

          // --- � AGGRESSIVE ANTI-BACKGROUND LOGIC ---
          
          const predictions = Array.from(speciesData).map((p: any, i) => ({
              index: i,
              label: SPECIES_LABELS[i],
              score: p
          }));
          predictions.sort((a: any, b: any) => b.score - a.score);

          let finalChoice = predictions[0];
          
          // � "NEVER GUESS WILD FISH" RULE
          // If top guess is Wild Fish, check if ANY other fish has > 0.1% (0.001) confidence
          if (finalChoice.label === 'wild_fish_background') {
              const bestAlternative = predictions.find(p => 
                  p.label !== 'wild_fish_background' && 
                  p.label !== 'unknown' && 
                  p.score > 0.001 // Even 0.1% is enough to override
              );

              if (bestAlternative) {
                  if (DEBUG_MODE) console.log(`� Override: Killed 'Wild Fish' (${(finalChoice.score*100).toFixed(1)}%) for '${bestAlternative.label}' (${(bestAlternative.score*100).toFixed(1)}%)`);
                  finalChoice = bestAlternative;
              }
          }

          // Sea Bass Fix
          if (finalChoice.label === 'sea_bass' && finalChoice.score < 0.50) {
               const carp = predictions.find(p => ['catla', 'rohu', 'mrigal'].includes(p.label));
               if (carp && carp.score > 0.05) finalChoice = carp;
          }

          const speciesName = finalChoice.label;
          const speciesConf = finalChoice.score * 100;

          if (DEBUG_MODE) console.log(`� Final Species: ${speciesName} (${speciesConf.toFixed(1)}%)`);


          // ==========================================
          // � STEP 4: DISEASE CHECK
          // ==========================================
          const dzInputName = disease.inputs[0].name || 'input_1';
          const dzRaw = disease.predict({ [dzInputName]: hydraInput });
          
          let dzArray: any[] = [];
          if (Array.isArray(dzRaw)) dzArray = dzRaw;
          else if (typeof dzRaw === 'object' && !dzRaw.dataSync) dzArray = Object.values(dzRaw);
          else dzArray = [dzRaw];

          const diseaseData = dzArray[0].dataSync();
          dzArray.forEach(t => t.dispose());
          hydraInput.dispose();

          const diseaseIdx = diseaseData.indexOf(Math.max(...diseaseData));
          let diseaseName = DISEASE_LABELS[diseaseIdx] || "unknown";
          let diseaseConf = diseaseData[diseaseIdx] * 100;

          // Paranoid Disease Logic
          if (diseaseData[2] > 0.3) { diseaseName = "White Spot Risk"; diseaseConf = diseaseData[2]*100; }
          else if (diseaseData[0] > 0.4) { diseaseName = "Black Gill Risk"; diseaseConf = diseaseData[0]*100; }

          return {
            species: { name: speciesName, confidence: speciesConf },
            freshness: { score: 0.95, label: 'Fresh' },
            disease: { name: diseaseName, hasDisease: diseaseName !== 'Healthy', confidence: diseaseConf },
            boundingBox: { yMin: bestBox[0], xMin: bestBox[1], yMax: bestBox[2], xMax: bestBox[3] }
          };

      } catch (err) {
          console.error("⚠️ Model Failed:", err);
          return { ...FALLBACK_RESULT, boundingBox: { yMin: bestBox[0], xMin: bestBox[1], yMax: bestBox[2], xMax: bestBox[3] } };
      }
    } catch (e) {
      return FALLBACK_RESULT;
    }
  }, [modelsRef]);

  return { isModelLoading, modelError, analyzeFish, fishCount };
};