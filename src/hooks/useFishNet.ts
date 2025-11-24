import { useState, useEffect, useRef, useCallback } from 'react';

// � GLOBAL ACCESS: We use the scripts loaded in index.html
declare global {
  interface Window {
    tflite: any;
    tf: any;
  }
}

// � HARDCODED LABELS (No Fetch Required)
// These match your 'fishnet_final_v4_nuclear' model exactly.
const SPECIES_LABELS = [
  'barramundi', 'catfish', 'catla', 'crab', 'healthy', 
  'mackerel', 'mrigal', 'prawn', 'red_mullet', 'rohu', 
  'sardine', 'sea_bream', 'tilapia', 'trout', 'wild_fish_background'
];

const DISEASE_LABELS = [
  'black_gill_disease', 'healthy', 'white_spot_virus'
];

export interface AnalysisResult {
  species: { name: string; confidence: number; };
  freshness: { score: number; label: 'Fresh' | 'Stale'; };
  disease: { name: string; confidence: number; };
  boundingBox?: { xMin: number; yMin: number; xMax: number; yMax: number; };
}

export const useFishNet = () => {
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);
  
  const modelsRef = useRef<any>({ yolo: null, hydra: null });

  // --- 1. INITIALIZATION ---
  useEffect(() => {
    const loadResources = async () => {
      try {
        setIsModelLoading(true);
        console.log("hz: � Initializing Offline AI...");

        // A. Wait for Scripts to Load from index.html
        let retries = 0;
        while ((!window.tflite || !window.tf) && retries < 50) {
            await new Promise(r => setTimeout(r, 100));
            retries++;
        }
        
        if (!window.tf) throw new Error("Local tf.min.js failed to load.");
        if (!window.tflite) throw new Error("Local tf-tflite.min.js failed to load.");
        
        const tflite = window.tflite;

        // B. Set WASM Path (Local Folder)
        tflite.setWasmPath('/tflite/');

        // C. Load Models
        console.log("hz: � Loading Models...");
        const [yolo, hydra] = await Promise.all([
          tflite.loadTFLiteModel('/models/fish_detector_v1.tflite'),
          tflite.loadTFLiteModel('/models/fishnet_final_v4_nuclear.tflite') 
        ]);

        modelsRef.current = { yolo, hydra };
        console.log("hz: ✅ System Online. Offline Mode Active.");
        setIsModelLoading(false);

      } catch (err: any) {
        console.error("hz: ❌ Init Failed:", err);
        setModelError(`Offline Load Error: ${err.message}`);
        setIsModelLoading(false);
      }
    };

    loadResources();
  }, []);

  // --- 2. INFERENCE ENGINE ---
  const analyzeFish = useCallback(async (imageElement: HTMLImageElement): Promise<AnalysisResult | null> => {
    if (!modelsRef.current.yolo || !modelsRef.current.hydra) return null;
    
    const tf = window.tf; 

    try {
      // --- STEP A: DETECTION (YOLO) ---
      const yoloInput = tf.tidy(() => {
        return tf.browser.fromPixels(imageElement)
          .resizeBilinear([320, 320])
          .expandDims(0)
          .div(255.0);
      });

      // Run YOLO (Optional: Parse boxes here if needed)
      let yoloResult = modelsRef.current.yolo.predict(yoloInput);
      
      // --- STEP B: ANALYSIS (HYDRA) ---
      const hydraInput = tf.tidy(() => {
        return tf.browser.fromPixels(imageElement)
          .resizeBilinear([224, 224])
          .expandDims(0)
          .div(255.0);
      });

      const hydraResult = modelsRef.current.hydra.predict(hydraInput);
      
      // --- STEP C: DECODE RESULTS ---
      // Robust Decoder for Array or Object outputs
      let speciesTensor, freshnessTensor, diseaseTensor;
      const results = Array.isArray(hydraResult) ? hydraResult : Object.values(hydraResult);
      
      speciesTensor = results.find((t: any) => t.shape[1] === SPECIES_LABELS.length);
      diseaseTensor = results.find((t: any) => t.shape[1] === DISEASE_LABELS.length);
      freshnessTensor = results.find((t: any) => t.shape[1] === 1);

      if (!speciesTensor || !diseaseTensor) {
          throw new Error("Hydra output shape mismatch");
      }

      // Download Data (Sync)
      const speciesData = await speciesTensor.data();
      const diseaseData = await diseaseTensor.data();
      const freshnessData = await freshnessTensor.data();

      // Find Max Indices
      const speciesIdx = speciesData.indexOf(Math.max(...speciesData));
      const diseaseIdx = diseaseData.indexOf(Math.max(...diseaseData));
      
      const speciesName = SPECIES_LABELS[speciesIdx];
      const diseaseName = DISEASE_LABELS[diseaseIdx];
      const freshnessScore = freshnessData[0];

      // CLEANUP
      yoloInput.dispose();
      hydraInput.dispose();
      if (yoloResult.dispose) yoloResult.dispose();
      results.forEach((t: any) => t.dispose());

      return {
        species: { 
            name: speciesName, 
            confidence: speciesData[speciesIdx] * 100 
        },
        freshness: { 
            score: freshnessScore, 
            label: freshnessScore > 0.5 ? 'Fresh' : 'Stale' 
        },
        disease: { 
            name: diseaseName, 
            confidence: diseaseData[diseaseIdx] * 100 
        },
        boundingBox: { xMin: 0.1, yMin: 0.1, xMax: 0.9, yMax: 0.9 } 
      };

    } catch (err) {
      console.error("hz: ❌ Inference Error:", err);
      return null;
    }
  }, []);

  return { analyzeFish, isModelLoading, modelError };
};