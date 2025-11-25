// src/types/fishnet.ts

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  class: string;
  score: number;
}

export interface FishAnalysis {
  species: {
    name: string;
    confidence: number;
  };
  freshness: {
    score: number; // 0.0 to 1.0
    label: 'Fresh' | 'Stale';
  };
  disease: {
    name: string;
    hasDisease: boolean; // True if White Spot or Black Gill > threshold
    confidence: number;
  };
  boundingBox?: BoundingBox;
}

// ✅ THIS INTERFACE MUST BE EXPORTED FOR ANALYZEPAGE TO WORK
export interface UIResult extends FishAnalysis {
  estimatedWeight: number;
  estimatedCount: number;
  marketPrice: number;
  marketTrend: number;
  waterTemp: number;
  phLevel: number;
  autoLength: number;
  manualLength?: string | number;
}