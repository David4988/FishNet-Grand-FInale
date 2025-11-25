export interface BoundingBox {
  yMin: number;
  xMin: number;
  yMax: number;
  xMax: number;
}

export interface FishAnalysis {
  species: {
    name: string;
    confidence: number;
  };
  freshness: {
    score: number;
    label: 'Fresh' | 'Stale';
  };
  disease: {
    name: string;
    hasDisease: boolean;
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