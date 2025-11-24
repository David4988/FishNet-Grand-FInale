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
  // The Real Detection Box (Normalized 0-1)
  boundingBox?: BoundingBox;
}

// Extended Interface for the "Pro" Dashboard UI
export interface UIResult extends FishAnalysis {
  estimatedWeight: number;
  estimatedCount: number;
  
  // Rich Data Fields
  marketPrice: number;
  marketTrend: number;
  waterTemp: number;
  phLevel: number;
  
  // Biometrics
  autoLength: number;
  manualLength?: string | number;
}