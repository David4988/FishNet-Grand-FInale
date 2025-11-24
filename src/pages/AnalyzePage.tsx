import { useEffect, useRef, useState } from "react";
import {
  Camera,
  Upload,
  Ruler,
  Check,
  Share2,
  MapPin,
  Calendar,
  Info,
  ArrowLeft,
  Thermometer,
  Droplets,
  TrendingUp,
  DollarSign,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CameraCapture } from "@/components/analyze/CameraCapture";
import { ExplainabilityOverlay } from "@/components/analyze/ExplainabilityOverlay";
import { CalibrationHelper } from "@/components/analyze/CalibrationHelper";
import { databaseService } from "@/services/database";
import { toast } from "@/components/ui/use-toast";
import { addLocalCatch } from "@/utils/localCatches";
import { useFishNet } from "@/hooks/useFishNet";
import { BoundingBox, UIResult } from "@/types/fishnet"; // Import UIResult

// --- INTELLIGENCE LAYER ---
// Robust DB Keys (Must contain both full scientific names and short names)
const SPECIES_DB: Record<string, { key: string; price: number; trend: number; weightFactor: number }> = {
  'Rohu (Labeo rohita)': { key: "rohu", price: 160, trend: 4.5, weightFactor: 2.8 },
  'Catla (Catla catla)': { key: "catla", price: 180, trend: 2.1, weightFactor: 3.2 },
  'Barramundi': { key: "barramundi", price: 450, trend: 8.4, weightFactor: 3.5 },
  'Tilapia': { key: "tilapia", price: 120, trend: -1.2, weightFactor: 1.5 },
  'Salmon': { key: "salmon", price: 850, trend: 12.0, weightFactor: 4.0 },
  'Unknown': { key: "unknown", price: 100, trend: 0.0, weightFactor: 1.0 },
  // Short Name Fallbacks (for robust lookup)
  'Rohu': { key: "rohu", price: 160, trend: 4.5, weightFactor: 2.8 },
  'Catla': { key: "catla", price: 180, trend: 2.1, weightFactor: 3.2 },
};

const calculateBioMetrics = (box: BoundingBox | undefined) => {
  if (!box) return { length: 0, weight: 0 };
  const widthPercent = box.xMax - box.xMin;
  const heightPercent = box.yMax - box.yMin;
  const diagonal = Math.sqrt(
    Math.pow(widthPercent, 2) + Math.pow(heightPercent, 2)
  );
  let estimatedLengthCm = diagonal * 50;
  estimatedLengthCm = Math.max(10, Math.min(120, estimatedLengthCm));
  const estimatedWeightKg = Math.pow(estimatedLengthCm / 10, 3) / 25;
  return {
    length: parseFloat(estimatedLengthCm.toFixed(1)),
    weight: parseFloat(estimatedWeightKg.toFixed(2)),
  };
};

export default function AnalyzePage() {
  const { t } = useTranslation();
  const { analyzeFish, isModelLoading, modelError } = useFishNet();

  const [showCamera, setShowCamera] = useState(false);
  const [showCalibration, setShowCalibration] = useState(false);
  const [imageData, setImageData] = useState<string | null>(null);
  const [result, setResult] = useState<UIResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [location, setLocation] = useState<
    { latitude: number; longitude: number } | undefined
  >(undefined);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [measuredLength, setMeasuredLength] = useState<string | number>("-");

  useEffect(() => {
    if (modelError)
      toast({
        variant: "destructive",
        title: t("analyze.error"),
        description: modelError,
      });
  }, [modelError, t]);

  const analyzeImage = async (dataUrl: string) => {
    if (isModelLoading) {
      toast({
        title: t("analyze.systemWarming"),
        description: t("analyze.loadingAI"),
      });
      return;
    }
    setIsAnalyzing(true);

    setTimeout(async () => {
      try {
        const img = new Image();
        img.src = dataUrl;
        await img.decode();
        const analysis = await analyzeFish(img);

        if (analysis) {
          setImageData(dataUrl);

          // ⚠️ FIX: Use the full name from the hook output for the lookup
          const fullName = analysis.species.name; 
          const shortName = fullName.split('(')[0].trim();
          
          // Try full name, then short name, then Unknown
          const dbEntry = SPECIES_DB[fullName] || 
                          SPECIES_DB[shortName] || 
                          SPECIES_DB["Unknown"];

          const bio = calculateBioMetrics(analysis.boundingBox);

          setResult({
            // Store the translation KEY
            species: dbEntry.key, 
            confidence: analysis.species.confidence,
            healthScore: analysis.freshness.score * 100,
            disease: analysis.disease.name,
            estimatedWeight: bio.weight * dbEntry.weightFactor,
            estimatedCount: 1,
            boundingBox: analysis.boundingBox,
            marketPrice: dbEntry.price,
            marketTrend: dbEntry.trend,
            waterTemp: 26 + Math.random() * 2,
            phLevel: 7.0 + Math.random() * 0.5,
            autoLength: bio.length,
          });

          if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition((pos) => {
              setLocation({
                latitude: pos.coords.latitude,
                longitude: pos.coords.longitude,
              });
            });
          }
        } else {
          // This path is hit if useFishNet returns null (only if models aren't ready)
          throw new Error("No analysis returned"); 
        }
      } catch (e) {
        console.error("Analysis Failed", e);
        toast({
          variant: "destructive",
          title: t("analyze.analysisFailed"),
          description: t("analyze.analysisFailedDesc"),
        });
      } finally {
        setIsAnalyzing(false);
      }
    }, 500);
  };

  const handleSave = async () => {
    if (!imageData || !result) return;
    try {
      await databaseService.initialize?.();
      addLocalCatch({
        id: "local-" + Date.now(),
        createdAt: Date.now(),
        species: result.species || "Unknown",
        image: imageData,
        lat: location?.latitude ?? 0,
        lng: location?.longitude ?? 0,
        healthScore: result.healthScore,
        confidence: result.confidence,
      });
      await databaseService.addCatch({
        species: result.species,
        confidence: result.confidence,
        health_score: result.healthScore,
        estimated_weight: result.estimatedWeight,
        count: result.estimatedCount,
        timestamp: new Date().toISOString(),
        latitude: location?.latitude ?? 0,
        longitude: location?.longitude ?? 0,
        image_data: imageData,
        is_synced: false,
      });
      toast({
        title: t("analyze.saved"),
        description: t("analyze.catchSaved"),
      });
    } catch (e) {
      console.error(e);
      toast({
        title: t("analyze.saveFailed"),
        description: t("analyze.couldNotSave"),
      });
    }
  };

  if (showCamera)
    return (
      <CameraCapture
        onImageCapture={(data) => {
          setShowCamera(false);
          analyzeImage(data);
        }}
        onClose={() => setShowCamera(false)}
      />
    );
  if (showCalibration && imageData)
    return (
      <CalibrationHelper
        imageData={imageData}
        onCalibrated={(_, len) => {
          setMeasuredLength(len?.toFixed(1) || "-");
          setShowCalibration(false);
        }}
        onClose={() => setShowCalibration(false)}
      />
    );

  // --- DASHBOARD (MOBILE FIXED) ---
  if (imageData && result) {
    return (
      // ✅ FIX: Use 100dvh for dynamic mobile height
      <div className="min-h-[100dvh] bg-slate-950 text-white font-sans flex flex-col">
        <div className="max-w-7xl mx-auto w-full flex-1 grid grid-cols-1 lg:grid-cols-2 gap-0 lg:gap-8 lg:p-8">
          
          {/* COLUMN 1: VISUALS */}
          <div className="relative w-full h-[45vh] lg:h-full lg:rounded-3xl overflow-hidden bg-black shadow-2xl shrink-0">
            <div className="absolute top-0 left-0 right-0 z-20 p-4 flex justify-between items-center bg-gradient-to-b from-black/90 to-transparent">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => {
                  setImageData(null);
                  setResult(null);
                }}
                className="text-white bg-black/20 hover:bg-black/40 backdrop-blur-md rounded-full"
              >
                <ArrowLeft className="w-6 h-6" />
              </Button>
              <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-black/40 backdrop-blur-md border border-white/10">
                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="text-xs font-mono font-bold text-emerald-400 tracking-wider">
                  LIVE FEED
                </span>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="text-white bg-black/20 hover:bg-black/40 backdrop-blur-md rounded-full"
              >
                <Share2 className="w-5 h-5" />
              </Button>
            </div>

            <ExplainabilityOverlay
              imageData={imageData}
              species={t(`species.${result.species}`)} // Pass translated name
              confidence={result.confidence}
              boundingBox={result.boundingBox}
              className="h-full w-full object-cover"
            />
          </div>

          {/* COLUMN 2: DATA SHEET */}
          <div className="relative flex-1 bg-slate-900 lg:bg-transparent flex flex-col overflow-hidden -mt-6 lg:mt-0 rounded-t-3xl lg:rounded-none shadow-[0_-10px_40px_rgba(0,0,0,0.5)]">
            
            {/* Scrollable Content */}
            <div className="flex-1 overflow-y-auto no-scrollbar p-6 pb-32 lg:p-0">
              <div className="w-12 h-1.5 bg-slate-700/50 rounded-full mx-auto mb-6 lg:hidden" />

              <div className="lg:bg-slate-900/50 lg:backdrop-blur-xl lg:border lg:border-white/5 lg:p-8 lg:rounded-3xl lg:h-full lg:overflow-y-auto">
                {/* HEADER */}
                <div className="flex justify-between items-start mb-8">
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <Badge
                        variant="outline"
                        className="border-cyan-500/30 text-cyan-400 bg-cyan-500/5 uppercase tracking-widest text-[10px]"
                      >
                        {t("analyze.identifiedSpecies")}
                      </Badge>
                      <Badge
                        variant="outline"
                        className="border-white/10 text-slate-400 uppercase tracking-widest text-[10px]"
                      >
                        {result.confidence.toFixed(1)}% {t("analyze.match")}
                      </Badge>
                    </div>
                    {/* ✅ FIX: break-words to prevent long names breaking layout */}
                    <h1 className="text-3xl lg:text-4xl font-bold text-white text-glow leading-tight mb-1 break-words">
                      {t(`species.${result.species}`)}
                    </h1>
                  </div>
                  <div className="flex flex-col items-center justify-center w-16 h-16 rounded-2xl bg-emerald-950/50 border border-emerald-500/30 shrink-0">
                    <span className="text-2xl font-bold text-emerald-400">
                      {Math.round(result.healthScore)}
                    </span>
                    <span className="text-[10px] text-emerald-600 uppercase font-bold">
                      {t("analyze.health")}
                    </span>
                  </div>
                </div>

                {/* BIOMETRICS */}
                <h3 className="text-slate-500 text-xs font-bold uppercase tracking-widest mb-4 flex items-center gap-2">
                  <Ruler className="w-3 h-3" /> {t("analyze.biometrics")}
                </h3>
                <div className="grid grid-cols-2 gap-4 mb-8">
                  <div className="bg-black/20 p-4 rounded-2xl border border-white/5">
                    <div className="text-slate-400 text-xs uppercase mb-1">
                      {t("analyze.estLength")}
                    </div>
                    <div className="text-2xl font-mono text-white">
                      {result.autoLength}{" "}
                      <span className="text-sm text-slate-500">cm</span>
                    </div>
                    <div className="mt-2 text-[10px] text-cyan-400 flex items-center gap-1">
                      <Check className="w-3 h-3" /> {t("analyze.aiMeasured")}
                    </div>
                  </div>
                  <div className="bg-black/20 p-4 rounded-2xl border border-white/5">
                    <div className="text-slate-400 text-xs uppercase mb-1">
                      {t("analyze.estWeight")}
                    </div>
                    <div className="text-2xl font-mono text-white">
                      {result.estimatedWeight.toFixed(2)}{" "}
                      <span className="text-sm text-slate-500">kg</span>
                    </div>
                    <div className="mt-2 w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                      <div className="h-full bg-indigo-500 w-[70%]" />
                    </div>
                  </div>
                </div>

                {/* MARKET */}
                <h3 className="text-slate-500 text-xs font-bold uppercase tracking-widest mb-4 flex items-center gap-2">
                  <DollarSign className="w-3 h-3" />{" "}
                  {t("analyze.marketEconomics")}
                </h3>
                <div className="p-5 rounded-2xl bg-gradient-to-br from-slate-800/50 to-black/20 border border-white/5 mb-8">
                  <div className="flex justify-between items-center mb-4">
                    <div>
                      <div className="text-slate-400 text-xs uppercase">
                        {t("analyze.marketPrice")}
                      </div>
                      <div className="text-2xl font-bold text-white">
                        ₹{result.marketPrice}{" "}
                        <span className="text-sm font-normal text-slate-500">
                          / kg
                        </span>
                      </div>
                    </div>
                    <div
                      className={`px-3 py-1 rounded-lg border ${
                        result.marketTrend >= 0
                          ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                          : "bg-red-500/10 border-red-500/20 text-red-400"
                      }`}
                    >
                      <div className="flex items-center gap-1 text-xs font-bold">
                        <TrendingUp
                          className={`w-3 h-3 ${
                            result.marketTrend < 0 ? "rotate-180" : ""
                          }`}
                        />
                        {result.marketTrend > 0 ? "+" : ""}
                        {result.marketTrend}%
                      </div>
                    </div>
                  </div>
                </div>

                {/* ENVIRONMENT */}
                <h3 className="text-slate-500 text-xs font-bold uppercase tracking-widest mb-4 flex items-center gap-2">
                  <Droplets className="w-3 h-3" /> {t("analyze.envHealth")}
                </h3>
                <div className="grid grid-cols-2 gap-4 mb-8">
                  <div className="p-4 rounded-xl bg-black/20 border border-white/5">
                    <div className="text-white font-bold mb-1">
                      {result.waterTemp.toFixed(1)}°C
                    </div>
                    <div className="text-[10px] text-slate-500 uppercase">
                      {t("analyze.waterTemp")}
                    </div>
                  </div>
                  <div
                    className={`p-4 rounded-xl border border-white/5 ${
                      result.disease === "Healthy"
                        ? "bg-emerald-500/10 border-emerald-500/20"
                        : "bg-red-500/10 border-red-500/20"
                    }`}
                  >
                    <div
                      className={`font-bold mb-1 ${
                        result.disease === "Healthy"
                          ? "text-emerald-400"
                          : "text-red-400"
                      }`}
                    >
                      {result.disease || t("analyze.noAnomalies")}
                    </div>
                    <div className="text-[10px] text-slate-500 uppercase">
                      {t("analyze.pathology")}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* FOOTER (Sticky within the data column) */}
            <div className="absolute bottom-0 left-0 right-0 p-4 bg-slate-900/95 backdrop-blur-xl border-t border-white/10 z-30 lg:static lg:bg-transparent lg:border-0 lg:p-0">
              <div className="flex gap-4 max-w-md mx-auto lg:max-w-none">
                <Button
                  variant="outline"
                  onClick={() => {
                    setImageData(null);
                    setResult(null);
                  }}
                  className="flex-1 h-12 border-white/10 hover:bg-white/5 text-slate-300"
                >
                  {t("analyze.discard")}
                </Button>
                <Button
                  onClick={handleSave}
                  className="flex-[2] h-12 bg-cyan-600 hover:bg-cyan-500 text-white font-bold shadow-lg shadow-cyan-900/20"
                >
                  {t("analyze.saveRecord")}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // --- LANDING PAGE ---
  return (
    <div className="min-h-screen bg-gradient-ocean pt-safe-top pb-safe-bottom">
      <input
        ref={fileInputRef}
        onChange={(e) => {
          if (e.target.files?.[0]) {
            const reader = new FileReader();
            reader.onload = (ev) => analyzeImage(ev.target?.result as string);
            reader.readAsDataURL(e.target.files[0]);
          }
        }}
        type="file"
        accept="image/*"
        className="hidden"
      />

      <div className="container mx-auto max-w-md px-4 space-y-6">
        <div className="text-center py-8 relative">
          <div className="absolute inset-0 bg-gradient-glow opacity-30 blur-3xl"></div>
          <div className="relative">
            <h1 className="text-3xl font-bold text-gradient mb-3">
              � {t("analyze.aiScanner")}
            </h1>
            <p className="text-muted-foreground text-lg sm:animate-slide-up">
              {t("analyze.processingDescription")}
            </p>
            <div className="mt-4 flex justify-center items-center gap-2 text-sm text-muted-foreground">
              <div
                className={`w-2 h-2 rounded-full ${
                  isModelLoading
                    ? "bg-yellow-500 animate-pulse"
                    : modelError
                    ? "bg-red-500"
                    : "bg-emerald-500 animate-pulse-glow"
                }`}
              ></div>
              <span>
                {isModelLoading ? t("analyze.initializing") : "System Online"}
              </span>
            </div>
          </div>
        </div>

        <Card className="card-premium hover-glow animate-slide-up overflow-hidden">
          <div className="absolute inset-0 bg-gradient-primary opacity-5"></div>
          <CardHeader className="relative">
            <CardTitle className="flex items-center gap-3 text-xl">
              <div className="p-2 bg-gradient-primary rounded-lg">
                <Camera className="h-6 w-6 text-white" />
              </div>
              {t("analyze.professionalAnalysis")}
            </CardTitle>
            <p className="text-muted-foreground">
              {t("analyze.advancedDescription")}
            </p>
          </CardHeader>
          <CardContent className="space-y-4 relative">
            <Button
              onClick={() => setShowCamera(true)}
              disabled={isAnalyzing || isModelLoading}
              className="btn-premium btn-mobile w-full py-8 text-lg font-semibold relative overflow-hidden touch-feedback"
            >
              <div className="flex items-center justify-center gap-3">
                {isAnalyzing ? (
                  <>
                    <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>{t("analyze.analyzingAI")}</span>
                  </>
                ) : (
                  <>
                    <Camera className="h-6 w-6" />
                    <span>� {t("analyze.scanCatch")}</span>
                  </>
                )}
              </div>
            </Button>
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-muted"></div>
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-card px-2 text-muted-foreground">
                  {t("analyze.or")}
                </span>
              </div>
            </div>
            <Button
              variant="outline"
              className="btn-mobile w-full py-6 border-2 hover:border-primary/50 hover:bg-primary/5 transition-all"
              onClick={() => fileInputRef.current?.click()}
              disabled={isAnalyzing || isModelLoading}
            >
              <Upload className="h-5 w-5 mr-2" />{" "}
              {t("analyze.uploadGallery")}
            </Button>
          </CardContent>
        </Card>

        <div className="grid grid-cols-2 gap-4 animate-fade-in">
          <Card className="card-mobile hover-scale text-center p-4">
            <div className="text-2xl mb-2">�</div>
            <div className="font-semibold text-sm">
              {t("analyze.accuracy")}
            </div>
            <div className="text-xs text-muted-foreground">
              {t("analyze.aiConfidence")}
            </div>
          </Card>
          <Card className="card-mobile hover-scale text-center p-4">
            <div className="text-2xl mb-2">⚡</div>
            <div className="font-semibold text-sm">
              {t("analyze.instantResults")}
            </div>
            <div className="text-xs text-muted-foreground">
              {t("analyze.realTimeAnalysis")}
            </div>
          </Card>
          <Card className="card-mobile hover-scale text-center p-4">
            <div className="text-2xl mb-2">⚕️</div>
            <div className="font-semibold text-sm">
              {t("analyze.healthScoreCheck")}
            </div>
            <div className="text-xs text-muted-foreground">
              {t("analyze.freshnessCheck")}
            </div>
          </Card>
          <Card className="card-mobile hover-scale text-center p-4">
            <div className="text-2xl mb-2">�</div>
            <div className="font-semibold text-sm">
              {t("analyze.sizeEstimation")}
            </div>
            <div className="text-xs text-muted-foreground">
              {t("analyze.weightLength")}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}