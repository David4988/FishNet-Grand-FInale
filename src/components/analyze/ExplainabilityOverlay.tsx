import { useState } from "react";
import { Eye, EyeOff, Scan, Target, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { BoundingBox } from "@/types/fishnet";
import { useTranslation } from "react-i18next";

interface ExplainabilityOverlayProps {
  imageData: string;
  species: string;
  confidence: number;
  boundingBox?: BoundingBox;
  className?: string;
}

export const ExplainabilityOverlay = ({ 
  imageData, 
  species, 
  confidence, 
  boundingBox,
  className 
}: ExplainabilityOverlayProps) => {
  const [showOverlay, setShowOverlay] = useState(true);

  // Default to center if no box (Fallback)
  const box = boundingBox || { yMin: 0.15, xMin: 0.15, yMax: 0.85, xMax: 0.85 };

  const style = {
    top: `${box.yMin * 100}%`,
    left: `${box.xMin * 100}%`,
    width: `${(box.xMax - box.xMin) * 100}%`,
    height: `${(box.yMax - box.yMin) * 100}%`,
  };
  const { t } = useTranslation();
  return (
    <div className={cn("relative w-full overflow-hidden rounded-2xl bg-slate-950 shadow-2xl", className)}>
      {/* Main Image */}
      <div className="relative">
        <img
          src={imageData}
          alt="Fish analysis"
          className="w-full h-auto max-h-[50vh] object-cover" // Changed to cover for immersive feel
        />
        
        {/* Grid Overlay Effect (Subtle texture) */}
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay pointer-events-none"></div>
        
        {/* �️ TECH OVERLAY */}
        {showOverlay && (
          <div className="absolute inset-0">
            {/* Darken edges to focus attention */}
            <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-slate-950/40" />
            
            {/* THE TARGET BOX */}
            <div 
              className="absolute border border-cyan-400/50 shadow-[0_0_30px_rgba(34,211,238,0.2)] transition-all duration-700 ease-out"
              style={style}
            >
              {/* Animated Scanline inside the box */}
              <div className="absolute inset-0 overflow-hidden opacity-30">
                <div className="w-full h-full bg-gradient-to-b from-transparent via-cyan-400/20 to-transparent animate-scanline" />
              </div>

              {/* Tech Corners */}
              <div className="absolute -top-[1px] -left-[1px] w-4 h-4 border-l-2 border-t-2 border-cyan-400" />
              <div className="absolute -top-[1px] -right-[1px] w-4 h-4 border-r-2 border-t-2 border-cyan-400" />
              <div className="absolute -bottom-[1px] -left-[1px] w-4 h-4 border-l-2 border-b-2 border-cyan-400" />
              <div className="absolute -bottom-[1px] -right-[1px] w-4 h-4 border-r-2 border-b-2 border-cyan-400" />
              
              {/* Floating HUD Label */}
              <div className="absolute -top-12 left-0 flex flex-col items-start">
                <div className="flex items-center gap-2 bg-slate-900/90 backdrop-blur-md border border-cyan-500/30 px-3 py-1.5 rounded-tr-lg rounded-tl-lg">
                   <Target className="w-3 h-3 text-cyan-400 animate-pulse" />
                   <span className="text-xs font-mono font-bold text-cyan-100 tracking-wider uppercase">
  {t('analyze.targetLocked')}
</span>
                </div>
                <div className="bg-cyan-500/90 text-slate-950 px-3 py-1 text-sm font-bold shadow-lg rounded-b-lg rounded-tr-lg">
                   {species} <span className="opacity-75 text-xs">({confidence.toFixed(1)}%)</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Bottom Control Bar (Inside Image) */}
      <div className="absolute bottom-4 right-4 flex gap-2">
        <Button
          variant="secondary"
          size="sm"
          onClick={() => setShowOverlay(!showOverlay)}
          className="bg-slate-900/60 backdrop-blur-md text-white hover:bg-slate-800 border border-white/10 text-xs"
        >
          {showOverlay ? <EyeOff className="w-3 h-3 mr-2" /> : <Eye className="w-3 h-3 mr-2" />}
          {showOverlay ? t('analyze.hideHud') : t('analyze.showHud')}
        </Button>
      </div>
    </div>
  );
};