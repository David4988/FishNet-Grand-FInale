import { useState, useEffect } from "react";
import { Eye, EyeOff, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface ExplainabilityOverlayProps {
  imageData: string;
  species: string;
  confidence: number;
  className?: string;
}

export const ExplainabilityOverlay = ({ 
  imageData, 
  species, 
  confidence, 
  className 
}: ExplainabilityOverlayProps) => {
  const [showOverlay, setShowOverlay] = useState(true);
  const [boundingBox, setBoundingBox] = useState({
    x: 15,
    y: 20,
    width: 70,
    height: 60
  });

  useEffect(() => {
    // Simulate AI processing and bounding box detection
    const timer = setTimeout(() => {
      // Generate a realistic bounding box based on confidence
      const centerX = 30 + Math.random() * 40; // 30-70%
      const centerY = 25 + Math.random() * 50; // 25-75%
      const size = Math.max(40, confidence * 0.8); // Larger box for higher confidence
      
      setBoundingBox({
        x: centerX - size/2,
        y: centerY - size/2,
        width: size,
        height: size * 0.75
      });
    }, 500);

    return () => clearTimeout(timer);
  }, [confidence]);

  return (
    <div className={cn("relative", className)}>
      <div className="relative overflow-hidden rounded-lg">
        <img
          src={imageData}
          alt="Fish analysis"
          className="w-full h-64 object-cover"
        />
        
        {/* Explainability Overlay */}
        {showOverlay && (
          <div className="absolute inset-0">
            {/* Semi-transparent overlay */}
            <div className="absolute inset-0 bg-black/20" />
            
            {/* Bounding Box */}
            <div 
              className="absolute border-2 border-primary bg-primary/10 rounded-lg transition-all duration-500"
              style={{
                left: `${boundingBox.x}%`,
                top: `${boundingBox.y}%`,
                width: `${boundingBox.width}%`,
                height: `${boundingBox.height}%`,
              }}
            >
              {/* Corner markers */}
              <div className="absolute -top-1 -left-1 w-3 h-3 bg-primary rounded-full" />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-primary rounded-full" />
              <div className="absolute -bottom-1 -left-1 w-3 h-3 bg-primary rounded-full" />
              <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-primary rounded-full" />
              
              {/* Label */}
              <div className="absolute -top-8 left-0 bg-primary text-primary-foreground px-2 py-1 rounded text-xs font-medium whitespace-nowrap">
                <div className="flex items-center gap-1">
                  <Zap className="w-3 h-3" />
                  Pattern Matched
                </div>
              </div>
            </div>
            
            {/* AI Analysis Info */}
            <Card className="absolute bottom-4 left-4 right-4 bg-card/90 backdrop-blur-sm border-primary/20">
              <div className="p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="bg-primary/20 text-primary">
                      AI Detection
                    </Badge>
                    <span className="text-sm font-medium">{species}</span>
                  </div>
                  <Badge 
                    variant={confidence >= 80 ? "default" : confidence >= 60 ? "secondary" : "outline"}
                    className="text-xs"
                  >
                    {confidence.toFixed(1)}%
                  </Badge>
                </div>
                
                <div className="text-xs text-muted-foreground space-y-1">
                  <div>✓ Fish body shape and proportions analyzed</div>
                  <div>✓ Color patterns and texture features extracted</div>
                  <div>✓ Species-specific characteristics identified</div>
                </div>
              </div>
            </Card>
          </div>
        )}
      </div>
      
      {/* Toggle Button */}
      <Button
        variant="outline"
        size="sm"
        onClick={() => setShowOverlay(!showOverlay)}
        className="absolute top-2 right-2 bg-card/90 backdrop-blur-sm"
      >
        {showOverlay ? (
          <>
            <EyeOff className="w-4 h-4 mr-1" />
            Hide AI
          </>
        ) : (
          <>
            <Eye className="w-4 h-4 mr-1" />
            Show AI
          </>
        )}
      </Button>
    </div>
  );
};