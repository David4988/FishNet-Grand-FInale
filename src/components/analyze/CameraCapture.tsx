import { useState, useRef } from "react";
import { Camera, Upload, X, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface CameraCaptureProps {
  onImageCapture: (imageData: string, file?: File) => void;
  onClose?: () => void;
  className?: string;
}

export const CameraCapture = ({ onImageCapture, onClose, className }: CameraCaptureProps) => {
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const startCamera = async () => {
    try {
      setIsLoading(true);
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'environment',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      // Fallback to file input if camera fails
      fileInputRef.current?.click();
    } finally {
      setIsLoading(false);
    }
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      if (context) {
        context.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg', 0.9);
        setCapturedImage(imageData);
        
        // Stop camera stream
        const stream = video.srcObject as MediaStream;
        stream?.getTracks().forEach(track => track.stop());
      }
    }
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setIsLoading(true);
      const reader = new FileReader();
      reader.onload = async (e) => {
        // Add 1.5 second loading state for better UX when uploading from device
        await new Promise(resolve => setTimeout(resolve, 1500));
        const imageData = e.target?.result as string;
        setCapturedImage(imageData);
        setIsLoading(false);
      };
      reader.readAsDataURL(file);
    }
  };

  const confirmImage = () => {
    if (capturedImage) {
      onImageCapture(capturedImage);
      setCapturedImage(null);
    }
  };

  const retakePhoto = () => {
    setCapturedImage(null);
    startCamera();
  };

  return (
    <div className={cn("relative w-full h-full", className)}>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleFileSelect}
        className="hidden"
      />
      
      <canvas ref={canvasRef} className="hidden" />

      {!capturedImage ? (
        <div className="relative w-full h-full bg-black rounded-lg overflow-hidden">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
          />
          
          {/* Camera overlay */}
          <div className="absolute inset-0 flex flex-col justify-between p-6">
            <div className="flex justify-between items-start">
              {onClose && (
                <Button
                  variant="secondary"
                  size="icon"
                  onClick={onClose}
                  className="bg-black/50 hover:bg-black/70 text-white border-none"
                >
                  <X size={20} />
                </Button>
              )}
            </div>

            {/* Camera controls */}
            <div className="flex justify-center items-center gap-4">
              <Button
                variant="secondary"
                onClick={() => fileInputRef.current?.click()}
                className="bg-black/50 hover:bg-black/70 text-white border-none"
              >
                <Upload size={20} className="mr-2" />
                Gallery
              </Button>

              <Button
                onClick={videoRef.current ? capturePhoto : startCamera}
                disabled={isLoading}
                className="w-16 h-16 rounded-full bg-primary hover:bg-primary-glow shadow-glow"
              >
                <Camera size={24} />
              </Button>
            </div>
          </div>

          {/* Viewfinder overlay */}
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute inset-6 border-2 border-primary/50 rounded-lg">
              <div className="absolute top-0 left-0 w-8 h-8 border-l-4 border-t-4 border-primary rounded-tl-lg" />
              <div className="absolute top-0 right-0 w-8 h-8 border-r-4 border-t-4 border-primary rounded-tr-lg" />
              <div className="absolute bottom-0 left-0 w-8 h-8 border-l-4 border-b-4 border-primary rounded-bl-lg" />
              <div className="absolute bottom-0 right-0 w-8 h-8 border-r-4 border-b-4 border-primary rounded-br-lg" />
            </div>
          </div>
        </div>
      ) : (
        <Card className="relative w-full h-full bg-black overflow-hidden">
          <img
            src={capturedImage}
            alt="Captured fish"
            className="w-full h-full object-cover"
          />
          
          <div className="absolute bottom-6 inset-x-6 flex justify-center gap-4">
            <Button
              variant="secondary"
              onClick={retakePhoto}
              className="bg-black/50 hover:bg-black/70 text-white border-none"
            >
              <X size={20} className="mr-2" />
              Retake
            </Button>
            
            <Button
              onClick={confirmImage}
              className="bg-primary hover:bg-primary-glow shadow-glow"
            >
              <Check size={20} className="mr-2" />
              Analyze Fish
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
};