
import { CatchMap } from '@/components/map/CatchMap';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function MapPage() {
  const navigate = useNavigate();
  return (
    <div className="h-screen bg-gradient-to-br from-blue-50 to-cyan-50">
      <div className="flex items-center p-4">
        <Button
          type="button"
          onClick={() => navigate(-1)}
          className="flex items-center gap-2 bg-gradient-primary text-white font-semibold shadow-glow px-4 py-2 rounded-lg hover:opacity-90 transition-all duration-200"
        >
          <ArrowLeft className="w-5 h-5" />
          <span>Back</span>
        </Button>
      </div>
      <div className="h-[calc(100%-64px)]">
        <CatchMap className="h-full" />
      </div>
    </div>
  );
}