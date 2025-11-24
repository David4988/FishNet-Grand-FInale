import { useState, useEffect } from "react";
import { Wifi, WifiOff, Cloud, CloudOff, RefreshCw, CircleAlert as AlertCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { syncService, SyncStatus } from "@/services/sync";
import { cn } from "@/lib/utils";

export const OfflineIndicator = () => {
  const [syncStatus, setSyncStatus] = useState<SyncStatus>(syncService.getStatus());
  const [showBanner, setShowBanner] = useState(false);

  useEffect(() => {
    const unsubscribe = syncService.subscribe((status) => {
      setSyncStatus(status);
      
      // Show banner when going offline or coming online
      if (status.isOnline !== syncStatus.isOnline) {
        setShowBanner(true);
        // Hide banner after 4 seconds for online, keep visible for offline
        if (status.isOnline) {
          setTimeout(() => setShowBanner(false), 4000);
        }
      }
    });

    // Show initial state if offline or has pending items
    if (!navigator.onLine || syncStatus.pendingCatches > 0 || syncStatus.pendingPosts > 0) {
      setShowBanner(true);
    }

    return unsubscribe;
  }, []);

  const handleSync = async () => {
    if (syncStatus.isOnline && !syncStatus.isSyncing) {
      await syncService.forcSync();
    }
  };

  const hasPendingItems = syncStatus.pendingCatches > 0 || syncStatus.pendingPosts > 0;

  // Always show if offline or has pending items
  if (!showBanner && syncStatus.isOnline && !hasPendingItems) return null;

  return (
    <div className={cn(
      "fixed top-0 left-0 right-0 z-50 transition-all duration-300",
      showBanner || !syncStatus.isOnline || hasPendingItems ? "translate-y-0" : "-translate-y-full"
    )}>
      <Card className={cn(
        "mx-4 mt-4 border-0 shadow-lg",
        syncStatus.isOnline 
          ? hasPendingItems 
            ? "bg-blue-50 border-blue-200" 
            : "bg-success/10 border-success/20"
          : "bg-warning/10 border-warning/20"
      )}>
        <div className="p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {syncStatus.isOnline ? (
                <Wifi className="w-4 h-4 text-success" />
              ) : (
                <WifiOff className="w-4 h-4 text-warning" />
              )}
              <span className={cn(
                "text-sm font-medium",
                syncStatus.isOnline ? "text-success" : "text-warning"
              )}>
                {syncStatus.isOnline ? "Online" : "Offline Mode"}
              </span>
              
              {syncStatus.isSyncing && (
                <RefreshCw className="w-4 h-4 text-blue-600 animate-spin" />
              )}
            </div>
            
            <div className="flex items-center gap-2">
              {hasPendingItems && (
                <Badge 
                  variant="secondary"
                  className="bg-blue-100 text-blue-700 border-blue-200 text-xs"
                >
                  {syncStatus.pendingCatches + syncStatus.pendingPosts} pending
                </Badge>
              )}
              
              <Badge 
                variant={syncStatus.isOnline ? "default" : "secondary"}
                className={cn(
                  "text-xs",
                  syncStatus.isOnline 
                    ? "bg-success/20 text-success border-success/30" 
                    : "bg-warning/20 text-warning border-warning/30"
                )}
              >
                {syncStatus.isOnline ? "Connected" : "Data cached locally"}
              </Badge>
            </div>
          </div>

          {/* Pending Items Details */}
          {hasPendingItems && (
            <div className="mt-3 pt-3 border-t border-border/50">
              <div className="flex items-center justify-between">
                <div className="text-xs text-muted-foreground space-y-1">
                  {syncStatus.pendingCatches > 0 && (
                    <div>📸 {syncStatus.pendingCatches} catches waiting to sync</div>
                  )}
                  {syncStatus.pendingPosts > 0 && (
                    <div>📝 {syncStatus.pendingPosts} posts waiting to sync</div>
                  )}
                  {syncStatus.lastSyncTime && (
                    <div>Last sync: {new Date(syncStatus.lastSyncTime).toLocaleTimeString()}</div>
                  )}
                </div>
                
                {syncStatus.isOnline && (
                  <Button
                    size="sm"
                    onClick={handleSync}
                    disabled={syncStatus.isSyncing}
                    className="bg-blue-600 hover:bg-blue-700 text-white text-xs h-8"
                  >
                    {syncStatus.isSyncing ? (
                      <>
                        <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
                        Syncing...
                      </>
                    ) : (
                      <>
                        <Cloud className="w-3 h-3 mr-1" />
                        Sync Now
                      </>
                    )}
                  </Button>
                )}
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export const ConnectionStatus = () => {
  const [syncStatus, setSyncStatus] = useState<SyncStatus>(syncService.getStatus());

  useEffect(() => {
    const unsubscribe = syncService.subscribe(setSyncStatus);
    return unsubscribe;
  }, []);

  return (
    <div className="flex items-center gap-1">
      {syncStatus.isOnline ? (
        <Cloud className="w-3 h-3 text-success" />
      ) : (
        <CloudOff className="w-3 h-3 text-warning" />
      )}
      <span className={cn(
        "text-xs",
        syncStatus.isOnline ? "text-success" : "text-warning"
      )}>
        {syncStatus.isOnline ? "Online" : "Offline"}
      </span>
      
      {(syncStatus.pendingCatches > 0 || syncStatus.pendingPosts > 0) && (
        <Badge variant="secondary" className="ml-1 text-xs bg-blue-100 text-blue-700">
          {syncStatus.pendingCatches + syncStatus.pendingPosts}
        </Badge>
      )}
    </div>
  );
};