import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface MobileLayoutProps {
  children: ReactNode;
  className?: string;
  showBottomNav?: boolean;
}

export const MobileLayout = ({ children, className, showBottomNav = true }: MobileLayoutProps) => {
  return (
    <div className={cn(
      "min-h-screen bg-background",
      // Safe area support for iOS
      "pt-[var(--safe-area-inset-top)] pb-[var(--safe-area-inset-bottom)]",
      className
    )}>
      {/* Main content area */}
      <main className={cn(
        "flex-1 overflow-x-hidden",
        showBottomNav && "pb-20" // Space for bottom navigation
      )}>
        {children}
      </main>
    </div>
  );
};