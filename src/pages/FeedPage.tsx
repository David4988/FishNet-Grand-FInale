import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Wifi, WifiOff, RefreshCw, Users, MessageSquare, Plus, Heart, MessageCircle, Share2 } from 'lucide-react';

// CORE COMPONENT IMPORTS
import { PostCard } from '@/components/social/PostCard';
import { MessagesDialog } from '@/components/social/MessagesDialog';
import { NewPostDialog } from '@/components/social/NewPostDialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

// CORE SERVICE IMPORTS
import { socialService, SocialPost, User } from '@/services/social';
import { authService } from '@/services/auth';
import { syncService, SyncStatus } from '@/services/sync';

// CORE HOOKS AND UTILS
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';


// --- ANIMATION VARIANTS ---
const feedContainerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

const postCardVariants = {
  hidden: { opacity: 0, y: 30, scale: 0.98 },
  visible: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.4, ease: 'easeOut' } },
  exit: { opacity: 0, y: -20, transition: { duration: 0.2 } },
};


export default function FeedPage() {
  const [posts, setPosts] = useState<SocialPost[]>([]);
  const [currentUser, setCurrentUser] = useState<User | null>(authService.getState().user);
  const [syncStatus, setSyncStatus] = useState<SyncStatus>(syncService.getStatus());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const { toast } = useToast();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [fabOpen, setFabOpen] = useState(false);

  useEffect(() => {
    loadPosts(); // Initial load

    const unsubscribeAuth = authService.subscribe((state) => setCurrentUser(state.user));
    const unsubscribeSync = syncService.subscribe((status) => {
      setSyncStatus(status);
      if (!status.isSyncing && status.isOnline) {
        loadPosts();
      }
    });
    // Subscribe to the social service for real-time post updates
    const unsubscribeSocial = socialService.subscribe(loadPosts);

    return () => {
      unsubscribeAuth();
      unsubscribeSync();
      unsubscribeSocial();
    };
  }, []);

  const loadPosts = () => {
    const feedPosts = socialService.getFeedPosts();
    setPosts(feedPosts);
  };

  const handleLike = async (postId: string) => {
    if (!currentUser) return;
    // The socialService now handles the state update and notifies listeners,
    // so we don't need to manually setPosts here anymore.
    await socialService.toggleLike(postId, currentUser.id);
    if (!syncStatus.isOnline) {
      toast({ title: "Liked offline", description: "Your like will sync when you're back online" });
    }
  };

  const handleComment = async (postId: string) => {
    if (!currentUser) return;
    try {
      const sampleComments = ["Amazing catch! 🎣", "What a beauty!", "Great photo quality!", "This species is rare!", "Perfect timing! 🌊"];
      const randomComment = sampleComments[Math.floor(Math.random() * sampleComments.length)];
      await socialService.addComment(postId, currentUser.id, currentUser, randomComment);
      toast({ title: "Comment added!", description: syncStatus.isOnline ? "Comment posted successfully" : "Comment saved offline" });
    } catch (error) {
      toast({ title: "Error", description: "Failed to add comment", variant: "destructive" });
    }
  };

  const handleShare = (postId: string) => {
    const post = posts.find(p => p.id === postId);
    if (!post) return;
    if (navigator.share) {
      navigator.share({ title: `Check out this ${post.species} catch on Fish Net!`, text: post.caption, url: window.location.href });
    } else {
      navigator.clipboard.writeText(window.location.href);
      toast({ title: "Link copied!", description: "Share this awesome catch with friends." });
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      if (syncStatus.isOnline) await syncService.forcSync();
      loadPosts(); // Manually reload posts on refresh
      toast({ title: "Feed refreshed", description: syncStatus.isOnline ? "Latest posts loaded" : "Showing cached posts" });
    } catch (error) {
      toast({ title: "Refresh failed", description: "Could not refresh feed", variant: "destructive" });
    } finally {
      setIsRefreshing(false);
    }
  };

  const isPostLiked = (postId: string) => currentUser ? socialService.isPostLiked(postId, currentUser.id) : false;

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      <div className="absolute inset-0 bg-grid-sky-400/[0.05]" />
      
      <header className="sticky top-0 z-40 bg-slate-900/60 backdrop-blur-xl border-b border-sky-400/20">
        <div className="flex items-center justify-between p-4 h-20">
          <div className="flex items-center gap-3">
            <div>
              <h1 className="text-2xl font-bold text-sky-400 tracking-wider">Fish Net</h1>
            <p>Feed</p>
            </div>
            <Badge className={cn("hidden sm:inline-flex items-center text-xs", syncStatus.isOnline ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30" : "bg-amber-500/20 text-amber-300 border-amber-500/30")}>
              {syncStatus.isOnline ? <><Wifi className="h-3 w-3 mr-1.5" /> Online</> : <><WifiOff className="h-3 w-3 mr-1.5" /> Offline</>}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <div className="hidden sm:block">
              <NewPostDialog onPostCreated={loadPosts}>
                <Button className="bg-sky-500 hover:bg-sky-600 text-white font-bold rounded-full shadow-lg shadow-sky-500/30 transition-all duration-300">
                    <Plus className="h-5 w-5 mr-2" /> New Post
                </Button>
              </NewPostDialog>
            </div>
            <MessagesDialog>
                <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white hover:bg-sky-500/20">
                    <Users className="h-5 w-5" />
                </Button>
            </MessagesDialog>
            {/* --- EDIT: Hidden on mobile, visible on sm screens and up --- */}
            <Button variant="ghost" size="icon" onClick={handleRefresh} disabled={isRefreshing} className="hidden sm:flex text-gray-400 hover:text-white hover:bg-sky-500/20">
              <RefreshCw className={cn("h-5 w-5", isRefreshing && "animate-spin")} />
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto max-w-md p-4 pb-32">
        {syncStatus.isSyncing && (
          <div className="bg-sky-500/20 p-3 rounded-lg text-center text-sm text-sky-300 flex items-center justify-center gap-2 mb-4">
            <RefreshCw className="h-4 w-4 animate-spin" />
            <span>Syncing your data...</span>
          </div>
        )}
        <AnimatePresence mode="wait">
          {posts.length > 0 ? (
            <motion.div
              key="posts-list"
              initial="hidden"
              animate="visible"
              exit={postCardVariants.exit}
              variants={feedContainerVariants}
              className="space-y-6"
            >
              {posts.map((post) => (<PostCard key={post.id} post={post} onLike={handleLike} onComment={handleComment} onShare={handleShare} isLiked={isPostLiked(post.id)} />))}
            </motion.div>
          ) : (
            <motion.div
              key="empty-state"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={postCardVariants.exit}
              className="text-center py-20"
            >
              <div className="w-20 h-20 bg-sky-500/10 border-2 border-sky-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
                <Users className="h-10 w-10 text-sky-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">The Ocean is Quiet</h3>
              <p className="text-gray-400 mb-6">Be the first to share a catch with the community!</p>
              <NewPostDialog onPostCreated={loadPosts}>
                <Button size="lg" className="bg-sky-500 hover:bg-sky-600 text-white font-bold rounded-full shadow-lg shadow-sky-500/30 transition-all duration-300"><Plus className="h-5 w-5 mr-2" /> Create First Post</Button>
              </NewPostDialog>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <div className="sm:hidden fixed bottom-24 right-4 z-40">
         <NewPostDialog onPostCreated={loadPosts}>
             <motion.div
                 whileHover={{ scale: 1.1 }}
                 whileTap={{ scale: 0.9 }}
              >
                <Button
                  className="rounded-full h-16 w-16 bg-sky-500 hover:bg-sky-600 shadow-xl shadow-sky-500/40 text-white"
                >
                    <Plus size={28} />
                </Button>
              </motion.div>
        </NewPostDialog>
      </div>

    </div>
  );
}

