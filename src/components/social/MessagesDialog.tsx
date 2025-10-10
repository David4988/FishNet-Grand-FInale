import { useState } from 'react';
import { MessageCircle, Users } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent } from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { CommunityChat } from './CommunityChat';
import { ChatInterface } from './ChatInterface';

export const MessagesDialog = () => {
  const [open, setOpen] = useState(false);

  return (
    <>
      <Button 
        variant="ghost"
        size="sm"
        onClick={() => setOpen(true)}
        className="text-muted-foreground hover:text-foreground"
      >
        <MessageCircle className="h-4 w-4" />
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-4xl w-[95vw] h-[90vh] p-0 overflow-hidden">
          <Tabs defaultValue="community" className="w-full h-full flex flex-col">
            <div className="sticky top-0 z-30 bg-background border-b border-border">
              <TabsList className="w-full grid grid-cols-2 h-12 bg-muted/50">
                <TabsTrigger 
                  value="community" 
                  className="data-[state=active]:bg-gradient-primary data-[state=active]:text-white"
                >
                  <Users className="h-4 w-4 mr-2" />
                  Community
                </TabsTrigger>
                <TabsTrigger 
                  value="chat" 
                  className="data-[state=active]:bg-gradient-primary data-[state=active]:text-white"
                >
                  <MessageCircle className="h-4 w-4 mr-2" />
                  Chats
                </TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="community" className="mt-0 flex-1">
              <CommunityChat />
            </TabsContent>

            <TabsContent value="chat" className="mt-0 flex-1">
              <ChatInterface />
            </TabsContent>
          </Tabs>
        </DialogContent>
      </Dialog>
    </>
  );
};
