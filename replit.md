# Fish Net Application

## Overview
Fish Net is a Progressive Web App (PWA) for fishermen to identify fish species using AI, track catches, view locations on maps, and maintain a history of their catches. Built with Vite, React, TypeScript, and shadcn/ui.

## Project Architecture

### Technology Stack
- **Frontend Framework**: React 18 with TypeScript
- **Build Tool**: Vite 5
- **Styling**: Tailwind CSS with shadcn/ui components
- **Routing**: React Router DOM
- **AI/ML**: TensorFlow.js and Hugging Face Transformers
- **Maps**: Leaflet and Mapbox GL
- **State Management**: TanStack Query (React Query)
- **Local Storage**: IndexedDB via idb library
- **PWA**: Service Worker with offline support

### Project Structure
```
src/
├── components/        # Reusable UI components
│   ├── analyze/      # Fish analysis components
│   ├── auth/         # Authentication components
│   ├── history/      # Catch history components
│   ├── layout/       # Layout components (navigation, etc.)
│   ├── map/          # Map-related components
│   ├── offline/      # Offline indicator
│   ├── reference/    # Species library
│   ├── reports/      # PDF generation
│   ├── social/       # Social feed components
│   └── ui/           # shadcn/ui base components
├── hooks/            # Custom React hooks
├── lib/              # Utility functions
├── pages/            # Page components (routes)
├── services/         # Business logic and data services
│   ├── auth.ts       # Authentication service
│   ├── database.ts   # IndexedDB database service
│   ├── pwa.ts        # PWA/Service Worker service
│   ├── sampleData.ts # Sample data seeding
│   ├── social.ts     # Social features
│   ├── sync.ts       # Sync service
│   └── tensorflow.ts # AI/ML service
├── App.tsx           # Main app component
└── main.tsx          # Entry point
```

## Configuration

### Vite Configuration
- **Host**: 0.0.0.0 (required for Replit)
- **Port**: 5000 (Replit standard)
- **Allowed Hosts**: ["*"] (required for Replit's proxy environment)
- **HMR**: Configured for Replit's proxy environment with WSS protocol

### Workflow
- **Name**: Start application
- **Command**: `npm run dev`
- **Port**: 5000 (webview output)

### Deployment Configuration
- **Target**: Autoscale (stateless web app)
- **Build**: `npm run build`
- **Run**: `npm run preview`

## Key Features
1. **Fish Identification**: AI-powered fish species identification using camera
2. **Catch History**: Track and manage fishing catches with images
3. **Map Integration**: View catch locations on interactive maps
4. **Offline Support**: PWA with service worker for offline functionality
5. **Species Library**: Reference library for fish species
6. **Social Feed**: Share catches with the community
7. **PDF Reports**: Generate catch reports

## Development Notes

### Important Files
- `vite.config.ts`: Vite configuration with Replit-specific settings
- `src/services/database.ts`: IndexedDB implementation for local data storage
- `src/services/tensorflow.ts`: TensorFlow.js integration for AI features

### Known Issues Fixed
- Fixed import error: Changed `HistoryList` from named export to default export import in `HistoryPage.tsx`

### Environment Setup
- No backend server required (frontend-only app)
- Uses IndexedDB for local data persistence
- Service Worker handles offline capabilities
- Sample data is seeded on first load

## Recent Changes

### UI Enhancement & Internationalization (2025-09-29)
1. **Navigation Branding**: Updated all pages (Feed, My Catches, Profile) to display "Fish Net" branding consistently
2. **Loading Screens**: Added loading states (1s for login/signup, 1.5s for image upload/analyze) for better UX
3. **My Catches UI**: Fixed grid items to show dark background (active:bg-gray-900) when clicked
4. **Personal Chat**: Enhanced chat service with 3 personal conversations (2 with messages, 1 empty) with localStorage caching
5. **Real Fish Images**: Downloaded 30+ real fish photos and relocated from attached_assets to public/fish_images directory for proper Vite serving
6. **Settings Dialog**: Implemented comprehensive settings with name change, location permissions, email display, and language selection with confirmation
7. **Internationalization (i18n)**: Configured full i18n support with 12 languages:
   - English (en) - base language
   - Tamil (ta), Telugu (te), Hindi (hi), Kannada (kn), Malayalam (ml), Gujarati (gu), Marwadi (mwr), Bengali (bn), Punjabi (pa), Marathi (mr), Odia (or)
   - Language selection persisted to localStorage
   - Language change confirmation dialog
8. **Image Management**: Updated sample data to use /fish_images/ paths from public directory

### GitHub Import Setup (2025-09-29)
1. Installed npm dependencies (553 packages)
2. Added `allowedHosts: ["*"]` to vite.config.ts for Replit proxy compatibility
3. Verified workflow is running successfully on port 5000
4. Confirmed deployment configuration (autoscale target)
5. Tested application - all features working (IndexedDB, Service Worker, UI)

### Previous Changes (2025-09-29)
1. Configured Vite for Replit environment (port 5000, 0.0.0.0 host, WSS HMR)
2. Set up workflow for development server
3. Fixed import error in HistoryPage.tsx
4. Configured deployment settings for autoscale
5. Verified application runs successfully with all features working
6. Reorganized Feed page - moved Community and Chat into Messages dialog button
7. Created FishermanIndia community chat with fish-related messages and photos
8. Added 3 chat conversations (2 with messages, 1 empty) with localStorage caching
9. Fixed My Catches page image loading to show images immediately
10. Fixed Map page - replaced blue screen with OpenStreetMap tiles
11. Added custom fish image markers with hover effects on map
12. Implemented random fish images for sample data, original images for user catches
