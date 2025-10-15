# Eye Exercise Tracker

## Overview

This is a web-based eye exercise application that helps users perform guided eye tracking exercises with real-time focus detection. The system uses computer vision (MediaPipe) to detect face presence and provides visual feedback during exercises. The application features a healthcare-focused design system emphasizing clarity, accessibility, and professional trust.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Framework**: React with TypeScript using Vite as the build tool

**UI Component System**: 
- Shadcn/ui components with Radix UI primitives
- Tailwind CSS for styling with custom healthcare-focused design tokens
- New York style variant with custom color scheme
- Design system inspired by Fluent Design and Material Design principles

**State Management**:
- TanStack React Query (v5) for server state management
- Wouter for client-side routing

**Design Principles**:
- Clinical clarity with high contrast ratios for users with eye strain
- Calm professional aesthetic using blue/green color palette
- Accessibility-first approach with large touch targets
- Support for both light and dark modes

### Backend Architecture

**Primary Stack**: Python Flask servers with Node.js/TypeScript launcher

**Multi-Server Architecture**:
The system uses a microservices approach with three separate servers:

1. **Main Application Server** (Port 5000)
   - Serves HTML templates and static files
   - Acts as reverse proxy to tracking servers
   - Handles user-facing routes

2. **Eye Tracking Server** (Port 5001)
   - Dedicated computer vision processing service
   - Provides simplified face detection API
   - Streams video feed with visual feedback
   - Uses MediaPipe FaceDetection model with 0.7 confidence threshold

3. **Enhanced Eye Tracking Server** (Port 5002)
   - Advanced face detection with bounding box visualization
   - Processes frames in background thread for efficiency
   - Provides annotated video stream with confidence scores

**Computer Vision Pipeline**:
- MediaPipe FaceDetection model (full-range, model_selection=1)
- High confidence threshold (0.7) for accuracy
- Real-time frame processing with visual feedback (bounding boxes, keypoints)
- OpenCV for camera capture and frame manipulation

**Session Management**:
- Flask sessions with filesystem storage
- PostgreSQL-ready with Drizzle ORM schema defined
- Currently uses SQLite with planned Postgres migration

### Data Storage

**ORM**: Drizzle ORM configured for PostgreSQL

**Database Schema** (defined in shared/schema.ts):
- Users table with UUID primary keys
- Username/password authentication fields
- Schema uses Drizzle-Zod for validation

**Current State**: 
- Development uses in-memory storage (MemStorage class)
- Production expects PostgreSQL via DATABASE_URL environment variable
- Migration files configured to output to ./migrations directory

**Data Collection**:
- Real-time exercise metrics tracking (referenced in Python code)
- Exercise sessions storage
- Session-based user data management

### External Dependencies

**Python Computer Vision Stack**:
- MediaPipe (v0.10.14+) - Face detection model
- OpenCV (opencv-python v4.10.0.84+) - Camera capture and image processing
- NumPy - Array operations for image data
- Flask-CORS - Cross-origin resource sharing

**Node.js/TypeScript Stack**:
- Express.js - Not actively used; servers are Python Flask
- Vite - Frontend build tool and dev server
- ESBuild - Server bundle compilation

**UI Component Libraries**:
- @radix-ui/* - Unstyled accessible components (40+ packages)
- @tanstack/react-query - Server state management
- class-variance-authority - Component variant styling
- Tailwind CSS - Utility-first styling

**Database & ORM**:
- Drizzle ORM (v0.39.1) - Type-safe ORM
- Drizzle-Zod (v0.7.0) - Schema validation
- @neondatabase/serverless (v0.10.4) - Serverless Postgres driver
- connect-pg-simple (v10.0.0) - PostgreSQL session store

**Development Tools**:
- TypeScript - Type safety across codebase
- Replit plugins - Development environment integration
- Python Poetry (implied by pyproject.toml references) - Python dependency management

**Deployment Configuration**:
- Multi-process startup via Python script (start_servers.py)
- Environment variables for database configuration
- Node.js launcher (server/index.ts) spawns Python processes
- Session secret key configuration for production security