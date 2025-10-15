# Eye Exercise Tracker - Design Guidelines

## Design Approach
**Selected Approach:** Design System (Healthcare-focused)
**System:** Fluent Design with Material Design influences
**Rationale:** This is a utility-focused healthcare application where clarity, accessibility, and professional trust are paramount. The design must support precise eye tracking exercises while maintaining medical-grade credibility.

---

## Core Design Principles
1. **Clinical Clarity:** Every element must be instantly recognizable and readable
2. **Calm Professionalism:** Reduce visual stress for users with eye strain
3. **Trustworthy Feedback:** Real-time status indicators must be unmistakable
4. **Accessibility First:** High contrast, large touch targets, clear typography

---

## Color Palette

### Light Mode (Primary)
- **Primary Blue:** 210 100% 60% (Trust, medical professionalism)
- **Success Green:** 140 60% 45% (Focus detected, positive feedback)
- **Warning Amber:** 38 90% 55% (Pause states, caution)
- **Alert Red:** 0 75% 60% (Loss of focus, attention needed)
- **Background:** 210 20% 98% (Soft, eye-friendly base)
- **Surface:** 0 0% 100% (Cards, overlays)
- **Text Primary:** 220 15% 20% (High contrast for readability)
- **Text Secondary:** 220 10% 50%

### Dark Mode
- **Primary Blue:** 210 80% 65%
- **Success Green:** 140 50% 55%
- **Warning Amber:** 38 80% 60%
- **Alert Red:** 0 70% 65%
- **Background:** 220 15% 12%
- **Surface:** 220 13% 16%
- **Text Primary:** 210 15% 95%
- **Text Secondary:** 210 10% 70%

---

## Typography

### Font Families
- **Primary (UI/Body):** 'Inter', 'Segoe UI', -apple-system, sans-serif
- **Headings:** 'Inter', weight 600-700
- **Monospace (Metrics):** 'Roboto Mono', monospace

### Type Scale
- **Display (H1):** 2.5rem / 600 weight / -0.02em tracking
- **Heading (H2-H3):** 1.75-1.5rem / 600 weight
- **Body Large:** 1.125rem / 400 weight / 1.6 line-height
- **Body:** 1rem / 400 weight / 1.5 line-height
- **Small/Caption:** 0.875rem / 500 weight

---

## Layout System

### Spacing Scale (Tailwind Units)
**Primary Units:** 2, 4, 6, 8, 12, 16, 24
- Micro spacing: 2-4 (icon gaps, tight padding)
- Component padding: 6-8 (buttons, cards)
- Section spacing: 12-16 (between major elements)
- Macro spacing: 24 (page sections)

### Grid Structure
- **Container:** max-w-7xl with px-4 sm:px-6 lg:px-8
- **Main Exercise Area:** 8 columns (md:col-span-8)
- **Sidebar (Exercise List):** 4 columns (md:col-span-4)
- **Camera Preview:** Fixed position bottom, width 640px desktop / 280px mobile

---

## Component Library

### Navigation & Controls
- **Control Panel Buttons:**
  - Large touch targets (min 48px height)
  - Primary: Solid blue with white text
  - Secondary: Outlined with blue border
  - Disabled: 50% opacity with cursor-not-allowed
  - Icon + text for clarity (Bootstrap Icons)

### Exercise Components
- **Exercise List Cards:**
  - Rounded-xl borders (12px radius)
  - Subtle shadow on hover (0 4px 12px rgba(0,0,0,0.08))
  - Active state: Blue left border (4px) + light blue background
  - Click ripple effect on selection
  
- **Moving Focus Dot (#calibration-point):**
  - Size: 30px diameter
  - Red gradient center (0 75% 60% to 0 85% 50%)
  - White 3px border
  - Glowing shadow (0 0 20px rgba(255,0,0,0.6))
  - Smooth 300ms ease transitions

- **Gaze Cursor:**
  - 18px yellow circle (45 100% 60%)
  - White 2px border
  - Subtle glow (0 0 12px rgba(255,255,0,0.5))
  - Position fixed, pointer-events none

### Status Indicators
- **Focus Indicator Badge:**
  - Top-right position, absolute
  - Green background when focused (140 60% 45%)
  - Red background when lost (0 75% 60%)
  - Backdrop blur for readability
  - Icon + text: "Face Detected" / "Face Lost"

- **Exercise Timer:**
  - Large monospace display (2.5rem)
  - Primary blue color
  - Format: MM:SS
  - Pulsing animation during active exercise

### Camera Feed
- **Container Styling:**
  - Fixed bottom position
  - Rounded top corners (15px)
  - 3px primary blue border
  - Box shadow elevation (0 -4px 20px rgba(0,0,0,0.15))
  - Mirrored video (scaleX(-1) transform)
  - Desktop: 640x480px
  - Mobile: Full width, 240px height

### Overlays
- **Exercise Guide Overlay:**
  - Full viewport coverage (fixed position)
  - Dark backdrop (rgba(0,0,0,0.85))
  - White text, centered content
  - Large instruction text (1.5rem)
  - Progress bar at bottom (h-2, blue gradient)

- **Notification Alerts:**
  - Slide down from top animation
  - Yellow/amber background (38 90% 95%)
  - Dark amber text (38 90% 30%)
  - Icon (exclamation-triangle) + message
  - Auto-dismiss after 5s or manual close

### Performance Summary Cards
- **Layout:** 2x2 grid on desktop, stack on mobile
- **Card Style:**
  - White background (dark mode: surface color)
  - Rounded-lg (8px)
  - Padding: p-6
  - Shadow: 0 2px 8px rgba(0,0,0,0.06)
  - Hover: Lift effect (translateY(-2px))

- **Metric Display:**
  - Large number: 3rem, weight 700, primary blue
  - Label below: 0.875rem, secondary text
  - Icon above metric (2rem size, accent color)
  - Percentage/streak indicators with color coding

---

## Animations & Transitions

### Exercise Flow
- **Countdown:** Scale pulse (1 → 1.15 → 1) over 600ms
- **Dot Movement:** Ease-in-out 400ms transition between positions
- **Success Feedback:** Scale pulse + green color flash when focused

### Micro-interactions
- **Button Hover:** translateY(-1px) + subtle shadow increase
- **Card Selection:** Left border slide-in (150ms)
- **Status Changes:** Color cross-fade (200ms)
- **Notification Entry:** slideDown 300ms ease-out

**Animation Budget:** Minimal, purposeful only - no decorative animations

---

## Accessibility Standards

### Contrast & Readability
- Minimum 4.5:1 contrast ratio for all text
- 7:1 for critical alerts and timers
- Large click targets (48x48px minimum)
- Clear focus indicators (2px blue outline, 4px offset)

### Assistive Technology
- Semantic HTML structure (header, main, aside, article)
- ARIA labels for dynamic content ("Exercise in progress", "Focus lost")
- Screen reader announcements for status changes
- Keyboard navigation (Tab, Enter, Space, Escape)

---

## Responsive Behavior

### Desktop (≥1024px)
- Side-by-side layout (exercise area + sidebar)
- Camera preview: 640x480px fixed bottom-right
- Full exercise visualization space

### Tablet (768-1023px)
- Stacked layout with collapsible sidebar
- Camera preview: 480x360px bottom-center
- Touch-optimized controls

### Mobile (<768px)
- Vertical stack: controls → exercise → list
- Camera preview: Full-width, 240px height
- Simplified metrics (one per row)
- Larger touch targets (56px buttons)

---

## Images

**Hero Section:** None - This is a utility app focused on immediate functionality
**Exercise Icons:** Bootstrap Icons (bi-eye-fill, bi-bullseye, bi-graph-up) in primary blue
**Camera Feed:** Live webcam stream (mirrored for natural viewing)
**Background:** Subtle gradient (light mode: white to light blue-gray / dark mode: dark to darker blue-gray)