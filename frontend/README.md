# Comments Analyzer

A modern web application that extracts and visualizes insights from YouTube comments. Built with a premium dark-themed UI featuring glassmorphism effects, gradient backgrounds, and intelligent data visualization.

## What We Do

Comments Analyzer helps content creators and marketers understand audience sentiment by analyzing YouTube video or channel comments. The application processes comments through semantic analysis, clustering similar discussions, filtering spam, and generating actionable insights.

Key features:
- Paste YouTube video or channel links for analysis
- Real-time processing with step-by-step progress tracking
- Visual cluster distribution charts
- Discussion clusters organized by themes
- Common ideas and themes extraction
- Opportunities and signals identification (feature requests, complaints, questions)

## Technologies Used

### Frontend Framework
- React 19.2.0
- TypeScript 5.9.3

### Styling
- Tailwind CSS 3.4.19
- Custom dark theme with indigo/violet gradients
- Glassmorphism UI effects
- Google Fonts (Inter Tight, Archivo, Instrument Sans, Google Sans)

### Data Visualization
- Recharts 2.10.3 for interactive charts

### Icons
- React Icons 5.5.0 (Feather Icons)

### Build Tools
- Vite 7.2.4
- PostCSS with Autoprefixer

### Development
- ESLint 9.39.1
- TypeScript ESLint

## Project Structure

```
comments-analyzer-fe/
├── src/
│   ├── App.tsx          # Main application component
│   ├── App.css          # Component-specific styles
│   ├── index.css        # Global styles and Tailwind directives
│   ├── main.tsx         # Application entry point
│   └── SimpleBarChart.tsx  # Custom SVG chart component (fallback)
├── public/              # Static assets
├── index.html           # HTML template
├── tailwind.config.js   # Tailwind CSS configuration
├── postcss.config.js    # PostCSS configuration
├── tsconfig.json        # TypeScript configuration
└── package.json         # Dependencies and scripts
```

## How to Run

### Prerequisites

- Node.js (v18 or higher recommended)
- npm or yarn package manager

### Installation

1. Clone the repository or navigate to the project directory
2. Install dependencies:
   ```bash
   npm install
   ```

### Development

Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173` (or the port shown in the terminal).

### Build

Create a production build:
```bash
npm run build
```

The optimized build will be in the `dist` directory.

### Preview Production Build

Preview the production build locally:
```bash
npm run preview
```

### Linting

Run ESLint to check code quality:
```bash
npm run lint
```

## Usage

1. Start the development server
2. Paste a YouTube video or channel URL in the input field
3. Click "Analyze comments" to begin processing
4. Wait for the analysis to complete (simulated processing steps)
5. View insights including:
   - Audience signals chart showing cluster distribution
   - Discussion clusters grouped by themes
   - Common ideas and themes
   - Opportunities and signals (feature requests, complaints, questions)

## Design Philosophy

The application features a modern, premium dark theme inspired by Apple and Stripe design aesthetics:
- Deep indigo/violet gradient backgrounds
- Glassmorphism effects with backdrop blur
- Soft shadows and subtle glow accents
- Clean typography with strong visual hierarchy
- Minimal, expressive UI focused on clarity

## Notes

This is a frontend-only implementation. The UI simulates comment analysis processing. To connect to a real backend API, modify the `handleAnalyze` and `simulateProcessing` functions in `App.tsx` to make actual API calls.
