/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark-bg': '#0a0a0f',
        'dark-surface': '#13131a',
        'dark-elevated': '#1a1a24',
        'dark-accent': '#6366f1',
        'dark-accent-light': '#818cf8',
        'dark-purple': '#7c3aed',
        'dark-indigo': '#4f46e5',
        'text-primary': '#f5f5f7',
        'text-secondary': '#a1a1aa',
        'text-tertiary': '#71717a',
        'glow-blue': '#3b82f6',
        'glow-purple': '#8b5cf6',
      },
      fontFamily: {
        sans: ['Inter Tight', 'Archivo', 'Instrument Sans', 'Google Sans', '-apple-system', 'BlinkMacSystemFont', '"SF Pro Display"', '"SF Pro Text"', '"Segoe UI"', 'Roboto', '"Helvetica Neue"', 'Arial', 'sans-serif'],
      },
      backgroundImage: {
        'gradient-dark': 'linear-gradient(135deg, #0a0a0f 0%, #1a1a24 50%, #13131a 100%)',
        'gradient-accent': 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #6366f1 100%)',
        'gradient-hero': 'linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%)',
      },
      boxShadow: {
        'glow-blue': '0 0 20px rgba(59, 130, 246, 0.3)',
        'glow-purple': '0 0 20px rgba(139, 92, 246, 0.3)',
        'glass': '0 8px 32px rgba(0, 0, 0, 0.3)',
        'glass-sm': '0 4px 16px rgba(0, 0, 0, 0.2)',
      },
      backdropBlur: {
        'glass': '20px',
      },
    },
  },
  plugins: [],
}
