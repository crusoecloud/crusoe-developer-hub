/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'host-a': '#8B5CF6',
        'host-b': '#10B981',
      },
    },
  },
  plugins: [],
}
