import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    fs: {
      strict: false,
      allow: ['..']  // Allow serving files from parent directory
    }
  },
  publicDir: '../outputs'  // Serve the outputs directory as static files
}) 