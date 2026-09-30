import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Development proxy: forwards API/stream/vault requests to both the Flask
// backend (port 5000) and the new FastAPI streaming backend (port 8000).
// In production the built SPA is served by FastAPI itself, so all requests
// are same-origin and no proxy is needed.
const FLASK_PORT = 5000
const FASTAPI_PORT = 8000
const FLASK_URL = `http://127.0.0.1:${FLASK_PORT}`
const FASTAPI_URL = `http://127.0.0.1:${FASTAPI_PORT}`

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      // Week 6: FastAPI streaming endpoints (WebSocket + REST)
      '/api/v1': { target: FASTAPI_URL, changeOrigin: true, ws: true },
      // Legacy Flask endpoints
      '/video_feed': { target: FLASK_URL, changeOrigin: true },
      '/stream': { target: FLASK_URL, changeOrigin: true },
      '/api': { target: FLASK_URL, changeOrigin: true },
      '/vault': { target: FLASK_URL, changeOrigin: true },
      '/dataset': { target: FLASK_URL, changeOrigin: true },
      '/upload_jobs': { target: FLASK_URL, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
