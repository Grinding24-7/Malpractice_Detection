import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Development proxy: forwards API/stream/vault requests to the Flask backend
// (default port 5000). In production the built SPA is served by Flask itself,
// so all requests are same-origin and no proxy is needed.
const BACKEND_PORT = 5000
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      // MJPEG classroom CCTV stream (backend-annotated frames)
      '/video_feed': { target: BACKEND_URL, changeOrigin: true },
      // Webcam inference endpoint
      '/stream': { target: BACKEND_URL, changeOrigin: true },
      // REST API (telemetry, evidence, classrooms, record_label)
      '/api': { target: BACKEND_URL, changeOrigin: true },
      // Evidence clip + dataset video media
      '/vault': { target: BACKEND_URL, changeOrigin: true },
      '/dataset': { target: BACKEND_URL, changeOrigin: true },
      // Offline analysis jobs (annotated video + sidecar JSON)
      '/upload_jobs': { target: BACKEND_URL, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
