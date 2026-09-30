import React from 'react'
import ReactDOM from 'react-dom/client'
import { ThemeProvider } from './theme'
import { ToastProvider } from './components/Toast'
import App from './App'
import './index.css'

// Register Chart.js globally once (the annotation plugin is used by the
// Alignment view's playhead).
import { Chart } from 'chart.js'
import annotationPlugin from 'chartjs-plugin-annotation'
import 'chart.js/auto'
Chart.register(annotationPlugin)

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ThemeProvider>
      <ToastProvider>
        <App />
      </ToastProvider>
    </ThemeProvider>
  </React.StrictMode>,
)
