import react from '@vitejs/plugin-react'
import path from 'path'
import { defineConfig } from 'vite'

const backendTarget = process.env.VITE_BACKEND_URL || 'http://localhost:8000'

const proxyPaths = [
  '/hedge-fund',
  '/flows',
  '/storage',
  '/ollama',
  '/language-models',
  '/api-keys',
  '/ping',
]

const proxy = Object.fromEntries(
  proxyPaths.map((p) => [
    p,
    {
      target: backendTarget,
      changeOrigin: true,
      ws: true,
    },
  ])
)

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5000,
    strictPort: true,
    allowedHosts: true,
    proxy,
  },
  preview: {
    host: '0.0.0.0',
    port: 5000,
    strictPort: true,
    allowedHosts: true,
  },
})
