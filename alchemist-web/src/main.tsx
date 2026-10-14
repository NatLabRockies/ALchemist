import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { apiClient } from './api/client'
import { installSessionChangeObservers } from './lib/sessionChanges'

installSessionChangeObservers(apiClient)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
