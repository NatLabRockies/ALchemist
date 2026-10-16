/**
 * Visualization Context - Share visualization state across app
 * Allows GPRPanel to trigger visualizations that render in center panel
 */
import { createContext, useCallback, useContext, useState, type ReactNode } from 'react';

interface VisualizationContextType {
  isVisualizationOpen: boolean;
  sessionId: string | null;
  modelBackend: string | null;
  openVisualization: (sessionId: string, backend: string) => void;
  closeVisualization: () => void;
}

const VisualizationContext = createContext<VisualizationContextType | undefined>(undefined);

export function VisualizationProvider({ children }: { children: ReactNode }) {
  const [isVisualizationOpen, setIsVisualizationOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [modelBackend, setModelBackend] = useState<string | null>(null);

  const openVisualization = useCallback((sid: string, backend: string) => {
    setSessionId(sid);
    setModelBackend(backend);
    setIsVisualizationOpen(true);
  }, []);

  const closeVisualization = useCallback(() => {
    setIsVisualizationOpen(false);
    setSessionId(null);
    setModelBackend(null);
  }, []);

  return (
    <VisualizationContext.Provider
      value={{
        isVisualizationOpen,
        sessionId,
        modelBackend,
        openVisualization,
        closeVisualization,
      }}
    >
      {children}
    </VisualizationContext.Provider>
  );
}

export function useVisualization() {
  const context = useContext(VisualizationContext);
  if (!context) {
    throw new Error('useVisualization must be used within VisualizationProvider');
  }
  return context;
}
