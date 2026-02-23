import { useEffect, useState, useRef } from 'react';
import { Toaster, toast } from 'sonner';
import { QueryProvider } from './providers/QueryProvider';
import { VisualizationProvider, useVisualization } from './providers/VisualizationProvider';
import { 
  clearStoredSessionId, 
  useCreateSession, 
  useSession,
  useExportSession,
  useImportSession,
  useUpdateSessionMetadata
} from './hooks/api/useSessions';
import { useSessionEvents } from './hooks/useSessionEvents';
import { VariablesPanel } from './features/variables/VariablesPanel';
import { ExperimentsPanel } from './features/experiments/ExperimentsPanel';
import { InitialDesignPanel } from './features/experiments/InitialDesignPanel';
import { OptimalDesignPanel } from './features/experiments/OptimalDesignPanel';
import { GPRPanel } from './features/models/GPRPanel';
import { AcquisitionPanel } from './features/acquisition/AcquisitionPanel';
import { MonitoringDashboard } from './features/monitoring/MonitoringDashboard';
import { VisualizationsPanel } from './components/visualizations';
import { TabView } from './components/ui';
import { SessionMetadataDialog } from './components/SessionMetadataDialog';
import { useTheme } from './hooks/useTheme';
import { Sun, Moon, X, Copy, Check } from 'lucide-react';
import './index.css';

function AppContent() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isMonitoringMode, setIsMonitoringMode] = useState<boolean>(false);
  const [showMetadataDialog, setShowMetadataDialog] = useState(false);
  const [copiedSessionId, setCopiedSessionId] = useState(false);
  const [sessionFromUrl, setSessionFromUrl] = useState<boolean>(false);
  const [showRecoveryBanner, setShowRecoveryBanner] = useState(false);
  const [recoverySession, setRecoverySession] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const urlSessionRef = useRef<string | null>(null);
  const [joinSessionId, setJoinSessionId] = useState('');
  
  const createSession = useCreateSession();
  const exportSession = useExportSession();
  const importSession = useImportSession();
  const updateMetadata = useUpdateSessionMetadata(sessionId);
  const { data: session, error: sessionError } = useSession(sessionId);
  const { theme, toggleTheme } = useTheme();
  const { isVisualizationOpen, closeVisualization, sessionId: vizSessionId } = useVisualization();
  
  // Monitor session events (lock status, experiment updates, model training) via WebSocket
  const { lockStatus } = useSessionEvents(sessionId, (locked) => {
    setIsMonitoringMode(locked);
  });
  
  // Global staged suggestions to mirror desktop main_app.pending_suggestions
  const [pendingSuggestions, setPendingSuggestions] = useState<any[]>([]);

  // Restore pending suggestions from staged experiments API on session load
  useEffect(() => {
    if (!sessionId) return;
    
    async function restoreStagedExperiments() {
      try {
        // First try the staged experiments API (preferred)
        const stagedResponse = await fetch(`/api/v1/sessions/${sessionId}/experiments/staged`);
        if (stagedResponse.ok) {
          const stagedData = await stagedResponse.json();
          if (stagedData.experiments && stagedData.experiments.length > 0) {
            // API now returns clean experiments + reason separately
            // Tag each experiment with the reason for the Add Point dialog
            const reason = stagedData.reason || 'Staged';
            const taggedExperiments = stagedData.experiments.map((exp: any) => ({
              ...exp,
              _reason: reason  // UI-only metadata for dialog auto-fill
            }));
            setPendingSuggestions(taggedExperiments);
            console.log(`✓ Restored ${stagedData.experiments.length} staged experiments from API (reason: ${reason})`);
            return;
          }
        }
        
        // Fallback: check audit log for backward compatibility
        const auditResponse = await fetch(`/api/v1/sessions/${sessionId}/audit?entry_type=acquisition_locked`);
        if (!auditResponse.ok) return;
        
        const data = await auditResponse.json();
        if (data.entries && data.entries.length > 0) {
          // Get latest acquisition entry
          const latestAcq = data.entries[data.entries.length - 1];
          const suggestions = latestAcq.parameters?.suggestions || [];
          
          if (suggestions.length > 0) {
            // Tag suggestions with strategy for reason auto-fill
            const strategy = latestAcq.parameters?.strategy || 'Acquisition';
            const taggedSuggestions = suggestions.map((s: any) => ({
              ...s,
              _reason: strategy,
              Iteration: latestAcq.parameters?.iteration
            }));
            setPendingSuggestions(taggedSuggestions);
            console.log(`✓ Restored ${suggestions.length} pending suggestions from audit log (fallback)`);
          }
        }
      } catch (e) {
        console.error('Failed to restore staged experiments:', e);
      }
    }
    
    restoreStagedExperiments();
  }, [sessionId]);

  // Check URL parameters on mount (always takes priority over recovery)
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const urlSessionId = urlParams.get('session');
    if (urlSessionId) {
      urlSessionRef.current = urlSessionId;
      setSessionId(urlSessionId);
      setSessionFromUrl(true);
      console.log(`✓ Loaded session from URL: ${urlSessionId}`);
    }
    const monitorParam = urlParams.get('mode');
    if (monitorParam === 'monitor') {
      setIsMonitoringMode(true);
      console.log('✓ Monitoring mode enabled');
    }
  }, []);

  // Check for recovery sessions on startup — skip if URL session is present
  useEffect(() => {
    if (urlSessionRef.current) return; // URL intent takes priority
    fetch('/api/v1/recovery/list')
      .then(res => res.json())
      .then(data => {
        if (data.recoveries && data.recoveries.length > 0) {
          setRecoverySession(data.recoveries[0]);
          setShowRecoveryBanner(true);
          console.log('✓ Found recovery session:', data.recoveries[0]);
        }
      })
      .catch(err => console.warn('Failed to check for recovery sessions:', err));
  }, []);

  // Auto-clear invalid session (but not if it came from URL - let user see the error)
  useEffect(() => {
    if (sessionError && sessionId && !sessionFromUrl) {
      toast.error('Session not found. Please create a new session.');
      handleClearSession();
    } else if (sessionError && sessionFromUrl) {
      // Show error but don't clear - might be loading or user wants to see the issue
      console.warn('Session from URL not found or error loading:', sessionError);
    }
  }, [sessionError, sessionId, sessionFromUrl]);

  // Silent auto-backup every 30 seconds for crash recovery
  useEffect(() => {
    if (!sessionId) return;
    
    // Only backup if session has data
    if (!session || (session.variable_count === 0 && session.experiment_count === 0)) {
      return;
    }
    
    const interval = setInterval(() => {
      fetch(`/api/v1/sessions/${sessionId}/backup`, { method: 'POST' })
        .then(() => console.debug('Recovery backup saved'))
        .catch(err => console.warn('Recovery backup failed:', err));
    }, 30000); // Every 30 seconds
    
    return () => clearInterval(interval);
  }, [sessionId, session]);

  // Prompt to save on page close/reload if session has data
  useEffect(() => {
    if (!sessionId) return;
    
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      // Only prompt if session has data
      if (session && (session.variable_count > 0 || session.experiment_count > 0)) {
        e.preventDefault();
        e.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
        return e.returnValue;
      }
    };
    
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [sessionId, session]);

  // Handle session creation
  const handleCreateSession = async () => {
    try {
      const newSession = await createSession.mutateAsync({ ttl_hours: 24 });
      setSessionId(newSession.session_id);
      toast.success('Session created successfully!');
      // Show metadata dialog for new session
      setShowMetadataDialog(true);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to create session');
      console.error('Error creating session:', error);
    }
  };

  // Handle clearing/resetting session
  const handleClearSession = () => {
    clearStoredSessionId();
    setSessionId(null);
    toast.info('Session cleared');
  };

  // Handle session export
  const handleExportSession = async () => {
    if (!sessionId) return;
    try {
      await exportSession.mutateAsync({ sessionId, serverSide: false });
      
      // Clear recovery backup after successful save
      fetch(`/api/v1/sessions/${sessionId}/backup`, { method: 'DELETE' })
        .then(() => console.log('Recovery backup cleared after save'))
        .catch(err => console.warn('Failed to clear recovery backup:', err));
      
      toast.success('Session saved to Downloads folder!');
    } catch (error: any) {
      toast.error('Failed to save session');
      console.error('Error exporting session:', error);
    }
  };

  // Handle session import
  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      try {
        const newSession = await importSession.mutateAsync(file);
        setSessionId(newSession.session_id);
        toast.success('Session imported successfully!');
        // Reset file input
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      } catch (error: any) {
        toast.error(error.response?.data?.detail || 'Failed to import session');
        console.error('Error importing session:', error);
      }
    }
  };

  // Handle recovery session restore
  const handleRestoreRecovery = async () => {
    if (!recoverySession) return;
    try {
      const response = await fetch(`/api/v1/recovery/${recoverySession.session_id}/restore`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Failed to restore recovery session');
      
      const data = await response.json();
      setSessionId(data.session_id);
      setShowRecoveryBanner(false);
      setRecoverySession(null);
      toast.success('Session restored from recovery backup!');
    } catch (error: any) {
      toast.error('Failed to restore recovery session');
      console.error('Error restoring recovery:', error);
    }
  };

  // Handle recovery dismissal
  const handleDismissRecovery = async () => {
    if (!recoverySession) return;
    try {
      // Delete the recovery backup
      await fetch(`/api/v1/sessions/${recoverySession.session_id}/backup`, {
        method: 'DELETE'
      });
      setShowRecoveryBanner(false);
      setRecoverySession(null);
      toast.info('Recovery session discarded');
    } catch (error: any) {
      console.error('Error dismissing recovery:', error);
      // Still hide banner even if delete fails
      setShowRecoveryBanner(false);
      setRecoverySession(null);
    }
  };

  // Handle audit log export (matches desktop File → Export Audit Log)
  const handleExportAuditLog = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`/api/v1/sessions/${sessionId}/audit/export`);
      if (!response.ok) throw new Error('Failed to export audit log');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `audit_log_${sessionId.substring(0, 8)}.md`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      toast.success('Audit log exported successfully!');
    } catch (error: any) {
      toast.error('Failed to export audit log');
      console.error('Error exporting audit log:', error);
    }
  };

  // Handle copy session ID to clipboard
  const handleCopySessionId = async () => {
    if (!sessionId) return;
    try {
      await navigator.clipboard.writeText(sessionId);
      setCopiedSessionId(true);
      toast.success('Session ID copied to clipboard!');
      setTimeout(() => setCopiedSessionId(false), 2000);
    } catch (error) {
      toast.error('Failed to copy session ID');
      console.error('Error copying session ID:', error);
    }
  };

  // Handle joining an existing session by ID
  const handleJoinSession = () => {
    const trimmed = joinSessionId.trim();
    if (!trimmed) return;
    setSessionId(trimmed);
    setSessionFromUrl(true); // Prevent auto-clear on 404
    setJoinSessionId('');
  };

  // Debug: Log state before render
  console.log('=== Render State ===');
  console.log('sessionId:', sessionId);
  console.log('isMonitoringMode:', isMonitoringMode);
  console.log('Should show monitoring:', isMonitoringMode && sessionId);

  return (
    <div className="h-screen bg-background flex flex-col overflow-hidden">
      {/* Recovery Banner */}
      {showRecoveryBanner && recoverySession && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border-b border-yellow-200 dark:border-yellow-800">
          <div className="max-w-7xl mx-auto px-6 py-4">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <svg className="w-5 h-5 text-yellow-600 dark:text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <h3 className="text-sm font-semibold text-yellow-900 dark:text-yellow-100">
                    Unsaved work detected
                  </h3>
                </div>
                <p className="text-sm text-yellow-800 dark:text-yellow-200 mb-2">
                  We found a session from {new Date(recoverySession.backup_time).toLocaleString()} that wasn't saved.
                  Would you like to restore it?
                </p>
                <div className="text-xs text-yellow-700 dark:text-yellow-300 space-y-1">
                  <div><strong>Session:</strong> {recoverySession.session_name || 'Untitled Session'}</div>
                  <div>
                    <strong>Data:</strong> {recoverySession.n_variables} variables, {recoverySession.n_experiments} experiments
                    {recoverySession.model_trained && ', trained model'}
                  </div>
                </div>
              </div>
              <div className="flex gap-2 ml-4">
                <button
                  onClick={handleRestoreRecovery}
                  className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white text-sm font-medium rounded-md transition-colors"
                >
                  Restore Session
                </button>
                <button
                  onClick={handleDismissRecovery}
                  className="px-4 py-2 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 text-sm font-medium rounded-md transition-colors"
                >
                  Discard
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Monitoring Mode - Show dedicated dashboard */}
      {isMonitoringMode && sessionId ? (
        <MonitoringDashboard sessionId={sessionId} pollingInterval={90000} />
      ) : (
        <>
          {/* Header - Always visible */}
          <header className="border-b bg-card px-6 py-1 flex-shrink-0">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="flex flex-col gap-0.5">
                  <img 
                    src={theme === 'dark' ? '/NEW_LOGO_DARK.png' : '/NEW_LOGO_LIGHT.png'} 
                    alt="ALchemist" 
                    className="h-auto"
                    style={{ width: '250px' }}
                  />
                  <p className="text-xs text-muted-foreground">
                    Active Learning Toolkit for Chemical and Materials Research
                  </p>
                </div>
                
                {/* Theme Toggle */}
                <button
                  onClick={toggleTheme}
                  className="p-2 rounded-md hover:bg-accent transition-colors"
                  title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                >
                  {theme === 'dark' ? (
                    <Sun className="h-5 w-5 text-muted-foreground hover:text-foreground" />
                  ) : (
                    <Moon className="h-5 w-5 text-muted-foreground hover:text-foreground" />
                  )}
                </button>
              </div>
              
              {/* Session Controls */}
              {sessionId ? (
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2">
                    <div className="text-sm text-muted-foreground">
                      <code className="bg-muted px-2 py-1 rounded text-xs">
                        {sessionId.substring(0, 8)}
                      </code>
                      {session && (
                        <span className="ml-2">
                          {session.variable_count}V · {session.experiment_count}E
                        </span>
                      )}
                      {lockStatus?.locked && (
                        <span className="ml-2 inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-blue-500/10 text-blue-500 border border-blue-500/20">
                          <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
                          </svg>
                          {lockStatus.locked_by}
                        </span>
                      )}
                    </div>
                    <button
                      onClick={handleCopySessionId}
                      className="p-1.5 rounded hover:bg-accent transition-colors border border-transparent hover:border-border"
                      title="Copy full session ID to clipboard"
                    >
                      {copiedSessionId ? (
                        <Check className="h-4 w-4 text-green-500" />
                      ) : (
                        <Copy className="h-4 w-4 text-muted-foreground hover:text-foreground" />
                      )}
                    </button>
                  </div>
                  <button
                    onClick={() => setShowMetadataDialog(true)}
                    className="text-xs text-muted-foreground hover:text-foreground px-2 py-1 rounded hover:bg-accent"
                    title="Edit session metadata"
                  >
                    Edit Info
                  </button>
                  <button
                    onClick={handleExportAuditLog}
                    className="text-xs text-muted-foreground hover:text-foreground px-2 py-1 rounded hover:bg-accent"
                    title="Export audit log as markdown"
                  >
                    Export Log
                  </button>
                  <button
                    onClick={handleExportSession}
                    disabled={exportSession.isPending}
                    className="text-xs bg-primary text-primary-foreground px-3 py-1.5 rounded hover:bg-primary/90 transition-colors disabled:opacity-50"
                  >
                    Save
                  </button>
                  <button
                    onClick={handleClearSession}
                    className="text-xs text-destructive hover:text-destructive/80 px-3 py-1.5 border border-destructive/30 rounded hover:bg-destructive/10 transition-colors"
                  >
                    Clear
                  </button>
                </div>
              ) : (
                <div className="flex gap-2">
                  <button 
                    onClick={handleCreateSession}
                    disabled={createSession.isPending}
                    className="text-sm bg-primary text-primary-foreground px-4 py-2 rounded hover:bg-primary/90 disabled:opacity-50"
                  >
                    {createSession.isPending ? 'Creating...' : 'New Session'}
                  </button>
                  <button 
                    onClick={handleImportClick}
                    disabled={importSession.isPending}
                    className="text-sm bg-secondary text-secondary-foreground px-4 py-2 rounded hover:bg-secondary/90 disabled:opacity-50"
                  >
                    {importSession.isPending ? 'Loading...' : 'Load Session'}
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".json"
                    onChange={handleFileSelected}
                    className="hidden"
                  />
                  <div className="flex gap-2 items-center ml-4 pl-4 border-l border-border">
                    <input
                      type="text"
                      value={joinSessionId}
                      onChange={(e) => setJoinSessionId(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleJoinSession()}
                      placeholder="Paste session ID..."
                      className="text-sm px-3 py-2 rounded border border-input bg-background w-64"
                    />
                    <button
                      onClick={handleJoinSession}
                      disabled={!joinSessionId.trim()}
                      className="text-sm bg-accent text-accent-foreground px-4 py-2 rounded hover:bg-accent/80 disabled:opacity-50"
                    >
                      Join
                    </button>
                  </div>
                </div>
              )}
            </div>
          </header>

          {/* Main Content Area - 3 Column Desktop Layout */}
          {sessionId ? (
            <div className="flex-1 flex overflow-hidden">
              {/* LEFT SIDEBAR - Variables & Experiments (fixed width, increased for better readability) */}
              <div className="w-[580px] flex-shrink-0 overflow-y-auto border-r bg-card p-4 space-y-4">
                <VariablesPanel sessionId={sessionId} />
                <ExperimentsPanel 
                  sessionId={sessionId} 
                  pendingSuggestions={pendingSuggestions}
                  onStageSuggestions={(s:any[])=>setPendingSuggestions(s)}
                />
                <InitialDesignPanel 
                  sessionId={sessionId}
                  onStageSuggestions={(s:any[])=>setPendingSuggestions(s)}
                />
                <OptimalDesignPanel 
                  sessionId={sessionId}
                  onStageSuggestions={(s:any[])=>setPendingSuggestions(s)}
                />
              </div>

              {/* CENTER - Visualization Area (expandable) */}
              <div className="flex-1 flex flex-col bg-background">
                {isVisualizationOpen && vizSessionId ? (
                  <>
                    {/* Visualization Header */}
                    <div className="border-b bg-card px-4 py-3 flex items-center justify-between">
                      <h3 className="font-semibold">Model Visualizations</h3>
                      <button
                        onClick={closeVisualization}
                        className="p-1 rounded hover:bg-accent transition-colors"
                        title="Close visualizations"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                    
                    {/* Embedded Visualizations */}
                    <div className="flex-1 overflow-auto">
                      <VisualizationsPanel 
                        sessionId={vizSessionId} 
                        embedded={true}
                      />
                    </div>
                  </>
                ) : (
                  <div className="flex-1 flex items-center justify-center p-6">
                    <div className="text-center text-muted-foreground">
                      <div className="text-6xl mb-4">📊</div>
                      <p className="text-lg font-medium mb-2">Visualization Panel</p>
                      <p className="text-sm">
                        Train a model to see visualizations here
                      </p>
                      <p className="text-xs mt-2 text-muted-foreground/60">
                        Plots will be embedded in this panel
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* RIGHT PANEL - Model & Acquisition Tabs (fixed width) */}
              <div className="w-[320px] flex-shrink-0 border-l bg-card">
                <TabView
                  tabs={[
                    {
                      id: 'model',
                      label: 'Model',
                      content: <GPRPanel sessionId={sessionId} />,
                    },
                    {
                      id: 'acquisition',
                      label: 'Acquisition',
                      content: (
                          <AcquisitionPanel 
                            sessionId={sessionId} 
                            modelBackend={session?.model_trained ? (session as any).model_backend : null} 
                            pendingSuggestions={pendingSuggestions}
                            onStageSuggestions={(s:any[])=>setPendingSuggestions(s)}
                          />
                        ),
                    },
                  ]}
                  defaultTab="model"
                />
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center bg-background">
              <div className="text-center max-w-md">
                <div className="text-6xl mb-4">🧪</div>
                <h2 className="text-2xl font-bold mb-4">Welcome to ALchemist</h2>
                <p className="text-muted-foreground mb-6">
                  Create a new session or load a previously saved session to begin your optimization workflow.
                </p>
              </div>
            </div>
          )}
        </>
      )}
      
      {/* Toast notifications */}
      <Toaster position="top-right" richColors />
      
      {/* Session Metadata Dialog */}
      {showMetadataDialog && sessionId && session && (
        <SessionMetadataDialog
          sessionId={sessionId}
          metadata={session.metadata || {}}
          onSave={async (metadata) => {
            try {
              await updateMetadata.mutateAsync(metadata);
              toast.success('Session metadata updated');
              setShowMetadataDialog(false);
            } catch (e: any) {
              toast.error('Failed to update metadata: ' + (e?.message || String(e)));
            }
          }}
          onCancel={() => setShowMetadataDialog(false)}
        />
      )}
    </div>
  );
}

function App() {
  return (
    <QueryProvider>
      <VisualizationProvider>
        <AppContent />
      </VisualizationProvider>
    </QueryProvider>
  );
}

export default App;
