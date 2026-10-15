import { useEffect, useState, useRef } from 'react';
import { Toaster } from 'sonner';
import { toast } from './lib/toast';
import { QueryProvider } from './providers/QueryProvider';
import { VisualizationProvider, useVisualization } from './providers/VisualizationProvider';
import { 
  clearStoredSessionId, 
  getStoredSessionId,
  storeSessionId,
  useCreateSession, 
  useSession,
  useImportSession,
  useUpdateSessionMetadata,
  useSessionMetadata
} from './hooks/api/useSessions';
import { isUntitledSession, sessionFileStem } from './lib/fileNames';
import * as sessionAPI from './api/endpoints/sessions';
import { JSON_FILE, MARKDOWN_FILE, pickOpenFile, saveFileOnce } from './lib/fileAccess';
import { fileLinkStore } from './lib/fileLinks';
import { isSessionChanged, markSessionChanged, markSessionSaved } from './lib/sessionChanges';
import {
  chooseStartupSession,
  lookupSession,
  needsRecoveryBackup,
  recoveryToOffer,
  type RecoveryEntry,
} from './lib/sessionResume';
import { reportSaveOutcome, useLinkedFile, useSessionChanged } from './hooks/useLinkedFile';
import { SplitSaveButton } from './components/ui/SplitSaveButton';
import { SAVE_SHORTCUT, SAVE_AS_SHORTCUT } from './lib/shortcuts';
import { useSessionEvents } from './hooks/useSessionEvents';
import { VariablesPanel } from './features/variables/VariablesPanel';
import { ExperimentsPanel } from './features/experiments/ExperimentsPanel';
import { InitialDesignPanel } from './features/experiments/InitialDesignPanel';
import { OptimalDesignPanel } from './features/experiments/OptimalDesignPanel';
import { GPRPanel } from './features/models/GPRPanel';
import { AcquisitionPanel } from './features/acquisition/AcquisitionPanel';
import { LiveMonitor } from './features/monitoring/LiveMonitor';
import { VisualizationsPanel } from './components/visualizations';
import { TabView } from './components/ui';
import { SessionMetadataDialog } from './components/SessionMetadataDialog';
import { PanelErrorBoundary } from './components/PanelErrorBoundary';
import { useTheme } from './hooks/useTheme';
import { restorePendingSuggestions } from './lib/restoreStaged';
import { Sun, Moon, X, Copy, Check, ArrowUpRight } from 'lucide-react';
import './index.css';

const DOCS_URL = 'https://natlabrockies.github.io/ALchemist/';

/** "14:02" today, "Oct 9, 14:02" on another day. */
function formatSavedAt(iso: string): string {
  const date = new Date(iso);
  const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return date.toDateString() === new Date().toDateString()
    ? time
    : `${date.toLocaleDateString([], { month: 'short', day: 'numeric' })}, ${time}`;
}

function AppContent() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isMonitoringMode, setIsMonitoringMode] = useState<boolean>(false);
  const [showMetadataDialog, setShowMetadataDialog] = useState(false);
  const [copiedSessionId, setCopiedSessionId] = useState(false);
  const [sessionFromUrl, setSessionFromUrl] = useState<boolean>(false);
  const [showRecoveryBanner, setShowRecoveryBanner] = useState(false);
  const [recoverySession, setRecoverySession] = useState<RecoveryEntry | null>(null);
  const [reopening, setReopening] = useState(() => getStoredSessionId() !== null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [joinSessionId, setJoinSessionId] = useState('');
  
  const createSession = useCreateSession();
  const importSession = useImportSession();
  const updateMetadata = useUpdateSessionMetadata(sessionId);
  const { data: session, error: sessionError } = useSession(sessionId);
  const { data: sessionMetadata, isLoading: sessionMetadataLoading } = useSessionMetadata(sessionId);
  const sessionFile = useLinkedFile('session', sessionId);
  const sessionChanged = useSessionChanged(sessionId);
  const [isSavingSession, setIsSavingSession] = useState(false);
  const { theme, toggleTheme } = useTheme();
  const { isVisualizationOpen, closeVisualization, sessionId: vizSessionId } = useVisualization();
  
  // Monitor session events (lock status, experiment updates, model training) via WebSocket
  const { lockStatus } = useSessionEvents(sessionId, (locked) => {
    setIsMonitoringMode(locked);
  });
  
  // Global staged suggestions to mirror desktop main_app.pending_suggestions
  const [pendingSuggestions, setPendingSuggestions] = useState<any[]>([]);

  // Restore pending suggestions from staged experiments API on session load.
  // This effect runs on every sessionId change, so it must ALWAYS resolve to a
  // definitive value — including clearing to [] when the current session has no
  // staged suggestions. Otherwise stale suggestions from a previous session
  // linger in state across "New Session" / session switches (ghost points).
  useEffect(() => {
    if (!sessionId) {
      setPendingSuggestions([]);
      return;
    }

    let cancelled = false;
    const activeSessionId = sessionId;

    restorePendingSuggestions(activeSessionId).then((resolved) => {
      // Guard against a race: if the session changed while we were fetching,
      // don't clobber the newer session's state.
      if (!cancelled && activeSessionId === sessionId) {
        setPendingSuggestions(resolved);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  // On start, open the session named in the URL, else the one this browser
  // last worked on (lib/sessionResume.ts), then offer a recovery backup of any
  // other session. A URL session gets no recovery prompt.
  useEffect(() => {
    let cancelled = false;
    if (new URLSearchParams(window.location.search).get('mode') === 'monitor') {
      setIsMonitoringMode(true);
      console.log('✓ Monitoring mode enabled');
    }
    (async () => {
      const storedId = getStoredSessionId();
      const choice = await chooseStartupSession({
        search: window.location.search,
        storedId,
        lookup: lookupSession,
      });
      if (cancelled) return;
      setReopening(false);
      if (choice.forgetStored) clearStoredSessionId();
      if (choice.sessionId) {
        // A session the user opened while this was checking stays open.
        setSessionId(current => current ?? choice.sessionId);
        if (choice.source === 'url') setSessionFromUrl(true);
        console.log(`✓ Opened session from ${choice.source}: ${choice.sessionId}`);
      } else if (storedId && !choice.forgetStored) {
        toast.warning("Couldn't reopen your last session just now. Reload to try again.");
      }
      if (choice.source === 'url') return;
      try {
        const data = await (await fetch('/api/v1/recovery/list')).json();
        if (cancelled) return;
        const offer = recoveryToOffer(data.recoveries, choice.sessionId);
        if (offer) {
          setRecoverySession(offer);
          setShowRecoveryBanner(true);
          console.log('✓ Found recovery session:', offer);
        }
      } catch (err) {
        console.warn('Failed to check for recovery sessions:', err);
      }
    })();
    return () => { cancelled = true; };
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

  // Silent auto-backup every 30 seconds for crash recovery, while the session
  // has data that is not in its saved file
  useEffect(() => {
    if (!sessionId || !session) return;
    
    const interval = setInterval(() => {
      if (!needsRecoveryBackup(session, isSessionChanged(sessionId))) return;
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
      // Only prompt if the session changed since it was last saved to a file
      if (isSessionChanged(sessionId)) {
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
      const newSession = await createSession.mutateAsync({});
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

  // The Clear button also forgets which files this session saves to
  const handleClearClick = () => {
    if (sessionId) {
      fileLinkStore.remove('session', sessionId);
      fileLinkStore.remove('variables', sessionId);
    }
    handleClearSession();
  };

  // Save the session to its linked file (first time / Save As: ask where).
  // Runs straight from the click or shortcut: the Save dialog needs that gesture,
  // so the suggested name comes from cached metadata rather than a fetch.
  const handleSaveSession = async (saveAs = false) => {
    if (!sessionId || isSavingSession) return;
    setIsSavingSession(true);
    try {
      const outcome = await sessionFile.save({
        saveAs,
        suggestedName: `${sessionFileStem(sessionMetadata?.name, sessionId)}.json`,
        type: JSON_FILE,
        getBlob: () => sessionAPI.exportSession(sessionId),
      });
      reportSaveOutcome(outcome, () => handleSaveSession(true));
      if (outcome.status === 'saved' || outcome.status === 'downloaded') {
        markSessionSaved(sessionId);
        // Clear recovery backup after successful save
        fetch(`/api/v1/sessions/${sessionId}/backup`, { method: 'DELETE' })
          .then(() => console.log('Recovery backup cleared after save'))
          .catch(err => console.warn('Failed to clear recovery backup:', err));
      }
    } catch (error) {
      toast.error('Failed to save session');
      console.error('Error saving session:', error);
    } finally {
      setIsSavingSession(false);
    }
  };

  // ⌘S / Ctrl+S saves, ⇧⌘S / Ctrl+Shift+S is Save As, while a session is open
  const saveShortcutRef = useRef(handleSaveSession);
  saveShortcutRef.current = handleSaveSession;
  useEffect(() => {
    if (!sessionId) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && !event.altKey && event.key.toLowerCase() === 's') {
        event.preventDefault();
        saveShortcutRef.current(event.shiftKey);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [sessionId]);

  // Handle session import
  // Where the browser allows it, open through the file picker so the opened
  // file stays linked and Save writes back to it; otherwise a file input.
  const handleImportClick = async () => {
    if (!sessionFile.supported) {
      fileInputRef.current?.click();
      return;
    }
    try {
      const handle = await pickOpenFile(JSON_FILE);
      if (!handle) return;
      const newSessionId = await importSessionFile(await handle.getFile());
      if (newSessionId) {
        await sessionFile.adopt(newSessionId, handle);
        markSessionSaved(newSessionId);
      }
    } catch (error: any) {
      toast.error('Failed to open session file');
      console.error('Error opening session file:', error);
    }
  };

  const importSessionFile = async (file: File): Promise<string | null> => {
    try {
      const newSession = await importSession.mutateAsync(file);
      setSessionId(newSession.session_id);
      toast.success(`Opened ${file.name}`);
      return newSession.session_id;
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to import session');
      console.error('Error importing session:', error);
      return null;
    }
  };

  const handleFileSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      await importSessionFile(file);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
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
      storeSessionId(data.session_id);
      // Restored work has, by definition, not been saved to a file
      markSessionChanged(data.session_id);
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
      const saved = await saveFileOnce(
        async () => {
          const response = await fetch(`/api/v1/sessions/${sessionId}/audit/export`);
          if (!response.ok) throw new Error('Failed to export audit log');
          return response.blob();
        },
        `${sessionFileStem(sessionMetadata?.name, sessionId)}_audit_log.md`,
        MARKDOWN_FILE,
      );
      if (saved) toast.success(`Saved audit log to ${saved}`);
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
    storeSessionId(trimmed);
    setSessionFromUrl(true); // Prevent auto-clear on 404
    setJoinSessionId('');
  };

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
        <LiveMonitor sessionId={sessionId} />
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
                    <span aria-hidden="true" className="mx-1.5">·</span>
                    <a
                      href={DOCS_URL}
                      target="_blank"
                      rel="noopener noreferrer"
                      title="Open the ALchemist user guide in a new tab"
                      className="inline-flex items-center gap-0.5 text-foreground/80 underline underline-offset-2 decoration-border hover:text-foreground hover:decoration-foreground transition-colors"
                    >
                      User guide
                      <ArrowUpRight className="h-3 w-3" aria-hidden="true" />
                      <span className="sr-only">(opens in a new tab)</span>
                    </a>
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
                        <span className="ml-2 whitespace-nowrap">
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
                    className="whitespace-nowrap text-xs text-muted-foreground hover:text-foreground px-2 py-1 rounded hover:bg-accent"
                    title="Edit session metadata"
                  >
                    Edit Info
                  </button>
                  <button
                    onClick={handleExportAuditLog}
                    className="whitespace-nowrap text-xs text-muted-foreground hover:text-foreground px-2 py-1 rounded hover:bg-accent"
                    title="Export audit log as markdown"
                  >
                    Export Log
                  </button>
                  {sessionFile.supported && (
                    <span
                      className="flex min-w-0 max-w-[9rem] xl:max-w-[14rem] items-center gap-1.5 text-xs text-muted-foreground"
                      title={
                        sessionFile.link
                          ? `${sessionFile.link.name}\nLast saved ${formatSavedAt(sessionFile.link.savedAt)}${sessionChanged ? '\nUnsaved changes since then' : ''}`
                          : 'Save asks where to keep the session file'
                      }
                    >
                      {sessionFile.link ? (
                        <>
                          {sessionChanged && (
                            <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-amber-500" aria-hidden="true" />
                          )}
                          <span className="truncate">{sessionFile.link.name}</span>
                          {sessionChanged && <span className="sr-only">(unsaved changes)</span>}
                        </>
                      ) : (
                        <span className="truncate">Not saved to a file</span>
                      )}
                    </span>
                  )}
                  <SplitSaveButton
                    label="Save"
                    variant="primary"
                    shortcuts
                    showMenu={sessionFile.supported}
                    busy={isSavingSession}
                    onSave={() => handleSaveSession(false)}
                    onSaveAs={() => handleSaveSession(true)}
                    title={
                      !sessionFile.supported
                        ? `${sessionFile.unavailableReason} (${SAVE_SHORTCUT})`
                        : sessionFile.link
                          ? `Save to ${sessionFile.link.name} (${SAVE_SHORTCUT}); Save As… ${SAVE_AS_SHORTCUT}`
                          : `Save to a file (${SAVE_SHORTCUT})`
                    }
                  />
                  <button
                    onClick={handleClearClick}
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
                <PanelErrorBoundary name="Variables and Experiments" key={sessionId}>
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
                </PanelErrorBoundary>
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
                      <PanelErrorBoundary name="Visualizations" key={vizSessionId}>
                        <VisualizationsPanel 
                          sessionId={vizSessionId} 
                          embedded={true}
                        />
                      </PanelErrorBoundary>
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
                      content: (
                        <PanelErrorBoundary name="Model" key={sessionId}>
                          <GPRPanel sessionId={sessionId} />
                        </PanelErrorBoundary>
                      ),
                    },
                    {
                      id: 'acquisition',
                      label: 'Acquisition',
                      content: (
                        <PanelErrorBoundary name="Acquisition" key={sessionId}>
                          <AcquisitionPanel 
                            sessionId={sessionId} 
                            modelBackend={session?.model_trained ? (session as any).model_backend : null} 
                            pendingSuggestions={pendingSuggestions}
                            onStageSuggestions={(s:any[])=>setPendingSuggestions(s)}
                          />
                        </PanelErrorBoundary>
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
                  {reopening
                    ? 'Reopening your last session…'
                    : 'Create a new session or load a previously saved session to begin your optimization workflow.'}
                </p>
              </div>
            </div>
          )}
        </>
      )}
      
      {/* Toast notifications */}
      <Toaster position="bottom-right" theme={theme} richColors closeButton />
      
      {/* Session Metadata Dialog */}
      {showMetadataDialog && sessionId && session && !sessionMetadataLoading && (
        <SessionMetadataDialog
          sessionId={sessionId}
          metadata={
            sessionMetadata
              ? { ...sessionMetadata, name: isUntitledSession(sessionMetadata.name) ? '' : sessionMetadata.name }
              : {}
          }
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
