/**
 * React side of linked-file saving: which file a session's Save (or its
 * variables' Save File) writes to, and whether the session has unsaved changes.
 */
import { useCallback, useEffect, useState, useSyncExternalStore } from 'react';
import { toast } from '../lib/toast';
import { supportsFileAccess, fileAccessUnavailableReason, type WritableFileHandle } from '../lib/fileAccess';
import { fileLinkStore, type LinkKind } from '../lib/fileLinks';
import { saveToLinkedFile, type LinkedSaveRequest, type SaveOutcome } from '../lib/linkedSave';
import { isSessionChanged, subscribeSessionChanges } from '../lib/sessionChanges';

export interface LinkInfo {
  name: string;
  savedAt: string;
}

export function useLinkedFile(kind: LinkKind, sessionId: string | null) {
  const [link, setLink] = useState<LinkInfo | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLink(null);
    if (sessionId) {
      fileLinkStore.get(kind, sessionId).then(found => {
        if (!cancelled && found) setLink({ name: found.name, savedAt: found.savedAt });
      });
    }
    return () => { cancelled = true; };
  }, [kind, sessionId]);

  const save = useCallback(
    async (request: Omit<LinkedSaveRequest, 'kind' | 'sessionId'>): Promise<SaveOutcome> => {
      if (!sessionId) return { status: 'cancelled' };
      const outcome = await saveToLinkedFile({ ...request, kind, sessionId });
      if (outcome.status === 'saved') {
        const stored = await fileLinkStore.get(kind, sessionId);
        if (stored) setLink({ name: stored.name, savedAt: stored.savedAt });
      }
      return outcome;
    },
    [kind, sessionId],
  );

  /** Link a file the user just opened (Load Session, Load File). */
  const adopt = useCallback(
    async (targetSessionId: string, handle: WritableFileHandle) => {
      const info = { name: handle.name, savedAt: new Date().toISOString() };
      await fileLinkStore.set(kind, targetSessionId, { handle, ...info });
      if (targetSessionId === sessionId) setLink(info);
    },
    [kind, sessionId],
  );

  const forget = useCallback(async (targetSessionId: string) => {
    await fileLinkStore.remove(kind, targetSessionId);
    if (targetSessionId === sessionId) setLink(null);
  }, [kind, sessionId]);

  return { link, save, adopt, forget, supported: supportsFileAccess(), unavailableReason: fileAccessUnavailableReason() };
}

export function useSessionChanged(sessionId: string | null): boolean {
  return useSyncExternalStore(subscribeSessionChanges, () => (sessionId ? isSessionChanged(sessionId) : false));
}

/** The toast for a save. Cancelled saves say nothing. */
export function reportSaveOutcome(outcome: SaveOutcome, onSaveAs: () => void): void {
  switch (outcome.status) {
    case 'saved':
      toast.success(`Saved to ${outcome.name}`);
      break;
    case 'downloaded':
      toast.success(`Downloaded ${outcome.name}`);
      break;
    case 'failed':
      toast.error(
        outcome.reason === 'permission'
          ? `ALchemist isn't allowed to write ${outcome.name ?? 'the file'}`
          : `Couldn't write ${outcome.name ?? 'the file'}. It may have been moved, renamed or deleted.`,
        { action: { label: 'Save As…', onClick: onSaveAs } },
      );
      break;
  }
}
