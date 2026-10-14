/**
 * Save to the file a session (or its variables) is linked to.
 *
 * - Linked: write that file in place.
 * - Not linked, or Save As: ask where, write, and link the chosen file.
 * - No File System Access support: download, as before.
 *
 * The picker and Chrome's permission prompt both need a recent user gesture,
 * so they run before the (network) export; the blob is fetched afterwards.
 */
import {
  downloadBlob,
  ensureWritePermission,
  pickSaveFile,
  supportsFileAccess,
  writeFile,
  type FilePickerType,
  type WritableFileHandle,
} from './fileAccess';
import { fileLinkStore, type FileLinkStore, type LinkKind } from './fileLinks';

export type SaveOutcome =
  | { status: 'saved'; name: string }
  | { status: 'downloaded'; name: string }
  | { status: 'cancelled' }
  | { status: 'failed'; name?: string; reason: 'permission' | 'write'; error?: unknown };

export interface SaveIO {
  supported: () => boolean;
  pickSave: (suggestedName: string, type: FilePickerType) => Promise<WritableFileHandle | null>;
  ensurePermission: (handle: WritableFileHandle) => Promise<boolean>;
  write: (handle: WritableFileHandle, blob: Blob) => Promise<void>;
  download: (blob: Blob, filename: string) => void;
  now: () => Date;
}

const browserIO: SaveIO = {
  supported: supportsFileAccess,
  pickSave: pickSaveFile,
  ensurePermission: ensureWritePermission,
  write: writeFile,
  download: downloadBlob,
  now: () => new Date(),
};

export interface LinkedSaveRequest {
  kind: LinkKind;
  sessionId: string;
  /** Used for the Save dialog and for downloads. */
  suggestedName: string;
  type: FilePickerType;
  /** Ignore any existing link and ask where to save. */
  saveAs: boolean;
  getBlob: () => Promise<Blob>;
}

export async function saveToLinkedFile(
  request: LinkedSaveRequest,
  store: FileLinkStore = fileLinkStore,
  io: SaveIO = browserIO,
): Promise<SaveOutcome> {
  const { kind, sessionId, suggestedName, type, saveAs, getBlob } = request;

  if (!io.supported()) {
    io.download(await getBlob(), suggestedName);
    return { status: 'downloaded', name: suggestedName };
  }

  const existing = saveAs ? undefined : (await store.get(kind, sessionId))?.handle;
  let handle: WritableFileHandle;
  if (existing) {
    if (!(await io.ensurePermission(existing).catch(() => false))) {
      return { status: 'failed', name: existing.name, reason: 'permission' };
    }
    handle = existing;
  } else {
    const picked = await io.pickSave(suggestedName, type);
    if (!picked) return { status: 'cancelled' };
    handle = picked;
  }

  try {
    await io.write(handle, await getBlob());
  } catch (error) {
    return { status: 'failed', name: handle.name, reason: 'write', error };
  }
  await store.set(kind, sessionId, { handle, name: handle.name, savedAt: io.now().toISOString() });
  return { status: 'saved', name: handle.name };
}
