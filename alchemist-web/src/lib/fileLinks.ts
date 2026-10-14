/**
 * Which file on disk each session's Save writes to.
 *
 * File handles can only be stored in IndexedDB (not localStorage), and only in
 * this browser: a link never reaches the server or another machine. Every call
 * tolerates IndexedDB being unavailable (private windows, blocked storage); the
 * link then lasts until the page reloads.
 */
import type { WritableFileHandle } from './fileAccess';

export type LinkKind = 'session' | 'variables';

export interface FileLink {
  handle: WritableFileHandle;
  /** Filename, kept so the header can show it before the handle is used. */
  name: string;
  /** ISO time of the last successful save or open through this link. */
  savedAt: string;
}

export interface FileLinkStore {
  get(kind: LinkKind, sessionId: string): Promise<FileLink | undefined>;
  set(kind: LinkKind, sessionId: string, link: FileLink): Promise<void>;
  remove(kind: LinkKind, sessionId: string): Promise<void>;
}

const DB_NAME = 'alchemist-file-links';
const STORE = 'links';
const key = (kind: LinkKind, sessionId: string) => `${kind}:${sessionId}`;

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function withStore<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openDb();
  try {
    return await new Promise<T>((resolve, reject) => {
      const request = run(db.transaction(STORE, mode).objectStore(STORE));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  } finally {
    db.close();
  }
}

/** In-memory store; also the fallback when IndexedDB is unavailable. */
export function createMemoryFileLinkStore(): FileLinkStore {
  const links = new Map<string, FileLink>();
  return {
    async get(kind, sessionId) { return links.get(key(kind, sessionId)); },
    async set(kind, sessionId, link) { links.set(key(kind, sessionId), link); },
    async remove(kind, sessionId) { links.delete(key(kind, sessionId)); },
  };
}

function createIndexedDbFileLinkStore(): FileLinkStore {
  const memory = createMemoryFileLinkStore();
  return {
    async get(kind, sessionId) {
      try {
        return (await withStore<FileLink | undefined>('readonly', s => s.get(key(kind, sessionId)))) ?? memory.get(kind, sessionId);
      } catch {
        return memory.get(kind, sessionId);
      }
    },
    async set(kind, sessionId, link) {
      await memory.set(kind, sessionId, link);
      try { await withStore('readwrite', s => s.put(link, key(kind, sessionId))); } catch { /* memory copy stands */ }
    },
    async remove(kind, sessionId) {
      await memory.remove(kind, sessionId);
      try { await withStore('readwrite', s => s.delete(key(kind, sessionId))); } catch { /* nothing persisted */ }
    },
  };
}

export const fileLinkStore: FileLinkStore =
  typeof indexedDB === 'undefined' ? createMemoryFileLinkStore() : createIndexedDbFileLinkStore();
