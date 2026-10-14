/**
 * Saving to and opening files on the user's own disk.
 *
 * In Chrome and Edge on a secure origin (https or localhost) this uses the
 * File System Access API: the user picks a file once and later saves write it
 * in place. Everywhere else (Safari, Firefox, plain-http hosts such as
 * http://fscl2:8200) saves fall back to a browser download and opens to a file
 * input, which is what the app did before.
 */

/** Minimal File System Access API types; lib.dom does not ship the picker functions. */
export interface FilePickerType {
  description: string;
  accept: Record<string, string[]>;
}

export interface WritableFileHandle {
  readonly kind: 'file';
  readonly name: string;
  getFile(): Promise<File>;
  createWritable(): Promise<{ write(data: Blob): Promise<void>; close(): Promise<void> }>;
  queryPermission?(descriptor: { mode: 'readwrite' }): Promise<PermissionState>;
  requestPermission?(descriptor: { mode: 'readwrite' }): Promise<PermissionState>;
}

interface PickerWindow {
  showSaveFilePicker(options: { suggestedName?: string; types?: FilePickerType[]; id?: string }): Promise<WritableFileHandle>;
  showOpenFilePicker(options: { types?: FilePickerType[]; multiple?: false; id?: string }): Promise<WritableFileHandle[]>;
}

export const JSON_FILE: FilePickerType = { description: 'JSON file', accept: { 'application/json': ['.json'] } };
export const CSV_FILE: FilePickerType = { description: 'CSV file', accept: { 'text/csv': ['.csv'] } };
export const MARKDOWN_FILE: FilePickerType = { description: 'Markdown file', accept: { 'text/markdown': ['.md'] } };

/** Why linked saving is unavailable, or null when it works here. */
export function fileAccessUnavailableReason(): string | null {
  if (typeof window === 'undefined') return 'No browser window.';
  if (!window.isSecureContext) {
    return 'This page is served over plain http, so the browser only allows downloads. Open it via https or localhost to save to a file.';
  }
  if (!('showSaveFilePicker' in window)) {
    return 'This browser only supports downloads. Use Chrome or Edge to save to a file.';
  }
  return null;
}

export function supportsFileAccess(): boolean {
  return fileAccessUnavailableReason() === null;
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

/**
 * Ask where to save. Returns null if the user cancelled.
 * Call straight from a click handler: the browser requires a recent user gesture.
 */
export async function pickSaveFile(suggestedName: string, type: FilePickerType): Promise<WritableFileHandle | null> {
  try {
    return await (window as unknown as PickerWindow).showSaveFilePicker({ suggestedName, types: [type] });
  } catch (error) {
    if (isAbort(error)) return null;
    throw error;
  }
}

/** Ask which file to open. Returns null if the user cancelled. */
export async function pickOpenFile(type: FilePickerType): Promise<WritableFileHandle | null> {
  try {
    const [handle] = await (window as unknown as PickerWindow).showOpenFilePicker({ types: [type], multiple: false });
    return handle ?? null;
  } catch (error) {
    if (isAbort(error)) return null;
    throw error;
  }
}

/**
 * Make sure we may write to a handle remembered from earlier. After a reload
 * Chrome asks the user again; that prompt also needs a recent user gesture.
 */
export async function ensureWritePermission(handle: WritableFileHandle): Promise<boolean> {
  if (!handle.queryPermission || !handle.requestPermission) return true;
  if ((await handle.queryPermission({ mode: 'readwrite' })) === 'granted') return true;
  return (await handle.requestPermission({ mode: 'readwrite' })) === 'granted';
}

export async function writeFile(handle: WritableFileHandle, blob: Blob): Promise<void> {
  const writable = await handle.createWritable();
  await writable.write(blob);
  await writable.close();
}

/** Browser download: the fallback where linked saving is unavailable. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

/**
 * One-off export (audit log, DoE CSVs): the Save dialog where supported, so the
 * user picks folder and name; a download otherwise. Nothing is remembered.
 * Returns the saved filename, or null if the user cancelled.
 */
export async function saveFileOnce(
  getBlob: () => Promise<Blob>,
  suggestedName: string,
  type: FilePickerType,
): Promise<string | null> {
  if (!supportsFileAccess()) {
    downloadBlob(await getBlob(), suggestedName);
    return suggestedName;
  }
  const handle = await pickSaveFile(suggestedName, type);
  if (!handle) return null;
  await writeFile(handle, await getBlob());
  return handle.name;
}
