import { describe, it, expect, vi, beforeEach } from 'vitest';
import { saveToLinkedFile, type SaveIO, type LinkedSaveRequest } from './linkedSave';
import { createMemoryFileLinkStore, type FileLinkStore } from './fileLinks';
import { JSON_FILE, type WritableFileHandle } from './fileAccess';

const ID = 'session-1';
const handle = (name: string): WritableFileHandle =>
  ({ kind: 'file', name, getFile: vi.fn(), createWritable: vi.fn() }) as unknown as WritableFileHandle;

function makeIO(overrides: Partial<SaveIO> = {}): SaveIO {
  return {
    supported: () => true,
    pickSave: vi.fn().mockResolvedValue(handle('picked.json')),
    ensurePermission: vi.fn().mockResolvedValue(true),
    write: vi.fn().mockResolvedValue(undefined),
    download: vi.fn(),
    now: () => new Date('2026-10-14T12:00:00Z'),
    ...overrides,
  };
}

const request = (overrides: Partial<LinkedSaveRequest> = {}): LinkedSaveRequest => ({
  kind: 'session',
  sessionId: ID,
  suggestedName: 'C1 screen.json',
  type: JSON_FILE,
  saveAs: false,
  getBlob: vi.fn().mockResolvedValue(new Blob(['{}'])),
  ...overrides,
});

describe('saveToLinkedFile', () => {
  let store: FileLinkStore;
  beforeEach(() => { store = createMemoryFileLinkStore(); });

  it('asks where on the first save, writes, and links the chosen file', async () => {
    const io = makeIO();
    const outcome = await saveToLinkedFile(request(), store, io);
    expect(io.pickSave).toHaveBeenCalledWith('C1 screen.json', JSON_FILE);
    expect(io.write).toHaveBeenCalledTimes(1);
    expect(outcome).toEqual({ status: 'saved', name: 'picked.json' });
    expect(await store.get('session', ID)).toMatchObject({ name: 'picked.json', savedAt: '2026-10-14T12:00:00.000Z' });
  });

  it('writes a linked file in place without asking', async () => {
    const linked = handle('linked.json');
    await store.set('session', ID, { handle: linked, name: 'linked.json', savedAt: 'x' });
    const io = makeIO();
    const outcome = await saveToLinkedFile(request(), store, io);
    expect(io.pickSave).not.toHaveBeenCalled();
    expect(io.ensurePermission).toHaveBeenCalledWith(linked);
    expect(io.write).toHaveBeenCalledWith(linked, expect.any(Blob));
    expect(outcome).toEqual({ status: 'saved', name: 'linked.json' });
  });

  it('Save As ignores the link and re-links to the new file', async () => {
    await store.set('session', ID, { handle: handle('old.json'), name: 'old.json', savedAt: 'x' });
    const io = makeIO();
    await saveToLinkedFile(request({ saveAs: true }), store, io);
    expect(io.pickSave).toHaveBeenCalled();
    expect((await store.get('session', ID))?.name).toBe('picked.json');
  });

  it('does nothing when the picker is cancelled', async () => {
    const req = request();
    const io = makeIO({ pickSave: vi.fn().mockResolvedValue(null) });
    expect(await saveToLinkedFile(req, store, io)).toEqual({ status: 'cancelled' });
    expect(req.getBlob).not.toHaveBeenCalled();
    expect(await store.get('session', ID)).toBeUndefined();
  });

  it('reports a refused permission without writing', async () => {
    await store.set('session', ID, { handle: handle('linked.json'), name: 'linked.json', savedAt: 'x' });
    const io = makeIO({ ensurePermission: vi.fn().mockResolvedValue(false) });
    const outcome = await saveToLinkedFile(request(), store, io);
    expect(outcome).toEqual({ status: 'failed', name: 'linked.json', reason: 'permission' });
    expect(io.write).not.toHaveBeenCalled();
  });

  it('reports a failed write and keeps the previous link time', async () => {
    await store.set('session', ID, { handle: handle('gone.json'), name: 'gone.json', savedAt: 'before' });
    const error = new DOMException('missing', 'NotFoundError');
    const io = makeIO({ write: vi.fn().mockRejectedValue(error) });
    const outcome = await saveToLinkedFile(request(), store, io);
    expect(outcome).toEqual({ status: 'failed', name: 'gone.json', reason: 'write', error });
    expect((await store.get('session', ID))?.savedAt).toBe('before');
  });

  it('downloads where linked saving is unsupported', async () => {
    const io = makeIO({ supported: () => false });
    const outcome = await saveToLinkedFile(request(), store, io);
    expect(io.download).toHaveBeenCalledWith(expect.any(Blob), 'C1 screen.json');
    expect(io.pickSave).not.toHaveBeenCalled();
    expect(outcome).toEqual({ status: 'downloaded', name: 'C1 screen.json' });
  });

  it('asks for permission before fetching the export', async () => {
    await store.set('session', ID, { handle: handle('linked.json'), name: 'linked.json', savedAt: 'x' });
    const order: string[] = [];
    const io = makeIO({ ensurePermission: vi.fn(async () => { order.push('permission'); return true; }) });
    await saveToLinkedFile(request({ getBlob: async () => { order.push('export'); return new Blob(['{}']); } }), store, io);
    expect(order).toEqual(['permission', 'export']);
  });
});
