import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('sonner', () => {
  const base = Object.assign(vi.fn(), {
    success: vi.fn(), info: vi.fn(), error: vi.fn(), warning: vi.fn(), dismiss: vi.fn(),
  });
  return { toast: base };
});

import { toast as sonnerToast } from 'sonner';
import {
  SAVE_FAILED_MESSAGE,
  SAVE_WARNING_THROTTLE_MS,
  noteSavedHeader,
  observeFetchForSaves,
  resetSaveWarningsForTests,
} from './serverSave';

describe('X-ALchemist-Saved (FE-10)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetSaveWarningsForTests();
  });

  it('warns once when the server could not save the session file', () => {
    expect(noteSavedHeader('false', 1000)).toBe(true);
    expect(sonnerToast.warning).toHaveBeenCalledTimes(1);
    expect(sonnerToast.warning).toHaveBeenCalledWith(SAVE_FAILED_MESSAGE, expect.anything());
  });

  it('throttles repeated failures to one toast', () => {
    noteSavedHeader('false', 1000);
    expect(noteSavedHeader('false', 1000 + SAVE_WARNING_THROTTLE_MS - 1)).toBe(false);
    expect(sonnerToast.warning).toHaveBeenCalledTimes(1);
    expect(noteSavedHeader('false', 1000 + SAVE_WARNING_THROTTLE_MS)).toBe(true);
    expect(sonnerToast.warning).toHaveBeenCalledTimes(2);
  });

  it('is silent when the header says true or is absent', () => {
    expect(noteSavedHeader('true', 1000)).toBe(false);
    expect(noteSavedHeader(null, 1000)).toBe(false);
    expect(noteSavedHeader(undefined, 1000)).toBe(false);
    expect(sonnerToast.warning).not.toHaveBeenCalled();
  });

  it('reads the header off plain fetch responses', async () => {
    const fetchFn = vi.fn(async () =>
      new Response('{}', { status: 200, headers: { 'X-ALchemist-Saved': 'false' } }),
    );
    const observed = observeFetchForSaves(fetchFn as unknown as typeof fetch);
    await observed('/api/v1/sessions/x/experiments', { method: 'POST' });
    await observed('/api/v1/sessions/x/experiments', { method: 'POST' });
    expect(sonnerToast.warning).toHaveBeenCalledTimes(1);
  });
});
