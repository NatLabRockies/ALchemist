import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('sonner', () => {
  const base = Object.assign(vi.fn(), {
    success: vi.fn(),
    info: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    dismiss: vi.fn(),
  });
  return { toast: base };
});

import { toast as sonnerToast } from 'sonner';
import { toast, PROBLEM_TOAST_DURATION_MS } from './toast';

describe('toast wrapper', () => {
  beforeEach(() => vi.clearAllMocks());

  it('holds errors and warnings longer than the default', () => {
    toast.error('boom');
    toast.warning('careful', { description: 'detail' });
    expect(sonnerToast.error).toHaveBeenCalledWith('boom', { duration: PROBLEM_TOAST_DURATION_MS });
    expect(sonnerToast.warning).toHaveBeenCalledWith('careful', {
      duration: PROBLEM_TOAST_DURATION_MS,
      description: 'detail',
    });
  });

  it("lets a caller's own duration win", () => {
    toast.error('boom', { duration: 2000 });
    expect(sonnerToast.error).toHaveBeenCalledWith('boom', { duration: 2000 });
  });

  it('leaves success and info on the default duration', () => {
    toast.success('done');
    toast.info('fyi', { duration: 3000 });
    expect(sonnerToast.success).toHaveBeenCalledWith('done');
    expect(sonnerToast.info).toHaveBeenCalledWith('fyi', { duration: 3000 });
  });

  it('is still callable as a plain toast', () => {
    toast('hello');
    expect(sonnerToast).toHaveBeenCalledWith('hello', undefined);
  });
});
