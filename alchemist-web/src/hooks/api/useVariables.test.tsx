import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AxiosError, AxiosHeaders } from 'axios';

vi.mock('sonner', () => {
  const base = Object.assign(vi.fn(), {
    success: vi.fn(), info: vi.fn(), error: vi.fn(), warning: vi.fn(), dismiss: vi.fn(),
  });
  return { toast: base };
});

import { toast as sonnerToast } from 'sonner';
import * as variablesAPI from '../../api/endpoints/variables';
import { useCreateVariable } from './useVariables';

const httpError = (status: number, data: unknown) =>
  new AxiosError('Request failed', String(status), undefined, undefined, {
    status, statusText: '', headers: {}, config: { headers: new AxiosHeaders() }, data,
  });

function setup() {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  return renderHook(() => useCreateVariable('s1'), {
    wrapper: ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    ),
  });
}

describe('useCreateVariable errors', () => {
  beforeEach(() => vi.clearAllMocks());

  it('shows the server detail of a 400 as an error toast', async () => {
    const detail = "Variable 'T' already exists. Please use a different name or delete the existing variable first.";
    vi.spyOn(variablesAPI, 'createVariable').mockRejectedValueOnce(httpError(400, { detail }));
    const { result } = setup();
    await act(async () => {
      await result.current.mutateAsync({ name: 'T', type: 'continuous', bounds: [0, 1] }).catch(() => {});
    });
    expect(sonnerToast.error).toHaveBeenCalledWith(detail, expect.anything());
  });

  it('shows a 422 validation error as text, not [object Object]', async () => {
    vi.spyOn(variablesAPI, 'createVariable').mockRejectedValueOnce(
      httpError(422, { detail: [{ loc: ['body', 'min'], msg: 'Input should be a valid number', type: 'float_parsing' }] }),
    );
    const { result } = setup();
    await act(async () => {
      await result.current.mutateAsync({ name: 'T', type: 'continuous' }).catch(() => {});
    });
    const [message] = (sonnerToast.error as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(typeof message).toBe('string');
    expect(message).toContain('min');
    expect(message).toContain('Input should be a valid number');
  });

  it('says the server could not be reached on a network error', async () => {
    vi.spyOn(variablesAPI, 'createVariable').mockRejectedValueOnce(new AxiosError('Network Error', 'ERR_NETWORK'));
    const { result } = setup();
    await act(async () => {
      await result.current.mutateAsync({ name: 'T', type: 'continuous' }).catch(() => {});
    });
    const [message] = (sonnerToast.error as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(message).toMatch(/could not reach the server/i);
  });
});
