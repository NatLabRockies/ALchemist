import { describe, it, expect } from 'vitest';
import { queryClient } from './QueryProvider';

describe('queryClient defaults', () => {
  it('does not retry mutations: a failed train, suggest or upload must not be sent twice (FE-02)', () => {
    expect(queryClient.getDefaultOptions().mutations?.retry).toBe(0);
  });

  it('retries queries once', () => {
    expect(queryClient.getDefaultOptions().queries?.retry).toBe(1);
  });
});
