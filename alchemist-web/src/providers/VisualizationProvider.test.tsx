import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import type { ReactNode } from 'react';
import { VisualizationProvider, useVisualizationForSession } from './VisualizationProvider';

const wrapper = ({ children }: { children: ReactNode }) => (
  <VisualizationProvider>{children}</VisualizationProvider>
);

describe('useVisualizationForSession', () => {
  it('closes the panel when the app moves to another session, so Clear → New Session never shows the old plots (W24)', () => {
    const { result, rerender } = renderHook(
      ({ sessionId }: { sessionId: string | null }) => useVisualizationForSession(sessionId),
      { wrapper, initialProps: { sessionId: 'A' } }
    );

    act(() => result.current.openVisualization('A', 'botorch'));
    expect(result.current.isVisualizationOpen).toBe(true);
    expect(result.current.sessionId).toBe('A');

    // Clear
    rerender({ sessionId: null });
    expect(result.current.isVisualizationOpen).toBe(false);

    // New Session: still closed, and nothing of the old session remains
    rerender({ sessionId: 'B' });
    expect(result.current.isVisualizationOpen).toBe(false);
    expect(result.current.sessionId).toBeNull();
  });

  it('stays open while the session is the one it was opened for', () => {
    const { result, rerender } = renderHook(
      ({ sessionId }: { sessionId: string | null }) => useVisualizationForSession(sessionId),
      { wrapper, initialProps: { sessionId: 'A' } }
    );
    act(() => result.current.openVisualization('A', 'sklearn'));
    rerender({ sessionId: 'A' });
    expect(result.current.isVisualizationOpen).toBe(true);
    expect(result.current.modelBackend).toBe('sklearn');
  });
});
