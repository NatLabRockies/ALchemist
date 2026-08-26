import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ControlStrip } from './ControlStrip';
import type { ControlRecord } from '../../hooks/api/useControl';

const mockControl = vi.fn();
vi.mock('../../hooks/api/useControl', async (importOriginal) => ({
  ...(await importOriginal<any>()),
  useControl: () => mockControl(),
  useRequestControl: () => ({ mutate: vi.fn(), isPending: false }),
}));

const NOW = new Date('2026-08-26T18:00:00Z');
const agoSeconds = (s: number) => new Date(NOW.getTime() - s * 1000).toISOString();

function record(over: Partial<ControlRecord> = {}): ControlRecord {
  return {
    requested: 'run', requested_at: null, requested_by: null,
    reported: 'running', reported_at: agoSeconds(2), reported_by: 'ctl',
    detail: null, ...over,
  };
}

beforeEach(() => { vi.useFakeTimers(); vi.setSystemTime(NOW); });
afterEach(() => { vi.useRealTimers(); });

describe('ControlStrip', () => {
  it('renders the reported state, not the requested one', () => {
    // The core honesty property, checked in BOTH branches -- the settled one
    // and the request-pending one. Checking only the pending branch left the
    // settled branch free to render `requested` with every test still green.

    // Settled: requested 'run', reported 'running'. Rendering `requested`
    // here would put the word "run" on screen instead of "running".
    mockControl.mockReturnValue({ data: record() });
    const { unmount } = render(<ControlStrip sessionId="s1" />);
    expect(screen.getByText('running')).toBeInTheDocument();
    unmount();

    // Pending: a standing pause request must NOT make the strip say "paused"
    // while the controller is still running.
    mockControl.mockReturnValue({ data: record({ requested: 'pause' }) });
    render(<ControlStrip sessionId="s1" />);
    expect(screen.getByText('running')).toBeInTheDocument();
    expect(screen.queryByText(/^paused$/i)).not.toBeInTheDocument();
  });

  it('warns when a request is not yet acknowledged', () => {
    mockControl.mockReturnValue({
      data: record({ requested: 'pause', requested_at: agoSeconds(8) }),
    });
    render(<ControlStrip sessionId="s1" />);
    expect(screen.getByText(/not yet acknowledged/i)).toBeInTheDocument();
  });

  it('reads unknown, not the last state, once reports go stale', () => {
    // ALchemist unreachable must never be mistaken for the reactor in hand.
    mockControl.mockReturnValue({ data: record({ reported_at: agoSeconds(47) }) });
    render(<ControlStrip sessionId="s1" />);
    expect(screen.getByText(/unknown/i)).toBeInTheDocument();
    expect(screen.queryByText(/running/i)).not.toBeInTheDocument();
  });

  it('shows Resume when the controller reports paused', () => {
    mockControl.mockReturnValue({
      data: record({ requested: 'pause', requested_at: agoSeconds(9),
                     reported: 'paused' }),
    });
    render(<ControlStrip sessionId="s1" />);
    expect(screen.getByRole('button', { name: /resume/i })).toBeInTheDocument();
  });

  it('shows an age that advances without any refetch', () => {
    mockControl.mockReturnValue({ data: record({ reported_at: agoSeconds(2) }) });
    render(<ControlStrip sessionId="s1" />);
    expect(screen.getByText(/2s ago/)).toBeInTheDocument();
  });
});
