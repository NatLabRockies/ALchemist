import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithQuery } from '../../test/queryWrapper';

vi.mock('./tabs/ConfigTab', () => ({ ConfigTab: () => <div>CONFIG TAB</div> }));
vi.mock('./tabs/LiveTab', () => ({ LiveTab: ({ goal }: { goal: string }) => <div>LIVE TAB goal={goal}</div> }));
let lastGoal: 'maximize' | 'minimize' | null = null;
vi.mock('../../hooks/api/useSessions', () => ({
  useSessionState: () => ({ data: { session_id: 's1', n_variables: 1, n_experiments: 1, model_trained: false, last_suggestion: null, last_goal: lastGoal } }),
}));
vi.mock('./tabs/HistoryTab', () => ({ HistoryTab: () => <div>HISTORY TAB</div> }));

import { LiveMonitor } from './LiveMonitor';

describe('LiveMonitor', () => {
  beforeEach(() => { window.history.replaceState({}, '', '/'); });

  it('defaults to the Live tab', () => {
    renderWithQuery(<LiveMonitor sessionId="s1" />);
    expect(screen.getByText(/LIVE TAB/)).toBeInTheDocument();
  });

  it("traces best-so-far with the session's last acquisition goal, maximize when none (W40)", () => {
    lastGoal = 'minimize';
    try {
      renderWithQuery(<LiveMonitor sessionId="s1" />);
      expect(screen.getByText('LIVE TAB goal=minimize')).toBeInTheDocument();
    } finally {
      lastGoal = null;
    }
  });

  it('defaults the goal to maximize before any suggestion', () => {
    renderWithQuery(<LiveMonitor sessionId="s1" />);
    expect(screen.getByText('LIVE TAB goal=maximize')).toBeInTheDocument();
  });

  it('honors ?tab=config', () => {
    window.history.replaceState({}, '', '/?tab=config');
    renderWithQuery(<LiveMonitor sessionId="s1" />);
    expect(screen.getByText('CONFIG TAB')).toBeInTheDocument();
  });

  it('switches tabs on click', async () => {
    renderWithQuery(<LiveMonitor sessionId="s1" />);
    await userEvent.click(screen.getByRole('tab', { name: /history/i }));
    expect(screen.getByText('HISTORY TAB')).toBeInTheDocument();
  });
});
