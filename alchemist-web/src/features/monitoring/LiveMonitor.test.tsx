import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithQuery } from '../../test/queryWrapper';

vi.mock('./tabs/ConfigTab', () => ({ ConfigTab: () => <div>CONFIG TAB</div> }));
vi.mock('./tabs/LiveTab', () => ({ LiveTab: () => <div>LIVE TAB</div> }));
vi.mock('./tabs/HistoryTab', () => ({ HistoryTab: () => <div>HISTORY TAB</div> }));

import { LiveMonitor } from './LiveMonitor';

describe('LiveMonitor', () => {
  beforeEach(() => { window.history.replaceState({}, '', '/'); });

  it('defaults to the Live tab', () => {
    renderWithQuery(<LiveMonitor sessionId="s1" />);
    expect(screen.getByText('LIVE TAB')).toBeInTheDocument();
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
