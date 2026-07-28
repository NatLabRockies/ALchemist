import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithQuery } from '../../test/queryWrapper';
import * as useQueue from '../../hooks/api/useQueue';
import { ConfigChangeTimeline } from './ConfigChangeTimeline';

describe('ConfigChangeTimeline', () => {
  it('renders each config change with component and iteration', () => {
    vi.spyOn(useQueue, 'useConfigChanges').mockReturnValue({
      data: { changes: [
        { timestamp: '2026-07-28T00:00:00', component: 'acquisition',
          old: {}, new: { strategy: 'qEI' }, iteration: 12 },
      ] },
      isLoading: false, isSuccess: true,
    } as any);
    renderWithQuery(<ConfigChangeTimeline sessionId="s1" />);
    expect(screen.getByText(/acquisition/i)).toBeInTheDocument();
    expect(screen.getByText(/qEI/)).toBeInTheDocument();
    expect(screen.getByText(/12/)).toBeInTheDocument();
  });

  it('shows an empty state with no changes', () => {
    vi.spyOn(useQueue, 'useConfigChanges').mockReturnValue({
      data: { changes: [] }, isLoading: false, isSuccess: true,
    } as any);
    renderWithQuery(<ConfigChangeTimeline sessionId="s1" />);
    expect(screen.getByText(/no config changes/i)).toBeInTheDocument();
  });
});
