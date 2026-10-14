import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

vi.mock('../../hooks/api/useDeleteVariable', () => ({
  useDeleteVariable: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));

import { VariableList } from './VariableList';

describe('VariableList badges', () => {
  it('wraps role/changes badges inside the Type cell and leaves Range/Values intact', () => {
    render(
      <VariableList
        sessionId="s1"
        onEdit={() => {}}
        variables={[
          { name: 'c', type: 'categorical', categories: ['a', 'b'], role: 'controllable', changes: 'very_hard' },
          { name: 'k', type: 'categorical', categories: ['p', 'q'], role: 'observed' },
        ]}
      />,
    );
    const badge = screen.getByText('very hard');
    const typeCell = badge.parentElement!;
    expect(typeCell.className).toContain('flex-wrap');
    expect(typeCell.className).toContain('min-w-0');
    expect(badge.className).toContain('whitespace-nowrap');
    expect(screen.getByText('observed').parentElement!.className).toContain('flex-wrap');
    expect(screen.getByText('a, b')).toBeTruthy();
    expect(screen.getByText('p, q')).toBeTruthy();
  });
});
