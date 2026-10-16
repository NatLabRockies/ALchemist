import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const createMock = vi.fn().mockResolvedValue({});
vi.mock('../../hooks/api/useVariables', () => ({
  useCreateVariable: () => ({ mutateAsync: createMock, isPending: false }),
  useUpdateVariable: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useVariables: () => ({ data: { variables: [], n_variables: 0 } }),
}));
vi.mock('../../hooks/api/useDeleteVariable', () => ({
  useDeleteVariable: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));

import { VariableForm } from './VariableForm';

describe('VariableForm role and changes', () => {
  it('creates a hard categorical', async () => {
    render(<VariableForm sessionId="s1" onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'c' } });
    fireEvent.change(screen.getByLabelText('Type'), { target: { value: 'categorical' } });
    fireEvent.change(screen.getByLabelText('New category'), { target: { value: 'a' } });
    fireEvent.click(screen.getByText('Add'));
    fireEvent.change(screen.getByLabelText('Changes'), { target: { value: 'hard' } });
    fireEvent.click(screen.getByRole('button', { name: /add variable/i }));
    await waitFor(() => expect(createMock).toHaveBeenCalled());
    expect(createMock.mock.calls[0][0]).toMatchObject({ name: 'c', type: 'categorical', changes: 'hard' });
  });

  it('creates an observed real without bounds and hides Changes', async () => {
    createMock.mockClear();
    render(<VariableForm sessionId="s1" onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'z' } });
    fireEvent.change(screen.getByLabelText('Role'), { target: { value: 'observed' } });
    expect(screen.queryByLabelText('Changes')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /add variable/i }));
    await waitFor(() => expect(createMock).toHaveBeenCalled());
    expect(createMock.mock.calls[0][0]).toMatchObject({ name: 'z', role: 'observed' });
    expect(createMock.mock.calls[0][0].bounds).toBeUndefined();
  });
});

describe('VariableForm bounds', () => {
  it('refuses Min >= Max inline, before any request', async () => {
    createMock.mockClear();
    render(<VariableForm sessionId="s1" onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'T' } });
    fireEvent.change(screen.getByLabelText(/^Min/), { target: { value: '300' } });
    fireEvent.change(screen.getByLabelText(/^Max/), { target: { value: '200' } });
    fireEvent.click(screen.getByRole('button', { name: /add variable/i }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Min must be less than Max');
    expect(createMock).not.toHaveBeenCalled();
    // the message clears once the bounds are edited
    fireEvent.change(screen.getByLabelText(/^Max/), { target: { value: '400' } });
    expect(screen.queryByRole('alert')).toBeNull();
  });
});
