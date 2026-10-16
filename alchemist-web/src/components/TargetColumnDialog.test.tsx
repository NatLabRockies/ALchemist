import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TargetColumnDialog } from './TargetColumnDialog';

const COLUMNS = ['Temperature', 'Pressure', 'Yield'];

/** The dialog is always mounted by ExperimentsPanel; it opens with the columns of the file just previewed. */
function openDialog(recommended?: string) {
  const onConfirm = vi.fn();
  const props = { onClose: vi.fn(), onConfirm };
  const view = render(<TargetColumnDialog open={false} availableColumns={[]} {...props} />);
  view.rerender(
    <TargetColumnDialog open={true} availableColumns={COLUMNS} recommendedColumn={recommended} {...props} />,
  );
  return onConfirm;
}

describe('TargetColumnDialog (FE-04 / W6)', () => {
  it('pre-selects the recommended column when it opens, so Confirm works at once', () => {
    const onConfirm = openDialog('Yield');
    expect((screen.getByLabelText('Target Column') as HTMLSelectElement).value).toBe('Yield');
    const confirm = screen.getByRole('button', { name: 'Confirm' });
    expect(confirm).toBeEnabled();
    fireEvent.click(confirm);
    expect(onConfirm).toHaveBeenCalledWith('Yield');
  });

  it('pre-selects the first column without a recommendation', () => {
    const onConfirm = openDialog(undefined);
    expect((screen.getByLabelText('Target Column') as HTMLSelectElement).value).toBe('Temperature');
    fireEvent.click(screen.getByRole('button', { name: 'Confirm' }));
    expect(onConfirm).toHaveBeenCalledWith('Temperature');
  });

  it('lets the user pick another column', () => {
    const onConfirm = openDialog('Yield');
    fireEvent.change(screen.getByLabelText('Target Column'), { target: { value: 'Pressure' } });
    fireEvent.click(screen.getByRole('button', { name: 'Confirm' }));
    expect(onConfirm).toHaveBeenCalledWith('Pressure');
  });
});
