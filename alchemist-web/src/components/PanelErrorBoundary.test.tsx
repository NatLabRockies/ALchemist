import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PanelErrorBoundary } from './PanelErrorBoundary';

let shouldThrow = true;
function Bomb() {
  if (shouldThrow) throw new Error("Cannot read properties of undefined (reading 'toFixed')");
  return <p>panel content</p>;
}

describe('PanelErrorBoundary', () => {
  it('shows a contained message instead of unmounting, and can retry', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    render(
      <div>
        <p>header</p>
        <PanelErrorBoundary name="Model"><Bomb /></PanelErrorBoundary>
      </div>,
    );
    expect(screen.getByText('header')).toBeTruthy();
    expect(screen.getByRole('alert').textContent).toContain('The Model panel hit an error.');
    expect(screen.getByRole('alert').textContent).toContain('toFixed');

    shouldThrow = false;
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(screen.getByText('panel content')).toBeTruthy();
  });
});
