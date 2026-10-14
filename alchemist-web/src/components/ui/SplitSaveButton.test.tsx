import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SplitSaveButton } from './SplitSaveButton';

function setup(showMenu = true) {
  const onSave = vi.fn();
  const onSaveAs = vi.fn();
  render(<SplitSaveButton label="Save" onSave={onSave} onSaveAs={onSaveAs} showMenu={showMenu} variant="primary" />);
  return { onSave, onSaveAs };
}

describe('SplitSaveButton', () => {
  it('saves from the main button', () => {
    const { onSave } = setup();
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(onSave).toHaveBeenCalledTimes(1);
  });

  it('opens a menu with Save and Save As, focused on the first item', () => {
    setup();
    fireEvent.click(screen.getByRole('button', { name: 'Save options' }));
    const items = screen.getAllByRole('menuitem');
    expect(items.map(i => i.textContent)).toEqual(['Save', 'Save As…']);
    expect(document.activeElement).toBe(items[0]);
  });

  it('moves with the arrow keys and chooses with a click', () => {
    const { onSaveAs } = setup();
    fireEvent.click(screen.getByRole('button', { name: 'Save options' }));
    const menu = screen.getByRole('menu');
    fireEvent.keyDown(menu, { key: 'ArrowDown' });
    expect(document.activeElement?.textContent).toBe('Save As…');
    fireEvent.click(document.activeElement as HTMLElement);
    expect(onSaveAs).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('menu')).toBeNull();
  });

  it('closes on Escape and returns focus to the toggle', () => {
    setup();
    const toggle = screen.getByRole('button', { name: 'Save options' });
    fireEvent.click(toggle);
    fireEvent.keyDown(screen.getByRole('menu'), { key: 'Escape' });
    expect(screen.queryByRole('menu')).toBeNull();
    expect(document.activeElement).toBe(toggle);
  });

  it('is a plain button where only downloads work', () => {
    setup(false);
    expect(screen.queryByRole('button', { name: 'Save options' })).toBeNull();
  });
});
