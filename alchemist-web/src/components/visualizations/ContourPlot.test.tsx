import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { QueryClientProvider } from '@tanstack/react-query';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { createTestQueryClient, renderWithQuery } from '../../test/queryWrapper';
import { ContourPlot } from './ContourPlot';

const getContourData = vi.fn();

vi.mock('../../api/endpoints/visualizations', () => ({
  getContourData: (...args: unknown[]) => getContourData(...args),
}));

vi.mock('../../hooks/useTheme', () => ({
  useTheme: () => ({ theme: 'light' }),
}));

vi.mock('react-plotly.js', () => ({
  default: ({ config }: { config: unknown }) => <div data-testid="plot" data-config={JSON.stringify(config)} />,
}));

// A stable object: ContourPlot re-initialises its fixed values whenever `variables` changes identity.
const variablesData = {
  n_variables: 5,
  variables: [
    { name: 'a', type: 'real', bounds: [0, 1] },
    { name: 'b', type: 'real', bounds: [0, 1] },
    { name: 'temp', type: 'real', bounds: [0, 10] },
    { name: 'press', type: 'real', bounds: [100, 200] },
    { name: 'count', type: 'integer', bounds: [1, 10] },
  ],
};

vi.mock('../../hooks/api/useVariables', () => ({
  useVariables: () => ({ data: variablesData }),
}));

function contourResponse(x: string, y: string) {
  return {
    x_var: x,
    y_var: y,
    x_grid: [[0, 1], [0, 1]],
    y_grid: [[0, 0], [1, 1]],
    predictions: [[0, 1], [2, 3]],
    uncertainties: [[0, 0], [0, 0]],
    x_bounds: [0, 1],
    y_bounds: [0, 1],
    colorbar_bounds: [0, 3],
  };
}

const DEBOUNCE_MS = 150;

async function settle(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

/** Mount, let the initial contour requests finish, and return how many were made. */
async function mount() {
  renderWithQuery(<ContourPlot sessionId="s1" />);
  await settle(DEBOUNCE_MS * 2);
  return getContourData.mock.calls.length;
}

/** Type text the way a user does: an input event that carries an inputType. */
function typeText(input: HTMLElement, text: string) {
  fireEvent.input(input, { target: { value: text }, inputType: 'insertText' });
}

function lastFixed() {
  const calls = getContourData.mock.calls;
  return calls[calls.length - 1][1].fixed_values as Record<string, number>;
}

describe('ContourPlot fixed values', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    getContourData.mockReset();
    getContourData.mockImplementation((_id: string, req: { x_var: string; y_var: string }) =>
      Promise.resolve(contourResponse(req.x_var, req.y_var))
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it('requests the contour once after a keyboard change on the slider', async () => {
    const before = await mount();
    const slider = screen.getByLabelText('temp (slider)');

    // An arrow key on a range input fires one change event and no mouseup.
    fireEvent.change(slider, { target: { value: '5.1' } });
    await settle(DEBOUNCE_MS - 50);
    expect(getContourData).toHaveBeenCalledTimes(before);

    await settle(100);
    expect(getContourData).toHaveBeenCalledTimes(before + 1);
    expect(lastFixed().temp).toBeCloseTo(5.1);
  });

  it('makes one request for a burst of rapid changes', async () => {
    const before = await mount();
    const slider = screen.getByLabelText('temp (slider)');

    for (const v of ['5.1', '5.2', '5.3', '5.4', '5.5']) {
      fireEvent.change(slider, { target: { value: v } });
      await settle(40);
    }
    await settle(DEBOUNCE_MS);

    expect(getContourData).toHaveBeenCalledTimes(before + 1);
    expect(lastFixed().temp).toBeCloseTo(5.5);
  });

  it('commits at once when the slider is released', async () => {
    const before = await mount();
    const slider = screen.getByLabelText('temp (slider)');

    fireEvent.change(slider, { target: { value: '7' } });
    fireEvent.mouseUp(slider, { target: { value: '7' } });
    await settle(0);

    expect(getContourData).toHaveBeenCalledTimes(before + 1);
    expect(lastFixed().temp).toBe(7);
    await settle(DEBOUNCE_MS * 2);
    expect(getContourData).toHaveBeenCalledTimes(before + 1);
  });

  it('commits a typed value on Enter, clamped to the variable bounds', async () => {
    const before = await mount();
    const input = screen.getByLabelText('temp (value)') as HTMLInputElement;

    typeText(input, '42');
    await settle(DEBOUNCE_MS * 2);
    expect(getContourData).toHaveBeenCalledTimes(before); // typing alone commits nothing

    fireEvent.keyDown(input, { key: 'Enter' });
    await settle(0);

    expect(getContourData).toHaveBeenCalledTimes(before + 1);
    expect(lastFixed().temp).toBe(10);
    expect(input.value).toBe('10');
  });

  it('commits a typed value on blur', async () => {
    const before = await mount();
    const input = screen.getByLabelText('press (value)') as HTMLInputElement;

    typeText(input, '137.5');
    fireEvent.blur(input);
    await settle(0);

    expect(getContourData).toHaveBeenCalledTimes(before + 1);
    expect(lastFixed().press).toBe(137.5);
  });

  it('restores the last committed value on Escape', async () => {
    const before = await mount();
    const input = screen.getByLabelText('temp (value)') as HTMLInputElement;
    expect(input.value).toBe('5');

    typeText(input, '8');
    expect(input.value).toBe('8');
    fireEvent.keyDown(input, { key: 'Escape' });
    expect(input.value).toBe('5');

    fireEvent.blur(input);
    await settle(DEBOUNCE_MS * 2);
    expect(getContourData).toHaveBeenCalledTimes(before);
  });

  it('restores the last committed value when the text cannot be parsed', async () => {
    const before = await mount();
    const input = screen.getByLabelText('temp (value)') as HTMLInputElement;

    typeText(input, '');
    fireEvent.keyDown(input, { key: 'Enter' });
    await settle(DEBOUNCE_MS * 2);

    expect(input.value).toBe('5');
    expect(getContourData).toHaveBeenCalledTimes(before);
  });

  it('ignores an emptied Integer box and clamps a typed integer to its bounds (FE-03)', async () => {
    const before = await mount();
    expect(lastFixed().count).toBe(5);
    const input = screen.getByLabelText('count (value)') as HTMLInputElement;

    // Emptying the box must not send null (the API answers 422 and the figure disappears)
    fireEvent.change(input, { target: { value: '' } });
    await settle(DEBOUNCE_MS * 2);
    expect(getContourData).toHaveBeenCalledTimes(before);
    for (const call of getContourData.mock.calls) {
      expect(call[1].fixed_values.count).toBe(5);
    }

    fireEvent.change(input, { target: { value: '999' } });
    await settle(0);
    expect(getContourData).toHaveBeenCalledTimes(before + 1);
    expect(lastFixed().count).toBe(10);
    expect(input.value).toBe('10');

    fireEvent.change(input, { target: { value: '-3' } });
    await settle(0);
    expect(lastFixed().count).toBe(1);
  });

  it('shows the Plotly toolbar on hover only, so it does not cover the title (W38)', async () => {
    await mount();
    const plot = screen.getByTestId('plot');
    const config = JSON.parse(plot.getAttribute('data-config')!);
    expect(config.displayModeBar).toBe('hover');
  });

  it('steps the value with ArrowUp and ArrowDown like the slider and commits once', async () => {
    const before = await mount();
    const input = screen.getByLabelText('temp (value)') as HTMLInputElement;
    const step = (10 - 0) / 100;

    fireEvent.keyDown(input, { key: 'ArrowUp' });
    fireEvent.keyDown(input, { key: 'ArrowUp' });
    fireEvent.keyDown(input, { key: 'ArrowDown' });
    await settle(DEBOUNCE_MS - 50);
    expect(getContourData).toHaveBeenCalledTimes(before);
    await settle(100);

    expect(getContourData).toHaveBeenCalledTimes(before + 1);
    expect(lastFixed().temp).toBeCloseTo(5 + step);
    expect(input.step).toBe(String(step));
  });

  it('does not step past the bounds', async () => {
    await mount();
    const input = screen.getByLabelText('temp (value)') as HTMLInputElement;

    typeText(input, '10');
    fireEvent.keyDown(input, { key: 'Enter' });
    fireEvent.keyDown(input, { key: 'ArrowUp' });
    await settle(DEBOUNCE_MS * 2);

    expect(lastFixed().temp).toBe(10);
  });

  it('keeps the previous figure on screen while a new contour loads', async () => {
    await mount();
    expect(screen.getByTestId('plot')).toBeInTheDocument();

    let resolveNext: (v: unknown) => void = () => {};
    getContourData.mockImplementationOnce(
      () => new Promise((resolve) => { resolveNext = resolve; })
    );
    const input = screen.getByLabelText('temp (value)');
    typeText(input, '3');
    fireEvent.keyDown(input, { key: 'Enter' });
    await settle(0);

    expect(screen.getByTestId('plot')).toBeInTheDocument();
    expect(screen.queryByText('Generating contour plot...')).toBeNull();
    expect(screen.getByTestId('plot').closest('.transition-opacity')).toHaveClass('opacity-60');

    await act(async () => {
      resolveNext(contourResponse('a', 'b'));
    });
    await settle(0);
    expect(screen.getByTestId('plot')).toBeInTheDocument();
    expect(screen.getByTestId('plot').closest('.transition-opacity')).not.toHaveClass('opacity-60');
  });

  it('drops a pending commit when the axes change', async () => {
    const before = await mount();
    fireEvent.change(screen.getByLabelText('temp (slider)'), { target: { value: '5.1' } });

    // X-Axis is the first menu. press leaves the fixed set and temp is re-initialised at mid-range.
    fireEvent.change(screen.getAllByRole('combobox')[0], { target: { value: 'press' } });
    await settle(10);
    // The axis change makes its own requests; the cancelled debounce must add none after them.
    const afterAxisChange = getContourData.mock.calls.length;
    expect(afterAxisChange).toBeGreaterThan(before);
    await settle(DEBOUNCE_MS * 3);
    expect(getContourData).toHaveBeenCalledTimes(afterAxisChange);

    const requests = getContourData.mock.calls.map((c) => c[1]);
    expect(requests.at(-1).x_var).toBe('press');
    expect(requests.at(-1).fixed_values.temp).toBe(5);
    expect(requests.some((r) => r.fixed_values.temp === 5.1)).toBe(false);
    expect((screen.getByLabelText('temp (value)') as HTMLInputElement).value).toBe('5');
  });

  it('drops a pending commit when the plot unmounts', async () => {
    const before = await mount();
    const setTimeoutSpy = vi.spyOn(globalThis, 'setTimeout');
    const clearTimeoutSpy = vi.spyOn(globalThis, 'clearTimeout');
    fireEvent.change(screen.getByLabelText('temp (slider)'), { target: { value: '5.1' } });

    // The debounce is the 150 ms timer the change just set; unmounting must clear it.
    const debounceIndex = setTimeoutSpy.mock.calls.findIndex((c) => c[1] === DEBOUNCE_MS);
    expect(debounceIndex).toBeGreaterThanOrEqual(0);
    const timerId = setTimeoutSpy.mock.results[debounceIndex].value;
    clearTimeoutSpy.mockClear();
    cleanup(); // unmounts the plot

    expect(clearTimeoutSpy).toHaveBeenCalledWith(timerId);
    await settle(DEBOUNCE_MS * 3);
    expect(getContourData).toHaveBeenCalledTimes(before);
  });

  it('commits typed text once on blur, then the slider on release', async () => {
    const before = await mount();
    const input = screen.getByLabelText('press (value)') as HTMLInputElement;
    typeText(input, '130');

    // Grabbing the slider moves focus out of the box first.
    fireEvent.blur(input);
    const slider = screen.getByLabelText('press (slider)');
    fireEvent.change(slider, { target: { value: '120' } });
    fireEvent.mouseUp(slider, { target: { value: '120' } });
    await settle(DEBOUNCE_MS * 3);

    const fixed = getContourData.mock.calls.slice(before).map((c) => c[1].fixed_values.press);
    expect(fixed).toEqual([130, 120]);
    expect(input.value).toBe('120');
  });

  it('does not show the previous session\'s contour while another session loads', async () => {
    const client = createTestQueryClient();
    const ui = (id: string) => (
      <QueryClientProvider client={client}>
        <ContourPlot sessionId={id} />
      </QueryClientProvider>
    );
    const { rerender } = render(ui('s1'));
    await settle(DEBOUNCE_MS * 2);
    expect(screen.getByTestId('plot')).toBeInTheDocument();

    getContourData.mockImplementation(() => new Promise(() => {}));
    rerender(ui('s2'));
    await settle(0);

    expect(screen.queryByTestId('plot')).toBeNull();
    expect(screen.getByText('Generating contour plot...')).toBeInTheDocument();
  });
});
