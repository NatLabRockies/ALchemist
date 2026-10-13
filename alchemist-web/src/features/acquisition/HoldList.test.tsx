// src/features/acquisition/HoldList.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { useState } from 'react';
import type { VariableDetail } from '../../api/types';
import { HoldList, holdCurrentSetup, holdPayload, type HoldState } from './HoldList';

const VARS: VariableDetail[] = [
  { name: 'x1', type: 'real', bounds: [0, 10], role: 'controllable', changes: 'easy' },
  { name: 'c', type: 'categorical', categories: ['a', 'b'], role: 'controllable', changes: 'hard' },
  { name: 'h1', type: 'real', bounds: [0, 1], role: 'controllable', changes: 'very_hard' },
  { name: 'z', type: 'real', role: 'observed' },
];
const LAST = { x1: 3.2, c: 'b', h1: 0.5, z: 24.0, Output: 1.0 };

describe('hold helpers', () => {
  it('everything starts unlocked; Hold current setup locks hard variables at the last run', () => {
    const state = holdCurrentSetup({}, VARS, LAST);
    expect(state.c).toEqual({ locked: true, value: 'b' });
    expect(state.h1).toEqual({ locked: true, value: '0.5' });
    expect(state.x1?.locked ?? false).toBe(false);
  });

  it('builds hold and observed payloads with typed values', () => {
    const p = holdPayload({ c: { locked: true, value: 'b' }, h1: { locked: true, value: '0.5' },
      x1: { locked: false, value: '3' } }, { z: '24' }, VARS);
    expect(p).toEqual({ hold: { c: 'b', h1: 0.5 }, observed: { z: 24 }, missingObserved: [] });
  });

  it('reports missing observed values', () => {
    expect(holdPayload({}, {}, VARS).missingObserved).toEqual(['z']);
  });
});

describe('HoldList', () => {
  function Harness() {
    const [state, setState] = useState<HoldState>({});
    const [obs, setObs] = useState<Record<string, string>>({});
    return <HoldList variables={VARS} state={state} observedValues={obs} lastRow={LAST}
      onChange={setState} onObservedChange={setObs} />;
  }

  it('locks a categorical with a dropdown and fills hard variables with one click', () => {
    render(<Harness />);
    expect(screen.getByLabelText('hold c')).not.toBeChecked();
    fireEvent.click(screen.getByText('Hold current setup'));
    expect(screen.getByLabelText('hold c')).toBeChecked();
    expect(screen.getByLabelText('c held value')).toHaveValue('b');
    expect(screen.getByLabelText('hold x1')).not.toBeChecked();
    expect(screen.getByLabelText('z measured value')).toBeInTheDocument();
  });
});
