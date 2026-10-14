import { describe, it, expect } from 'vitest';
import { tagRestoredStaged } from './restoreStaged';

describe('tagRestoredStaged', () => {
  it('tags each row with its own reason and keeps plot keys', () => {
    const rows = tagRestoredStaged({
      experiments: [
        { x1: 1, c: 'a', WholePlot: 0, RunOrder: 1 },
        { x1: 2, c: 'a', WholePlot: 1, RunOrder: 2 },
      ],
      reason: 'Reference (split-plot)',
      reasons: ['Reference (split-plot)', 'Initial DoE (split-plot)'],
      ids: ['q1', 'q2'],
    });
    expect(rows.map((r) => r._reason)).toEqual(['Reference (split-plot)', 'Initial DoE (split-plot)']);
    expect(rows.map((r) => r.WholePlot)).toEqual([0, 1]);
    expect(rows.map((r) => r.RunOrder)).toEqual([1, 2]);
    expect(rows.map((r) => r._queueItemId)).toEqual(['q1', 'q2']);
  });

  it('falls back to the scalar reason, then to "Staged"', () => {
    const rows = tagRestoredStaged({
      experiments: [{ x1: 1 }, { x1: 2 }],
      reason: 'Initial DoE',
      reasons: [null, 'Acquisition'],
    });
    expect(rows.map((r) => r._reason)).toEqual(['Initial DoE', 'Acquisition']);
    expect(tagRestoredStaged({ experiments: [{ x1: 1 }] })[0]._reason).toBe('Staged');
  });
});
