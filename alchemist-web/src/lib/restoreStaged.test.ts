import { describe, it, expect } from 'vitest';
import { tagRestoredStaged, restorePendingSuggestions } from './restoreStaged';

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

// ---------------------------------------------------------------------------
// restorePendingSuggestions: the session-load wiring, with fetch injected
// ---------------------------------------------------------------------------

type Routes = Record<string, { ok?: boolean; body?: unknown; throws?: boolean }>;

function fakeFetch(routes: Routes): typeof fetch {
  return (async (url: string) => {
    const key = Object.keys(routes).find((k) => String(url).includes(k));
    if (!key) return { ok: false, json: async () => ({}) };
    const r = routes[key];
    if (r.throws) throw new Error('network down');
    return { ok: r.ok ?? true, json: async () => r.body };
  }) as unknown as typeof fetch;
}

const audit = {
  entries: [
    { parameters: { strategy: 'qEI', iteration: 4, suggestions: [{ temp: 250, cat: 'Pt' }, { temp: 300, cat: 'Pd' }] } },
  ],
};
const noStaged = { '/experiments/staged': { body: { experiments: [] } } };

describe('restorePendingSuggestions', () => {
  it('returns staged rows untouched when the staged queue has them', async () => {
    const f = fakeFetch({ '/experiments/staged': { body: { experiments: [{ temp: 1 }], ids: ['q1'] } } });
    const out = await restorePendingSuggestions('s1', f);
    expect(out).toHaveLength(1);
    expect(out[0]._queueItemId).toBe('q1');
  });

  it('audit fallback drops a suggestion already in the experiment table', async () => {
    const f = fakeFetch({
      ...noStaged,
      '/audit': { body: audit },
      '/variables': { body: { variables: [] } },
      '/experiments': { body: { experiments: [{ temp: 250, cat: 'Pt', Output: 0.7 }] } },
    });
    const out = await restorePendingSuggestions('s1', f);
    expect(out).toEqual([{ temp: 300, cat: 'Pd', _reason: 'qEI', Iteration: 4 }]);
  });

  it('audit fallback returns nothing when every suggestion was recorded', async () => {
    const f = fakeFetch({
      ...noStaged,
      '/audit': { body: audit },
      '/variables': { body: { variables: [] } },
      '/experiments': {
        body: { experiments: [{ temp: 250, cat: 'Pt', Output: 1 }, { temp: 300, cat: 'Pd', Output: 2 }] },
      },
    });
    expect(await restorePendingSuggestions('s1', f)).toEqual([]);
  });

  it('audit fallback keeps all suggestions when the table is empty', async () => {
    const f = fakeFetch({
      ...noStaged,
      '/audit': { body: audit },
      '/variables': { body: { variables: [] } },
      '/experiments': { body: { experiments: [] } },
    });
    expect(await restorePendingSuggestions('s1', f)).toHaveLength(2);
  });

  it('returns nothing, not possible duplicates, when the experiment fetch fails', async () => {
    const f = fakeFetch({
      ...noStaged,
      '/audit': { body: audit },
      '/variables': { body: { variables: [] } },
      '/experiments': { ok: false },
    });
    expect(await restorePendingSuggestions('s1', f)).toEqual([]);
  });

  it('still filters (generic rounding) when only the variables fetch fails', async () => {
    const f = fakeFetch({
      ...noStaged,
      '/audit': { body: audit },
      '/variables': { throws: true },
      '/experiments': { body: { experiments: [{ temp: 250, cat: 'Pt', Output: 0.7 }] } },
    });
    expect(await restorePendingSuggestions('s1', f)).toHaveLength(1);
  });

  it('drops a suggestion recorded with an edited actual value, via provenance', async () => {
    const f = fakeFetch({
      ...noStaged,
      '/audit': { body: audit },
      '/variables': { body: { variables: [] } },
      // the table holds what was actually run (temp edited to 255)
      '/experiments/provenance': {
        body: {
          records: [{ id: 'q1', suggested: { temp: 250, cat: 'Pt' }, actual: { temp: 255, cat: 'Pt' } }],
          n_records: 1,
        },
      },
      '/experiments': { body: { experiments: [{ temp: 255, cat: 'Pt', Output: 0.7 }] } },
    });
    const out = await restorePendingSuggestions('s1', f);
    expect(out).toEqual([{ temp: 300, cat: 'Pd', _reason: 'qEI', Iteration: 4 }]);
  });

  it('falls back to the table match when the provenance fetch fails', async () => {
    const f = fakeFetch({
      ...noStaged,
      '/audit': { body: audit },
      '/variables': { body: { variables: [] } },
      '/experiments/provenance': { throws: true },
      '/experiments': { body: { experiments: [{ temp: 255, cat: 'Pt', Output: 0.7 }] } },
    });
    expect(await restorePendingSuggestions('s1', f)).toHaveLength(2);
  });

  it('ignores provenance records with no suggested conditions', async () => {
    const f = fakeFetch({
      ...noStaged,
      '/audit': { body: audit },
      '/variables': { body: { variables: [] } },
      '/experiments/provenance': { body: { records: [{ id: 'm', suggested: null }], n_records: 1 } },
      '/experiments': { body: { experiments: [{ temp: 1, cat: 'Pt', Output: 0.7 }] } },
    });
    expect(await restorePendingSuggestions('s1', f)).toHaveLength(2);
  });
});
