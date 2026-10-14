/**
 * Split-plot mode of the Initial Design panel: one design per change level,
 * hard (and very hard) variables change between whole plots, easy variables
 * run inside each. Staged rows keep WholePlot/Subplot/RunOrder so Add Point
 * can show when the setup has to change.
 */
import { useMemo, useState } from 'react';
import { toast } from '../../lib/toast';
import type { DoEMethod, SplitPlotDesignResponse, SplitPlotLevel, VariableDetail } from '../../api/types';
import { useGenerateSplitPlotDesign } from '../../hooks/api/useExperiments';

const LEVELS: SplitPlotLevel[] = ['very_hard', 'hard', 'easy'];
const LEVEL_LABEL: Record<SplitPlotLevel, string> = { very_hard: 'Very hard', hard: 'Hard', easy: 'Easy' };
const METHODS: DoEMethod[] = ['lhs', 'sobol', 'halton', 'hammersly', 'random', 'full_factorial', 'gsd'];
const SPACE_FILLING: DoEMethod[] = ['lhs', 'sobol', 'halton', 'hammersly', 'random'];

interface LevelState { method: DoEMethod; nPoints: number; nLevels: number; replicates: number; order: 'random' | 'as_generated' }

export function groupByWholePlot(points: Array<Record<string, any>>) {
  const groups: Array<{ wholePlot: number; rows: Array<Record<string, any>> }> = [];
  for (const p of points) {
    const last = groups[groups.length - 1];
    if (last && last.wholePlot === p.WholePlot) last.rows.push(p);
    else groups.push({ wholePlot: p.WholePlot, rows: [p] });
  }
  return groups;
}

interface Props {
  sessionId: string;
  variables: VariableDetail[];
  onStageSuggestions?: (pending: any[]) => void;
}

/** Small visible caption above each per-level field. */
const FIELD_LABEL = 'block text-[10px] text-muted-foreground';

export function SplitPlotDesignSection({ sessionId, variables, onStageSuggestions }: Props) {
  const present = useMemo(
    () => LEVELS.filter((lvl) => variables.some((v) => v.role !== 'observed' && (v.changes ?? 'easy') === lvl)),
    [variables],
  );
  const [levels, setLevels] = useState<Record<string, LevelState>>(() =>
    Object.fromEntries(LEVELS.map((l) => [l, {
      method: l === 'easy' ? 'lhs' : 'full_factorial', nPoints: 8, nLevels: 2, replicates: 1, order: 'random',
    }])),
  );
  const [easyMode, setEasyMode] = useState<'fresh' | 'crossed'>('fresh');
  const [middleMode, setMiddleMode] = useState<'fresh' | 'crossed'>('fresh');
  const [reference, setReference] = useState<'none' | 'ends' | 'start'>('ends');
  const [seed, setSeed] = useState('');
  const [result, setResult] = useState<SplitPlotDesignResponse | null>(null);
  const [isStaging, setIsStaging] = useState(false);
  const generate = useGenerateSplitPlotDesign(sessionId);

  const update = (lvl: string, patch: Partial<LevelState>) =>
    setLevels((prev) => ({ ...prev, [lvl]: { ...prev[lvl], ...patch } }));

  const handleGenerate = async () => {
    const designs: Record<string, any> = {};
    const replicates: Record<string, number> = {};
    const order: Record<string, 'random' | 'as_generated'> = {};
    for (const lvl of present) {
      const s = levels[lvl];
      designs[lvl] = { method: s.method };
      order[lvl] = s.order;
      if (SPACE_FILLING.includes(s.method)) designs[lvl].n_points = s.nPoints;
      if (s.method === 'full_factorial' || s.method === 'gsd') designs[lvl].n_levels = s.nLevels;
      if (lvl !== 'easy' && s.replicates > 1) replicates[lvl] = s.replicates;
    }
    try {
      const res = await generate.mutateAsync({
        designs,
        replicates,
        order,
        easy_mode: easyMode,
        ...(present.includes('very_hard') && { middle_mode: middleMode }),
        reference: reference === 'none' ? null : { position: reference },
        random_seed: seed ? parseInt(seed) : null,
      });
      setResult(res);
    } catch {
      // The hook already reports the error; keep the previous result.
    }
  };

  const handleDownloadCSV = () => {
    if (!result || result.points.length === 0) return;
    const headers = Object.keys(result.points[0]);
    const csv = [headers.join(','), ...result.points.map((p) => headers.map((h) => p[h]).join(','))].join('\n');
    const url = window.URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `split_plot_design_${result.points.length}runs.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  const handleStage = async () => {
    if (!result) return;
    setIsStaging(true);
    try {
      const experiments = result.points.map(({ Reason, ...rest }) => ({ ...rest, _reason: Reason }));
      const r = await fetch(`/api/v1/sessions/${sessionId}/experiments/staged/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiments, reason: null }),
      });
      if (!r.ok) throw new Error('Failed to stage experiments');
      const { ids = [] } = await r.json();
      onStageSuggestions?.(experiments.map((p, i) => ({ ...p, _queueItemId: ids[i] })));
      toast.success(`${experiments.length} split-plot runs staged`, {
        description: 'Use "Add Point" in the Experiments panel; it shows each setup change.',
      });
    } catch (e: any) {
      toast.error('Failed to stage points: ' + (e?.message || String(e)));
    } finally {
      setIsStaging(false);
    }
  };

  const groups = result ? groupByWholePlot(result.points) : [];
  const nWhole = result ? (result.info.n_plots[present[0]] ?? groups.length) : 0;
  const uniformRuns = groups.length > 0 && groups.every((g) => g.rows.length === groups[0].rows.length)
    ? groups[0].rows.length : null;

  return (
    <div className="space-y-3">
      {present.map((lvl) => {
        const s = levels[lvl];
        const names = variables.filter((v) => v.role !== 'observed' && (v.changes ?? 'easy') === lvl).map((v) => v.name);
        return (
          <div key={lvl} className="rounded border p-2 space-y-2">
            <p className="text-xs font-medium">{LEVEL_LABEL[lvl]}: {names.join(', ')}</p>
            <div className="grid grid-cols-4 gap-2">
              <label className="block space-y-0.5">
                <span className={FIELD_LABEL}>Method</span>
                <select aria-label={`${lvl} method`} value={s.method}
                  onChange={(e) => update(lvl, { method: e.target.value as DoEMethod })}
                  className="w-full px-2 py-1 border border-input rounded bg-background text-xs">
                  {METHODS.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              </label>
              {SPACE_FILLING.includes(s.method) ? (
                <label className="block space-y-0.5">
                  <span className={FIELD_LABEL}>Points</span>
                  <input aria-label={`${lvl} points`} type="number" min={1} value={s.nPoints}
                    onChange={(e) => update(lvl, { nPoints: parseInt(e.target.value) || 1 })}
                    className="w-full px-2 py-1 border border-input rounded bg-background text-xs" />
                </label>
              ) : (
                <label className="block space-y-0.5">
                  <span className={FIELD_LABEL}>Levels</span>
                  <input aria-label={`${lvl} levels`} type="number" min={2} max={5} value={s.nLevels}
                    onChange={(e) => update(lvl, { nLevels: parseInt(e.target.value) || 2 })}
                    className="w-full px-2 py-1 border border-input rounded bg-background text-xs" />
                </label>
              )}
              <label className="block space-y-0.5">
                <span className={FIELD_LABEL}>Order</span>
                <select aria-label={`${lvl} order`} value={s.order}
                  onChange={(e) => update(lvl, { order: e.target.value as LevelState['order'] })}
                  className="w-full px-2 py-1 border border-input rounded bg-background text-xs">
                  <option value="random">random</option>
                  <option value="as_generated">As generated</option>
                </select>
              </label>
              {lvl !== 'easy' && (
                <label className="block space-y-0.5">
                  <span className={FIELD_LABEL}>Replicates</span>
                  <input aria-label={`${lvl} replicates`} type="number" min={1} value={s.replicates}
                    onChange={(e) => update(lvl, { replicates: parseInt(e.target.value) || 1 })}
                    className="w-full px-2 py-1 border border-input rounded bg-background text-xs" />
                </label>
              )}
            </div>
          </div>
        );
      })}
      <div className="grid grid-cols-3 gap-2 text-xs">
        {present.includes('very_hard') && (
          <select aria-label="middle mode" value={middleMode} onChange={(e) => setMiddleMode(e.target.value as any)}
            className="px-2 py-1 border border-input rounded bg-background">
            <option value="fresh">Fresh hard design per outer plot</option>
            <option value="crossed">Same hard design in every outer plot</option>
          </select>
        )}
        <select aria-label="easy mode" value={easyMode} onChange={(e) => setEasyMode(e.target.value as any)}
          className="px-2 py-1 border border-input rounded bg-background">
          <option value="fresh">Fresh design per plot</option>
          <option value="crossed">Same design in every plot</option>
        </select>
        <select aria-label="reference runs" value={reference} onChange={(e) => setReference(e.target.value as any)}
          className="px-2 py-1 border border-input rounded bg-background">
          <option value="ends">Reference run at start and end</option>
          <option value="start">Reference run at start</option>
          <option value="none">No reference run</option>
        </select>
        <input aria-label="split-plot seed" placeholder="Seed (opt)" value={seed} onChange={(e) => setSeed(e.target.value)}
          className="px-2 py-1 border border-input rounded bg-background" />
      </div>
      <button onClick={handleGenerate} disabled={generate.isPending}
        className="w-full px-3 py-2 text-xs bg-primary text-primary-foreground rounded">
        Generate split-plot design
      </button>
      {result && (
        <div className="space-y-2">
          <p className="text-xs">
            {uniformRuns !== null
              ? `${nWhole} whole plots × ${uniformRuns} runs = ${result.n_points} runs`
              : `${nWhole} whole plots, ${result.n_points} runs`}
          </p>
          {result.info.warnings.map((w) => (
            <p key={w} role="alert" className="text-xs text-amber-600">⚠ {w}</p>
          ))}
          <div className="max-h-48 overflow-auto border rounded text-xs">
            {groups.map((g) => (
              <div key={g.wholePlot} className="border-b">
                <p className="px-2 py-1 bg-muted/50 font-medium">Whole plot {g.wholePlot} ({g.rows.length} runs)</p>
                {g.rows.map((r) => (
                  <p key={r.RunOrder} className="px-2 tabular-nums">
                    {r.RunOrder}. {Object.entries(r)
                      .filter(([k]) => !['WholePlot', 'Subplot', 'RunOrder', 'Reason'].includes(k))
                      .map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : v}`).join(', ')}
                    {r.Reason === 'Reference (split-plot)' ? ' (reference)' : ''}
                  </p>
                ))}
              </div>
            ))}
          </div>
          <div className="flex gap-2">
            <button onClick={handleStage} disabled={isStaging}
              className="px-3 py-2 text-xs bg-green-600 text-white rounded">
              {isStaging ? 'Staging...' : 'Stage'}
            </button>
            <button onClick={handleDownloadCSV} className="px-3 py-2 text-xs border rounded hover:bg-accent">
              Export CSV
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
