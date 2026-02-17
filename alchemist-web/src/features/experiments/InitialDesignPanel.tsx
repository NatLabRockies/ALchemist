/**
 * Initial Design Panel - Generate DoE (Design of Experiments) points
 * For autonomous optimization workflows
 */
import { useState } from 'react';
import { useGenerateInitialDesign } from '../../hooks/api/useExperiments';
import { useVariables } from '../../hooks/api/useVariables';
import type { DoEMethod, LHSCriterion, CCDAlpha, CCDFace } from '../../api/types';
import { Download, ListPlus } from 'lucide-react';
import { toast } from 'sonner';

const SPACE_FILLING_METHODS: DoEMethod[] = ['lhs', 'sobol', 'halton', 'hammersly', 'random'];
const CLASSICAL_METHODS: DoEMethod[] = ['full_factorial', 'fractional_factorial', 'ccd', 'box_behnken'];
const SCREENING_METHODS: DoEMethod[] = ['plackett_burman', 'gsd'];

const METHOD_LABELS: Record<DoEMethod, string> = {
  lhs: 'LHS',
  sobol: 'Sobol',
  halton: 'Halton',
  hammersly: 'Hammersly',
  random: 'Random',
  full_factorial: 'Full Factorial',
  fractional_factorial: 'Fractional Factorial',
  ccd: 'CCD',
  box_behnken: 'Box-Behnken',
  plackett_burman: 'Plackett-Burman',
  gsd: 'GSD',
};

interface InitialDesignPanelProps {
  sessionId: string;
  onStageSuggestions?: (pending: any[]) => void;
}

export function InitialDesignPanel({ sessionId, onStageSuggestions }: InitialDesignPanelProps) {
  const [method, setMethod] = useState<DoEMethod>('lhs');
  const [nPoints, setNPoints] = useState<number>(10);
  const [randomSeed, setRandomSeed] = useState<string>('');
  const [lhsCriterion, setLhsCriterion] = useState<LHSCriterion>('maximin');
  // Classical design state
  const [nLevels, setNLevels] = useState<number>(2);
  const [nCenter, setNCenter] = useState<number>(1);
  const [generators, setGenerators] = useState<string>('');
  const [ccdAlpha, setCcdAlpha] = useState<CCDAlpha>('orthogonal');
  const [ccdFace, setCcdFace] = useState<CCDFace>('circumscribed');
  // GSD parameters
  const [gsdReduction, setGsdReduction] = useState<number>(2);

  const [generatedPoints, setGeneratedPoints] = useState<Array<Record<string, any>> | null>(null);
  const [isStaging, setIsStaging] = useState(false);

  const { data: variablesData } = useVariables(sessionId);
  const generateDesign = useGenerateInitialDesign(sessionId);

  const hasVariables = variablesData && variablesData.variables.length > 0;
  const hasCategoricals = variablesData?.variables.some((v: any) => v.type === 'categorical') ?? false;
  const isSpaceFilling = (SPACE_FILLING_METHODS as readonly string[]).includes(method);
  const hasSeed = method === 'random' || method === 'lhs';
  const hasCenterPts = ['full_factorial', 'fractional_factorial', 'ccd', 'box_behnken', 'plackett_burman'].includes(method);
  const NO_CATEGORICALS: DoEMethod[] = ['fractional_factorial', 'ccd', 'box_behnken', 'plackett_burman'];
  const isCategoricalIncompat = hasCategoricals && NO_CATEGORICALS.includes(method);

  const handleGenerate = async () => {
    const request: any = {
      method,
      random_seed: hasSeed && randomSeed ? parseInt(randomSeed) : null,
    };

    if (isSpaceFilling) {
      request.n_points = nPoints;
      if (method === 'lhs') request.lhs_criterion = lhsCriterion;
    } else {
      // Classical / screening methods
      if (hasCenterPts) request.n_center = nCenter;
      if (method === 'full_factorial') request.n_levels = nLevels;
      if (method === 'fractional_factorial' && generators.trim()) request.generators = generators.trim();
      if (method === 'ccd') {
        request.ccd_alpha = ccdAlpha;
        request.ccd_face = ccdFace;
      }
      if (method === 'gsd') {
        request.n_levels = nLevels;
        request.gsd_reduction = gsdReduction;
      }
    }

    const result = await generateDesign.mutateAsync(request);
    setGeneratedPoints(result.points);
  };

  const handleDownloadCSV = () => {
    if (!generatedPoints || generatedPoints.length === 0) return;

    // Get column headers from first point
    const headers = Object.keys(generatedPoints[0]);

    // Build CSV
    const csvRows = [
      headers.join(','),
      ...generatedPoints.map(point =>
        headers.map(h => point[h]).join(',')
      )
    ];

    const csvContent = csvRows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `initial_design_${method}_${generatedPoints.length}pts.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  const handleStagePoints = async () => {
    if (!generatedPoints || generatedPoints.length === 0) return;

    setIsStaging(true);
    try {
      // Tag points with reason for Add Point dialog
      const taggedPoints = generatedPoints.map(p => ({
        ...p,
        _reason: `Initial DoE (${METHOD_LABELS[method]})`
      }));

      // 1. Stage to API (for persistence across page reloads)
      const stageResponse = await fetch(`/api/v1/sessions/${sessionId}/experiments/staged/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          experiments: generatedPoints,
          reason: `Initial DoE (${METHOD_LABELS[method]})`
        })
      });

      if (!stageResponse.ok) {
        throw new Error('Failed to stage experiments');
      }

      // 2. Also log to audit for reproducibility
      await fetch(`/api/v1/sessions/${sessionId}/audit/lock`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lock_type: 'acquisition',
          strategy: `Initial DoE (${METHOD_LABELS[method]})`,
          parameters: {
            method,
            ...(isSpaceFilling && { n_points: nPoints }),
            ...(hasCenterPts && { n_center: nCenter }),
            ...(hasSeed && randomSeed && { random_seed: randomSeed }),
            ...(method === 'lhs' && { lhs_criterion: lhsCriterion }),
            ...(method === 'full_factorial' && { n_levels: nLevels }),
            ...(method === 'fractional_factorial' && generators.trim() && { generators: generators.trim() }),
            ...(method === 'ccd' && { ccd_alpha: ccdAlpha, ccd_face: ccdFace }),
            ...(method === 'gsd' && { n_levels: nLevels, gsd_reduction: gsdReduction }),
          },
          suggestions: generatedPoints,
          notes: 'Initial design points staged for execution'
        })
      });

      // 3. Update local React state for immediate UI feedback
      if (onStageSuggestions) {
        onStageSuggestions(taggedPoints);
      }

      toast.success(`${generatedPoints.length} DoE points staged`, {
        description: 'Use "Add Point" in Experiments panel to add results'
      });

    } catch (e: any) {
      toast.error('Failed to stage points: ' + (e?.message || String(e)));
      console.error('Failed to stage DoE points:', e);
    } finally {
      setIsStaging(false);
    }
  };

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground border-b pb-2">
        Initial Design (DoE)
      </h3>

      {!hasVariables ? (
        <div className="border border-dashed border-muted-foreground/20 rounded p-4 text-center">
          <p className="text-xs text-muted-foreground">Define variables first</p>
        </div>
      ) : (
        <>
          {/* Compact Config Form */}
          <div className="space-y-2">
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Method</label>
              <select
                value={method}
                onChange={(e) => setMethod(e.target.value as DoEMethod)}
                className="w-full px-2 py-1.5 text-sm border rounded bg-background"
              >
                <optgroup label="Space-Filling">
                  {SPACE_FILLING_METHODS.map((m) => (
                    <option key={m} value={m}>{METHOD_LABELS[m]}</option>
                  ))}
                </optgroup>
                <optgroup label="Classical RSM">
                  {CLASSICAL_METHODS.map((m) => (
                    <option key={m} value={m}>{METHOD_LABELS[m]}</option>
                  ))}
                </optgroup>
                <optgroup label="Screening">
                  {SCREENING_METHODS.map((m) => (
                    <option key={m} value={m}>{METHOD_LABELS[m]}</option>
                  ))}
                </optgroup>
              </select>
            </div>

            {/* Method info line */}
            {method === 'ccd' && (
              <p className="text-xs text-muted-foreground italic">
                2^k factorial + 2k axial + center runs
              </p>
            )}
            {method === 'box_behnken' && (
              <p className="text-xs text-muted-foreground italic">
                Requires 3+ continuous variables
              </p>
            )}
            {method === 'fractional_factorial' && (
              <p className="text-xs text-muted-foreground italic">
                2-level screening design
              </p>
            )}
            {method === 'full_factorial' && (
              <p className="text-xs text-muted-foreground italic">
                All combinations of factor levels
              </p>
            )}
            {method === 'plackett_burman' && (
              <p className="text-xs text-muted-foreground italic">
                Ultra-efficient 2-level main-effect screening
              </p>
            )}
            {method === 'gsd' && (
              <p className="text-xs text-muted-foreground italic">
                Fractional design for mixed/multi-level factors
              </p>
            )}

            {/* Categorical incompatibility warning */}
            {isCategoricalIncompat && (
              <div className="bg-amber-500/10 border border-amber-500/30 rounded px-3 py-2">
                <p className="text-xs text-amber-600 dark:text-amber-400">
                  ⚠ {METHOD_LABELS[method]} does not support categorical variables.
                  <br />
                  Compatible methods: Random, LHS, Sobol, Halton, Hammersly, Full Factorial, GSD
                </p>
              </div>
            )}

            {/* Space-filling: n_points */}
            {isSpaceFilling && (
              <div className={`grid ${hasSeed ? 'grid-cols-2' : 'grid-cols-1'} gap-2`}>
                <div>
                  <label className="text-xs text-muted-foreground mb-1 block">Points</label>
                  <input
                    type="number"
                    value={nPoints}
                    onChange={(e) => setNPoints(parseInt(e.target.value) || 10)}
                    min={5}
                    max={100}
                    className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                  />
                </div>
                {hasSeed && (
                  <div>
                    <label className="text-xs text-muted-foreground mb-1 block">Seed (opt)</label>
                    <input
                      type="number"
                      value={randomSeed}
                      onChange={(e) => setRandomSeed(e.target.value)}
                      placeholder="Auto"
                      className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                    />
                  </div>
                )}
              </div>
            )}

            {/* Classical/screening with center points */}
            {!isSpaceFilling && hasCenterPts && (
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Center points</label>
                <input
                  type="number"
                  value={nCenter}
                  onChange={(e) => setNCenter(parseInt(e.target.value) || 0)}
                  min={0}
                  max={10}
                  className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                />
              </div>
            )}

            {/* LHS criterion */}
            {method === 'lhs' && (
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">LHS Criterion</label>
                <select
                  value={lhsCriterion}
                  onChange={(e) => setLhsCriterion(e.target.value as LHSCriterion)}
                  className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                >
                  <option value="maximin">Maximin</option>
                  <option value="correlation">Correlation</option>
                  <option value="ratio">Ratio</option>
                </select>
              </div>
            )}

            {/* Full factorial: n_levels */}
            {method === 'full_factorial' && (
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Levels per factor</label>
                <select
                  value={nLevels}
                  onChange={(e) => setNLevels(parseInt(e.target.value))}
                  className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                >
                  <option value={2}>2 (min/max)</option>
                  <option value={3}>3 (min/center/max)</option>
                </select>
              </div>
            )}

            {/* Fractional factorial: generators */}
            {method === 'fractional_factorial' && (
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Generators (opt)</label>
                <input
                  type="text"
                  value={generators}
                  onChange={(e) => setGenerators(e.target.value)}
                  placeholder='e.g. "a b ab" (auto if blank)'
                  className="w-full px-2 py-1.5 text-sm border rounded bg-background font-mono"
                />
              </div>
            )}

            {/* CCD: alpha + face */}
            {method === 'ccd' && (
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs text-muted-foreground mb-1 block">Alpha</label>
                  <select
                    value={ccdAlpha}
                    onChange={(e) => setCcdAlpha(e.target.value as CCDAlpha)}
                    className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                  >
                    <option value="orthogonal">Orthogonal</option>
                    <option value="rotatable">Rotatable</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs text-muted-foreground mb-1 block">Face</label>
                  <select
                    value={ccdFace}
                    onChange={(e) => setCcdFace(e.target.value as CCDFace)}
                    className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                  >
                    <option value="circumscribed">Circumscribed</option>
                    <option value="inscribed">Inscribed</option>
                    <option value="faced">Faced</option>
                  </select>
                </div>
              </div>
            )}

            {/* GSD: levels + reduction */}
            {method === 'gsd' && (
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs text-muted-foreground mb-1 block">Levels per factor</label>
                  <select
                    value={nLevels}
                    onChange={(e) => setNLevels(parseInt(e.target.value))}
                    className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                  >
                    <option value={2}>2 (min/max)</option>
                    <option value={3}>3 (min/center/max)</option>
                    <option value={4}>4</option>
                    <option value={5}>5</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs text-muted-foreground mb-1 block">Reduction</label>
                  <select
                    value={gsdReduction}
                    onChange={(e) => setGsdReduction(parseInt(e.target.value))}
                    className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                  >
                    <option value={2}>÷2</option>
                    <option value={3}>÷3</option>
                    <option value={4}>÷4</option>
                    <option value={5}>÷5</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          <button
            onClick={handleGenerate}
            disabled={generateDesign.isPending || !hasVariables || isCategoricalIncompat}
            className="w-full bg-primary text-primary-foreground px-3 py-2 rounded text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
          >
            {generateDesign.isPending ? 'Generating...' : 'Generate Design'}
          </button>

          {/* Results - Compact */}
          {generatedPoints && generatedPoints.length > 0 && (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-muted-foreground">
                  {generatedPoints.length} points generated
                </span>
                <div className="flex gap-2">
                  <button
                    onClick={handleStagePoints}
                    disabled={isStaging}
                    className="flex items-center gap-1 text-xs bg-green-600 text-white px-2 py-1 rounded hover:bg-green-700 disabled:opacity-50"
                    title="Stage points for execution via Add Point dialog"
                  >
                    <ListPlus className="h-3 w-3" />
                    {isStaging ? 'Staging...' : 'Stage'}
                  </button>
                  <button
                    onClick={handleDownloadCSV}
                    className="flex items-center gap-1 text-xs border px-2 py-1 rounded hover:bg-accent"
                  >
                    <Download className="h-3 w-3" />
                    CSV
                  </button>
                </div>
              </div>

              <div className="border rounded overflow-hidden">
                <div className="overflow-x-auto max-h-48">
                  <table className="w-full text-xs">
                    <thead className="bg-muted/50 border-b sticky top-0">
                      <tr>
                        {Object.keys(generatedPoints[0]).map((key) => (
                          <th key={key} className="px-2 py-1 text-left font-medium">
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y">
                      {generatedPoints.map((point, idx) => (
                        <tr key={idx} className="hover:bg-accent/50">
                          {Object.values(point).map((val, i) => (
                            <td key={i} className="px-2 py-1 tabular-nums">
                              {typeof val === 'number' ? val.toFixed(3) : val}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
