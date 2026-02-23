/**
 * Optimal Design Panel - Generate D/A/I-optimal experimental designs
 * Supports both quick model types and custom effect specification.
 * Designed to be AI-agent-fillable: all parameters map to a flat JSON request.
 */
import { useState, useMemo } from 'react';
import { useOptimalDesignInfo, useGenerateOptimalDesign } from '../../hooks/api/useExperiments';
import { useVariables } from '../../hooks/api/useVariables';
import type {
  OptimalDesignCriterion,
  OptimalDesignAlgorithm,
  OptimalDesignModelType,
  OptimalDesignInfoResponse,
} from '../../api/types';
import { Download, ListPlus, ChevronDown, ChevronRight, Info } from 'lucide-react';
import { toast } from 'sonner';

const CRITERION_LABELS: Record<OptimalDesignCriterion, string> = {
  D: 'D-optimal (parameter estimation)',
  A: 'A-optimal (min avg variance)',
  I: 'I-optimal (min prediction variance)',
};

const ALGORITHM_LABELS: Record<OptimalDesignAlgorithm, string> = {
  fedorov: 'Fedorov (recommended)',
  modified_fedorov: 'Modified Fedorov',
  detmax: 'DetMax (best quality)',
  simple_exchange: 'Simple Exchange',
  sequential: 'Sequential (fastest)',
};

interface OptimalDesignPanelProps {
  sessionId: string;
  onStageSuggestions?: (pending: any[]) => void;
}

export function OptimalDesignPanel({ sessionId, onStageSuggestions }: OptimalDesignPanelProps) {
  // Model specification
  const [specMode, setSpecMode] = useState<'quick' | 'custom'>('quick');
  const [modelType, setModelType] = useState<OptimalDesignModelType>('quadratic');
  const [selectedEffects, setSelectedEffects] = useState<Set<string>>(new Set());
  const [customEffectText, setCustomEffectText] = useState('');

  // Design parameters
  const [runCountMode, setRunCountMode] = useState<'multiplier' | 'fixed'>('multiplier');
  const [pMultiplier, setPMultiplier] = useState<number>(2.0);
  const [nPoints, setNPoints] = useState<number>(12);
  const [criterion, setCriterion] = useState<OptimalDesignCriterion>('D');
  const [algorithm, setAlgorithm] = useState<OptimalDesignAlgorithm>('fedorov');

  // Advanced parameters
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [nLevels, setNLevels] = useState<number>(5);
  const [maxIter, setMaxIter] = useState<number>(200);
  const [randomSeed, setRandomSeed] = useState<string>('');

  // Results
  const [previewInfo, setPreviewInfo] = useState<OptimalDesignInfoResponse | null>(null);
  const [generatedPoints, setGeneratedPoints] = useState<Array<Record<string, any>> | null>(null);
  const [designInfo, setDesignInfo] = useState<Record<string, any> | null>(null);
  const [isStaging, setIsStaging] = useState(false);

  const { data: variablesData } = useVariables(sessionId);
  const previewMutation = useOptimalDesignInfo(sessionId);
  const generateMutation = useGenerateOptimalDesign(sessionId);

  const hasVariables = variablesData && variablesData.variables.length > 0;
  const variables = variablesData?.variables ?? [];

  // Build available effects from variables
  const availableEffects = useMemo(() => {
    const mainEffects = variables.map(v => v.name);
    const interactions: string[] = [];
    for (let i = 0; i < variables.length; i++) {
      for (let j = i + 1; j < variables.length; j++) {
        interactions.push(`${variables[i].name}*${variables[j].name}`);
      }
    }
    const quadratics: string[] = [];
    for (const v of variables) {
      if (v.type === 'real' || v.type === 'integer' || v.type === 'discrete') {
        quadratics.push(`${v.name}**2`);
      }
    }
    return { mainEffects, interactions, quadratics };
  }, [variables]);

  // Build effects list from selected checkboxes + custom text
  const buildEffectsList = (): string[] => {
    const effects = [...selectedEffects];
    if (customEffectText.trim()) {
      const custom = customEffectText.split(',').map(s => s.trim()).filter(Boolean);
      effects.push(...custom.filter(e => !selectedEffects.has(e)));
    }
    return effects;
  };

  const handlePreview = async () => {
    const request: any = {};
    if (specMode === 'quick') {
      request.model_type = modelType;
    } else {
      const effects = buildEffectsList();
      if (effects.length === 0) {
        toast.error('Select at least one effect');
        return;
      }
      request.effects = effects;
    }
    const info = await previewMutation.mutateAsync(request);
    setPreviewInfo(info);
    // Auto-update n_points to recommended
    setNPoints(info.n_points_recommended);
  };

  const handleGenerate = async () => {
    const request: any = {
      criterion,
      algorithm,
      n_levels: nLevels,
      max_iter: maxIter,
      random_seed: randomSeed ? parseInt(randomSeed) : null,
    };

    if (specMode === 'quick') {
      request.model_type = modelType;
    } else {
      const effects = buildEffectsList();
      if (effects.length === 0) {
        toast.error('Select at least one effect');
        return;
      }
      request.effects = effects;
    }

    if (runCountMode === 'multiplier') {
      request.p_multiplier = pMultiplier;
    } else {
      request.n_points = nPoints;
    }

    const result = await generateMutation.mutateAsync(request);
    setGeneratedPoints(result.points);
    setDesignInfo(result.design_info);
  };

  const handleDownloadCSV = () => {
    if (!generatedPoints || generatedPoints.length === 0) return;
    const headers = Object.keys(generatedPoints[0]);
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
    link.download = `optimal_design_${criterion}_${generatedPoints.length}pts.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  const handleStagePoints = async () => {
    if (!generatedPoints || generatedPoints.length === 0) return;
    setIsStaging(true);
    try {
      const reasonStr = `Optimal DoE (${criterion}-optimal, ${algorithm})`;
      const taggedPoints = generatedPoints.map(p => ({ ...p, _reason: reasonStr }));

      const stageResponse = await fetch(`/api/v1/sessions/${sessionId}/experiments/staged/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiments: generatedPoints, reason: reasonStr })
      });
      if (!stageResponse.ok) throw new Error('Failed to stage experiments');

      await fetch(`/api/v1/sessions/${sessionId}/audit/lock`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lock_type: 'acquisition',
          strategy: reasonStr,
          parameters: {
            method: 'optimal',
            criterion,
            algorithm,
            ...(specMode === 'quick' ? { model_type: modelType } : { effects: buildEffectsList() }),
            ...(runCountMode === 'multiplier' ? { p_multiplier: pMultiplier } : { n_points: nPoints }),
            ...(designInfo && { D_eff: designInfo.D_eff, A_eff: designInfo.A_eff }),
          },
          suggestions: generatedPoints,
          notes: 'Optimal design points staged for execution'
        })
      });

      if (onStageSuggestions) onStageSuggestions(taggedPoints);
      toast.success(`${generatedPoints.length} optimal design points staged`, {
        description: 'Use "Add Point" in Experiments panel to add results'
      });
    } catch (e: any) {
      toast.error('Failed to stage points: ' + (e?.message || String(e)));
    } finally {
      setIsStaging(false);
    }
  };

  const toggleEffect = (effect: string) => {
    setSelectedEffects(prev => {
      const next = new Set(prev);
      if (next.has(effect)) next.delete(effect);
      else next.add(effect);
      return next;
    });
    setPreviewInfo(null); // Invalidate preview
  };

  const selectAllEffects = (effects: string[]) => {
    setSelectedEffects(prev => {
      const next = new Set(prev);
      effects.forEach(e => next.add(e));
      return next;
    });
    setPreviewInfo(null);
  };

  const clearAllEffects = (effects: string[]) => {
    setSelectedEffects(prev => {
      const next = new Set(prev);
      effects.forEach(e => next.delete(e));
      return next;
    });
    setPreviewInfo(null);
  };

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground border-b pb-2">
        Optimal Design (OED)
      </h3>

      {!hasVariables ? (
        <div className="border border-dashed border-muted-foreground/20 rounded p-4 text-center">
          <p className="text-xs text-muted-foreground">Define variables first</p>
        </div>
      ) : (
        <>
          <div className="space-y-2">
            {/* Section 1: Model Specification */}
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Model Specification</label>
              <div className="flex gap-2 mb-2">
                <button
                  onClick={() => { setSpecMode('quick'); setPreviewInfo(null); }}
                  className={`text-xs px-2 py-1 rounded ${specMode === 'quick' ? 'bg-primary text-primary-foreground' : 'border hover:bg-accent'}`}
                >
                  Quick
                </button>
                <button
                  onClick={() => { setSpecMode('custom'); setPreviewInfo(null); }}
                  className={`text-xs px-2 py-1 rounded ${specMode === 'custom' ? 'bg-primary text-primary-foreground' : 'border hover:bg-accent'}`}
                >
                  Custom Effects
                </button>
              </div>

              {specMode === 'quick' ? (
                <select
                  value={modelType}
                  onChange={(e) => { setModelType(e.target.value as OptimalDesignModelType); setPreviewInfo(null); }}
                  className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                >
                  <option value="linear">Linear (main effects only)</option>
                  <option value="interaction">Interaction (main + pairwise)</option>
                  <option value="quadratic">Quadratic (main + pairwise + squared)</option>
                </select>
              ) : (
                <div className="space-y-1.5 max-h-48 overflow-y-auto border rounded p-2 text-xs">
                  {/* Main effects */}
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-muted-foreground">Main Effects</span>
                    <div className="flex gap-1">
                      <button onClick={() => selectAllEffects(availableEffects.mainEffects)} className="text-blue-500 hover:underline">all</button>
                      <button onClick={() => clearAllEffects(availableEffects.mainEffects)} className="text-red-500 hover:underline">none</button>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {availableEffects.mainEffects.map(e => (
                      <label key={e} className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded cursor-pointer ${selectedEffects.has(e) ? 'bg-primary/10 border-primary/30 border' : 'bg-muted/50 border border-transparent'}`}>
                        <input type="checkbox" checked={selectedEffects.has(e)} onChange={() => toggleEffect(e)} className="h-3 w-3" />
                        {e}
                      </label>
                    ))}
                  </div>

                  {/* Interactions */}
                  {availableEffects.interactions.length > 0 && (
                    <>
                      <div className="flex justify-between items-center pt-1">
                        <span className="font-medium text-muted-foreground">Interactions</span>
                        <div className="flex gap-1">
                          <button onClick={() => selectAllEffects(availableEffects.interactions)} className="text-blue-500 hover:underline">all</button>
                          <button onClick={() => clearAllEffects(availableEffects.interactions)} className="text-red-500 hover:underline">none</button>
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {availableEffects.interactions.map(e => (
                          <label key={e} className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded cursor-pointer ${selectedEffects.has(e) ? 'bg-primary/10 border-primary/30 border' : 'bg-muted/50 border border-transparent'}`}>
                            <input type="checkbox" checked={selectedEffects.has(e)} onChange={() => toggleEffect(e)} className="h-3 w-3" />
                            {e.replace('*', ' × ')}
                          </label>
                        ))}
                      </div>
                    </>
                  )}

                  {/* Quadratics */}
                  {availableEffects.quadratics.length > 0 && (
                    <>
                      <div className="flex justify-between items-center pt-1">
                        <span className="font-medium text-muted-foreground">Quadratic</span>
                        <div className="flex gap-1">
                          <button onClick={() => selectAllEffects(availableEffects.quadratics)} className="text-blue-500 hover:underline">all</button>
                          <button onClick={() => clearAllEffects(availableEffects.quadratics)} className="text-red-500 hover:underline">none</button>
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {availableEffects.quadratics.map(e => (
                          <label key={e} className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded cursor-pointer ${selectedEffects.has(e) ? 'bg-primary/10 border-primary/30 border' : 'bg-muted/50 border border-transparent'}`}>
                            <input type="checkbox" checked={selectedEffects.has(e)} onChange={() => toggleEffect(e)} className="h-3 w-3" />
                            {e.replace('**2', '²')}
                          </label>
                        ))}
                      </div>
                    </>
                  )}

                  {/* Custom effects text */}
                  <div className="pt-1">
                    <input
                      type="text"
                      placeholder="Additional effects (comma-separated)"
                      value={customEffectText}
                      onChange={(e) => { setCustomEffectText(e.target.value); setPreviewInfo(null); }}
                      className="w-full px-2 py-1 text-xs border rounded bg-background"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Section 2: Preview */}
            <div>
              <button
                onClick={handlePreview}
                disabled={previewMutation.isPending}
                className="w-full text-xs border px-2 py-1.5 rounded hover:bg-accent disabled:opacity-50 flex items-center justify-center gap-1"
              >
                <Info className="h-3 w-3" />
                {previewMutation.isPending ? 'Loading...' : 'Preview Model Terms'}
              </button>

              {previewInfo && (
                <div className="mt-1.5 border rounded p-2 bg-muted/30 text-xs space-y-1">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Model columns (p):</span>
                    <span className="font-medium">{previewInfo.p_columns}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Min runs:</span>
                    <span>{previewInfo.n_points_minimum}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Recommended runs (2p):</span>
                    <span className="font-medium text-green-600">{previewInfo.n_points_recommended}</span>
                  </div>
                  <details className="pt-1">
                    <summary className="text-muted-foreground cursor-pointer hover:text-foreground">
                      {previewInfo.model_terms.length} model terms
                    </summary>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {previewInfo.model_terms.map((term, i) => (
                        <span key={i} className="px-1.5 py-0.5 bg-background border rounded text-xs">
                          {term}
                        </span>
                      ))}
                    </div>
                  </details>
                </div>
              )}
            </div>

            {/* Section 3: Design Parameters */}
            <div className="space-y-2">
              {/* Run count */}
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Run Count</label>
                <div className="flex gap-2 mb-1">
                  <label className="inline-flex items-center gap-1 text-xs cursor-pointer">
                    <input type="radio" checked={runCountMode === 'multiplier'} onChange={() => setRunCountMode('multiplier')} className="h-3 w-3" />
                    Multiplier (×p)
                  </label>
                  <label className="inline-flex items-center gap-1 text-xs cursor-pointer">
                    <input type="radio" checked={runCountMode === 'fixed'} onChange={() => setRunCountMode('fixed')} className="h-3 w-3" />
                    Fixed count
                  </label>
                </div>
                {runCountMode === 'multiplier' ? (
                  <div className="flex items-center gap-2">
                    <input
                      type="range"
                      min={1} max={5} step={0.5}
                      value={pMultiplier}
                      onChange={(e) => setPMultiplier(parseFloat(e.target.value))}
                      className="flex-1"
                    />
                    <span className="text-xs font-medium w-10 text-right">{pMultiplier}×p</span>
                  </div>
                ) : (
                  <input
                    type="number"
                    min={1} max={10000}
                    value={nPoints}
                    onChange={(e) => setNPoints(parseInt(e.target.value) || 1)}
                    className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                  />
                )}
              </div>

              {/* Criterion */}
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Optimality Criterion</label>
                <select
                  value={criterion}
                  onChange={(e) => setCriterion(e.target.value as OptimalDesignCriterion)}
                  className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                >
                  {(Object.entries(CRITERION_LABELS) as [OptimalDesignCriterion, string][]).map(([val, label]) => (
                    <option key={val} value={val}>{label}</option>
                  ))}
                </select>
              </div>

              {/* Algorithm */}
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Algorithm</label>
                <select
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value as OptimalDesignAlgorithm)}
                  className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                >
                  {(Object.entries(ALGORITHM_LABELS) as [OptimalDesignAlgorithm, string][]).map(([val, label]) => (
                    <option key={val} value={val}>{label}</option>
                  ))}
                </select>
              </div>

              {/* Advanced */}
              <div>
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                >
                  {showAdvanced ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                  Advanced
                </button>
                {showAdvanced && (
                  <div className="mt-1 space-y-1.5 pl-3 border-l">
                    <div className="flex items-center gap-2">
                      <label className="text-xs text-muted-foreground w-24">Grid levels:</label>
                      <input
                        type="number" min={2} max={20} value={nLevels}
                        onChange={(e) => setNLevels(parseInt(e.target.value) || 5)}
                        className="w-20 px-2 py-1 text-xs border rounded bg-background"
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <label className="text-xs text-muted-foreground w-24">Max iterations:</label>
                      <input
                        type="number" min={10} max={10000} value={maxIter}
                        onChange={(e) => setMaxIter(parseInt(e.target.value) || 200)}
                        className="w-20 px-2 py-1 text-xs border rounded bg-background"
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <label className="text-xs text-muted-foreground w-24">Random seed:</label>
                      <input
                        type="text" placeholder="Optional"
                        value={randomSeed}
                        onChange={(e) => setRandomSeed(e.target.value)}
                        className="w-20 px-2 py-1 text-xs border rounded bg-background"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Section 4: Generate */}
            <button
              onClick={handleGenerate}
              disabled={generateMutation.isPending || !hasVariables}
              className="w-full bg-primary text-primary-foreground px-3 py-2 rounded text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
            >
              {generateMutation.isPending ? 'Generating...' : 'Generate Optimal Design'}
            </button>

            {/* Design quality metrics */}
            {designInfo && (
              <div className="border rounded p-2 bg-green-50 dark:bg-green-950/30 text-xs space-y-0.5">
                <div className="font-medium text-green-700 dark:text-green-400">
                  Design Quality
                </div>
                <div className="grid grid-cols-2 gap-x-4">
                  {designInfo.D_eff != null && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">D-efficiency:</span>
                      <span className="font-medium">{designInfo.D_eff.toFixed(1)}%</span>
                    </div>
                  )}
                  {designInfo.A_eff != null && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">A-efficiency:</span>
                      <span className="font-medium">{designInfo.A_eff.toFixed(1)}%</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Criterion:</span>
                    <span>{designInfo.criterion}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Runs:</span>
                    <span>{designInfo.n_runs}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Results table */}
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
          </div>
        </>
      )}
    </div>
  );
}
