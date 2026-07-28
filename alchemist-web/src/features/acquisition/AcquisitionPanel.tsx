/**
 * Acquisition Panel - Next Experiment Suggestions
 * Mimics desktop UI acquisition_panel.py layout and functionality
 */
import { useState } from 'react';
import { useModelInfo } from '../../hooks/api/useModels';
import { useSuggestNext, useFindOptimum } from '../../hooks/api/useAcquisition';
import type { 
  OptimizationGoal,
  AcquisitionRequest,
  ModelBackend
} from '../../api/types';
import { Play, Info, Target } from 'lucide-react';

interface AcquisitionPanelProps {
  sessionId: string;
  modelBackend?: ModelBackend | null;
  // optional props to lift pending suggestions to a parent (App)
  pendingSuggestions?: any[];
  onStageSuggestions?: (pending: any[]) => void;
}

export function AcquisitionPanel({ sessionId, modelBackend, pendingSuggestions: pendingFromProps, onStageSuggestions }: AcquisitionPanelProps) {
  // State for sklearn options
  const [skStrategy, setSkStrategy] = useState<string>('EI');
  const [skGoal, setSkGoal] = useState<OptimizationGoal>('maximize');
  const [skXi, setSkXi] = useState(0.01);
  const [skKappa, setSkKappa] = useState(1.96);
  
  // State for BoTorch options
  const [btAcqType, setBtAcqType] = useState<'regular' | 'batch' | 'exploratory'>('regular');
  const [btRegularStrategy, setBtRegularStrategy] = useState<string>('EI');
  const [btBatchStrategy, setBtBatchStrategy] = useState<string>('qEI');
  const [btBatchSize, setBtBatchSize] = useState(2);
  const [btBeta, setBtBeta] = useState(0.5);
  const [btGoal, setBtGoal] = useState<OptimizationGoal>('maximize');
  
  // State for Model Optimum
  const [optimumGoal, setOptimumGoal] = useState<OptimizationGoal>('maximize');
  
  // API hooks
  const { data: modelInfo } = useModelInfo(sessionId);
  const suggestNext = useSuggestNext(sessionId);
  const findOptimum = useFindOptimum(sessionId);
  // Local fallback state when parent doesn't provide staging handlers
  const [localPendingSuggestions, setLocalPendingSuggestions] = useState<any[]>([])
  const pendingSuggestions = pendingFromProps ?? localPendingSuggestions
  
  // Determine which backend is active
  const activeBackend = modelBackend || modelInfo?.backend || 'sklearn';
  const modelTrained = modelInfo?.is_trained || false;
  
  // Strategy descriptions
  const strategyDescriptions: Record<string, string> = {
    'EI': 'Expected Improvement (EI) balances exploration and exploitation by calculating the expected improvement over the current best observed value.',
    'PI': 'Probability of Improvement (PI) selects points based on the probability that they will improve over the current best observed value.',
    'UCB': 'Upper Confidence Bound (UCB) balances exploration and exploitation by selecting points where the upper confidence bound is highest.',
    'gp_hedge': 'GP Hedge automatically balances between EI, PI, and UCB strategies based on their past performance.',
    'logEI': 'Log Expected Improvement is a numerically stable version of EI that calculates the log of the expected improvement.',
    'logPI': 'Log Probability of Improvement is a numerically stable version of PI that calculates the log of the probability of improvement.',
    'qEI': 'q-Expected Improvement is a batch version of EI that selects multiple points simultaneously while accounting for interactions between selections.',
    'qUCB': 'q-Upper Confidence Bound is a batch version of UCB that selects multiple points simultaneously while balancing exploration and exploitation.',
    'qNIPV': 'q-Negative Integrated Posterior Variance selects points to reduce overall model uncertainty. This is purely exploratory for active learning, not optimization.'
  };
  
  const handleRunStrategy = async () => {
    if (!modelTrained) {
      return;
    }
    
    let request: AcquisitionRequest;
    let strategyName: string;
    
    if (activeBackend === 'sklearn') {
      strategyName = skStrategy;
      request = {
        strategy: skStrategy as any,
        goal: skGoal,
        n_suggestions: 1,
      };
      
      // Add parameters based on strategy
      if (skStrategy === 'EI' || skStrategy === 'PI') {
        request.xi = skXi;
      } else if (skStrategy === 'UCB') {
        request.kappa = skKappa;
      } else if (skStrategy === 'gp_hedge') {
        request.xi = skXi;
        request.kappa = skKappa;
      }
    } else {
      // BoTorch
      if (btAcqType === 'regular') {
        strategyName = btRegularStrategy;
        request = {
          strategy: btRegularStrategy as any,
          goal: btGoal,
          n_suggestions: 1,
        };
      } else if (btAcqType === 'batch') {
        strategyName = btBatchStrategy;
        request = {
          strategy: btBatchStrategy as any,
          goal: btGoal,
          n_suggestions: btBatchSize,
        };
        if (btBatchStrategy === 'qUCB') {
          request.kappa = btBeta; // BoTorch uses beta, but API expects kappa
        }
      } else {
        // Exploratory
        strategyName = 'qNIPV';
        request = {
          strategy: 'qNIPV',
          goal: btGoal,
          n_suggestions: 1,
        };
      }
    }
    
    try {
      const result = await suggestNext.mutateAsync(request);
      console.log('Suggestions:', result.suggestions);
      // Stage suggestions with strategy metadata (mirrors desktop UX)
      if (result?.suggestions && result.suggestions.length > 0) {
        // Tag each suggestion with acquisition strategy for auto-fill in Add Point dialog
        const taggedSuggestions = result.suggestions.map((s: any) => ({
          ...s,
          _reason: strategyName,  // Desktop workflow: tag with strategy name
          _strategyParams: request,  // Store full request for audit log
          // Iteration this acquisition round will be recorded under (shared by
          // the whole batch). Shown in the Add Point dialog and written on save.
          ...(result.iteration != null ? { Iteration: result.iteration } : {}),
        }));
        
        if (onStageSuggestions) {
          onStageSuggestions(taggedSuggestions);
        } else {
          setLocalPendingSuggestions(taggedSuggestions);
        }
      }
    } catch (error) {
      console.error('Failed to get suggestions:', error);
    }
  };
  
  const handleFindOptimum = async () => {
    if (!modelTrained) {
      return;
    }
    
    try {
      const result = await findOptimum.mutateAsync({ goal: optimumGoal });
      console.log('Found optimum:', result.optimum);
    } catch (error) {
      console.error('Failed to find optimum:', error);
    }
  };
  
  // Get current strategy for description
  const getCurrentStrategy = () => {
    if (activeBackend === 'sklearn') {
      return skStrategy;
    } else {
      if (btAcqType === 'regular') return btRegularStrategy;
      if (btAcqType === 'batch') return btBatchStrategy;
      return 'qNIPV';
    }
  };
  
  return (
    <div className="rounded-lg border bg-card p-4">
      <h2 className="text-sm font-semibold mb-4 uppercase tracking-wide text-muted-foreground border-b pb-2">Acquisition Functions</h2>
      
      {/* Scikit-learn Options */}
      {activeBackend === 'sklearn' && (
        <div className="space-y-2.5">
          <div>
            <label className="block text-xs font-medium mb-1.5">Acquisition Strategy</label>
            <select
              value={skStrategy}
              onChange={(e) => setSkStrategy(e.target.value)}
              disabled={!modelTrained}
              className="w-full px-2.5 py-1.5 text-xs border rounded-md disabled:opacity-50 disabled:cursor-not-allowed bg-background"
            >
              <option value="EI">Expected Improvement (EI)</option>
              <option value="UCB">Upper Confidence Bound (UCB)</option>
              <option value="PI">Probability of Improvement (PI)</option>
              <option value="gp_hedge">GP Hedge (Auto-balance)</option>
            </select>
          </div>
          
          <div>
            <label className="block text-xs font-medium mb-1.5">Optimization Goal</label>
            <div className="flex gap-1.5">
              <button
                onClick={() => setSkGoal('maximize')}
                disabled={!modelTrained}
                className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  skGoal === 'maximize'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }`}
              >
                Maximize
              </button>
              <button
                onClick={() => setSkGoal('minimize')}
                disabled={!modelTrained}
                className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  skGoal === 'minimize'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }`}
              >
                Minimize
              </button>
            </div>
          </div>
          
          <div className="border-t pt-3">
            <label className="block text-xs font-medium mb-2">Acquisition Parameters</label>
            
            {/* Xi parameter (for EI, PI, GP Hedge) */}
            {(skStrategy === 'EI' || skStrategy === 'PI' || skStrategy === 'gp_hedge') && (
              <div className="mb-3">
                <div className="flex justify-between items-center mb-1.5">
                  <span className="text-xs">ξ (xi):</span>
                  <span className="text-xs font-medium">{skXi.toFixed(4)}</span>
                </div>
                <input
                  type="range"
                  min="0.0001"
                  max="0.1"
                  step="0.0001"
                  value={skXi}
                  onChange={(e) => setSkXi(parseFloat(e.target.value))}
                  disabled={!modelTrained}
                  className="w-full disabled:opacity-50 disabled:cursor-not-allowed"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Higher values favor exploration over exploitation
                </p>
              </div>
            )}
            
            {/* Kappa parameter (for UCB, GP Hedge) */}
            {(skStrategy === 'UCB' || skStrategy === 'gp_hedge') && (
              <div className="mb-3">
                <div className="flex justify-between items-center mb-1.5">
                  <span className="text-xs">κ (kappa):</span>
                  <span className="text-xs font-medium">{skKappa.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="5.0"
                  step="0.1"
                  value={skKappa}
                  onChange={(e) => setSkKappa(parseFloat(e.target.value))}
                  disabled={!modelTrained}
                  className="w-full disabled:opacity-50 disabled:cursor-not-allowed"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Higher values increase exploration
                </p>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* BoTorch Options */}
      {activeBackend === 'botorch' && (
        <div className="space-y-2.5">
          <div>
            <label className="block text-xs font-medium mb-1.5">Acquisition Type</label>
            <div className="flex gap-2">
              <button
                onClick={() => setBtAcqType('regular')}
                disabled={!modelTrained}
                className={`flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  btAcqType === 'regular'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }`}
              >
                Regular
              </button>
              <button
                onClick={() => setBtAcqType('batch')}
                disabled={!modelTrained}
                className={`flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  btAcqType === 'batch'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }`}
              >
                Batch
              </button>
              <button
                onClick={() => setBtAcqType('exploratory')}
                disabled={!modelTrained}
                className={`flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  btAcqType === 'exploratory'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }`}
              >
                Exploratory
              </button>
            </div>
          </div>
          
          {/* Regular Acquisition */}
          {btAcqType === 'regular' && (
            <>
              <div>
                <label className="block text-xs font-medium mb-1.5">Acquisition Function</label>
                <select
                  value={btRegularStrategy}
                  onChange={(e) => setBtRegularStrategy(e.target.value)}
                  disabled={!modelTrained}
                  className="w-full px-2.5 py-1.5 text-xs border rounded-md disabled:opacity-50 disabled:cursor-not-allowed bg-background"
                >
                  <option value="EI">Expected Improvement</option>
                  <option value="logEI">Log Expected Improvement</option>
                  <option value="PI">Probability of Improvement</option>
                  <option value="logPI">Log Probability of Improvement</option>
                  <option value="UCB">Upper Confidence Bound</option>
                </select>
              </div>
              
              <div>
                <label className="block text-xs font-medium mb-1.5">Optimization Goal</label>
                <div className="flex gap-2">
                  <button
                    onClick={() => setBtGoal('maximize')}
                    disabled={!modelTrained}
                    className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                      btGoal === 'maximize'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted text-muted-foreground hover:bg-muted/80'
                    }`}
                  >
                    Maximize
                  </button>
                  <button
                    onClick={() => setBtGoal('minimize')}
                    disabled={!modelTrained}
                    className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                      btGoal === 'minimize'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted text-muted-foreground hover:bg-muted/80'
                    }`}
                  >
                    Minimize
                  </button>
                </div>
              </div>
            </>
          )}
          
          {/* Batch Acquisition */}
          {btAcqType === 'batch' && (
            <>
              <div>
                <label className="block text-xs font-medium mb-1.5">Batch Acquisition Function</label>
                <select
                  value={btBatchStrategy}
                  onChange={(e) => setBtBatchStrategy(e.target.value)}
                  disabled={!modelTrained}
                  className="w-full px-2.5 py-1.5 text-xs border rounded-md disabled:opacity-50 disabled:cursor-not-allowed bg-background"
                >
                  <option value="qEI">q-Expected Improvement</option>
                  <option value="qUCB">q-Upper Confidence Bound</option>
                </select>
              </div>
              
              <div>
                <label className="block text-xs font-medium mb-1.5">Batch Size (q)</label>
                <select
                  value={btBatchSize}
                  onChange={(e) => setBtBatchSize(parseInt(e.target.value))}
                  disabled={!modelTrained}
                  className="w-full px-2.5 py-1.5 text-xs border rounded-md disabled:opacity-50 disabled:cursor-not-allowed bg-background"
                >
                  {[2, 3, 4, 5, 6, 7, 8, 9, 10].map((size) => (
                    <option key={size} value={size}>{size}</option>
                  ))}
                </select>
              </div>
              
              {btBatchStrategy === 'qUCB' && (
                <div>
                  <div className="flex justify-between items-center mb-1.5">
                    <span className="text-xs">β (beta):</span>
                    <span className="text-xs font-medium">{btBeta.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="2.0"
                    step="0.1"
                    value={btBeta}
                    onChange={(e) => setBtBeta(parseFloat(e.target.value))}
                    disabled={!modelTrained}
                    className="w-full disabled:opacity-50 disabled:cursor-not-allowed"
                  />
                </div>
              )}
              
              <div>
                <label className="block text-xs font-medium mb-1.5">Optimization Goal</label>
                <div className="flex gap-2">
                  <button
                    onClick={() => setBtGoal('maximize')}
                    disabled={!modelTrained}
                    className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                      btGoal === 'maximize'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted text-muted-foreground hover:bg-muted/80'
                    }`}
                  >
                    Maximize
                  </button>
                  <button
                    onClick={() => setBtGoal('minimize')}
                    disabled={!modelTrained}
                    className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                      btGoal === 'minimize'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted text-muted-foreground hover:bg-muted/80'
                    }`}
                  >
                    Minimize
                  </button>
                </div>
              </div>
            </>
          )}
          
          {/* Exploratory Acquisition */}
          {btAcqType === 'exploratory' && (
            <div className="p-3 bg-muted/30 rounded-lg">
              <h3 className="font-semibold text-xs mb-1.5">Integrated Posterior Variance</h3>
              <p className="text-xs text-muted-foreground">
                This purely exploratory acquisition function selects points to reduce model uncertainty, 
                without considering optimization goals.
              </p>
            </div>
          )}
          
          {/* Strategy Description */}
          <div className="p-2.5 bg-muted/20 rounded-lg border border-muted">
            <div className="flex items-start gap-2">
              <Info className="w-4 h-4 text-muted-foreground mt-0.5 flex-shrink-0" />
              <p className="text-xs text-muted-foreground">
                {strategyDescriptions[getCurrentStrategy()]}
              </p>
            </div>
          </div>
        </div>
      )}
      
      {/* Run Strategy Button */}
      <button
        onClick={handleRunStrategy}
        disabled={!modelTrained || suggestNext.isPending}
        className="w-full mt-4 bg-primary text-primary-foreground px-3 py-2 rounded-md text-xs hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed font-medium flex items-center justify-center gap-2"
      >
        <Play className="w-3.5 h-3.5" />
        {suggestNext.isPending ? 'Generating Suggestions...' : 'Run Acquisition Strategy'}
      </button>
      
      {/* Model not trained message */}
      {!modelTrained && (
        <p className="mt-3 text-sm text-muted-foreground text-center">
          Train a model first to enable acquisition functions
        </p>
      )}
      
      {/* Display suggestions - now staged to App-level state for Add Point in Experiments panel */}
      {suggestNext.isSuccess && suggestNext.data && (
        <div className="mt-6 p-4 bg-muted/50 rounded-lg">
          <h3 className="font-semibold text-xs uppercase tracking-wide text-muted-foreground mb-2">Suggested Experiments</h3>
          <div className="space-y-2">
            {suggestNext.data.suggestions.map((suggestion, idx) => (
              <div key={idx} className="p-2.5 bg-background rounded border text-xs">
                <div className="font-medium mb-1">Suggestion {idx + 1}:</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {Object.entries(suggestion).map(([key, value]) => (
                    <div key={key}>
                      <span className="text-muted-foreground">{key}:</span>{' '}
                      <span className="font-medium">
                        {typeof value === 'number' ? value.toFixed(3) : value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-3 flex flex-col gap-2">
            <button
              onClick={async () => {
                // Stage to both API and audit log
                const { toast } = await import('sonner');
                try {
                  // Extract strategy info from first pending suggestion
                  const strategyInfo = pendingSuggestions[0]?._strategyParams;
                  const strategyName = pendingSuggestions[0]?._reason || 'Unknown';
                  
                  // Clean suggestions (remove internal metadata)
                  const cleanSuggestions = pendingSuggestions.map(s => {
                    const { _reason, _strategyParams, ...rest } = s;
                    return rest;
                  });
                  
                  // 1. Stage to experiments API (for persistence and queue management)
                  const stageResponse = await fetch(`/api/v1/sessions/${sessionId}/experiments/staged/batch`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      experiments: cleanSuggestions,
                      reason: strategyName
                    })
                  });
                  
                  if (!stageResponse.ok) throw new Error('Failed to stage experiments');
                  
                  // 2. Also log to audit for reproducibility
                  const auditResponse = await fetch(`/api/v1/sessions/${sessionId}/audit/lock`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      lock_type: 'acquisition',
                      strategy: strategyName,
                      parameters: strategyInfo || {},
                      suggestions: cleanSuggestions,
                      notes: ''
                    })
                  });
                  
                  if (!auditResponse.ok) {
                    console.warn('Audit log failed, but experiments were staged');
                  }
                  
                  toast.success(`✓ ${cleanSuggestions.length} suggestion(s) staged`, {
                    description: 'Use "Add Point" button in Experiments panel to add results'
                  });
                  console.log(`✓ Suggestions staged (${strategyName})`);
                } catch (e: any) {
                  toast.error('Failed to stage suggestions: ' + (e?.message || String(e)));
                  console.error('Failed to stage suggestions:', e);
                }
              }}
              className="px-3 py-2 text-xs bg-green-600 text-white rounded hover:bg-green-700 font-medium flex items-center justify-center gap-2"
            >
              📝 Stage Suggestions
            </button>
            <p className="text-xs text-muted-foreground text-center">
              Staging persists suggestions across page reloads. Use "Add Point" to enter experimental results.
            </p>
          </div>
        </div>
      )}
      
      {/* Separator */}
      <div className="my-8 border-t border-border" />
      
      {/* Model Optimum Section */}
      <div className="space-y-2.5">
        <h3 className="text-lg font-semibold">Model Prediction Optimum</h3>
        
        <p className="text-xs text-muted-foreground">
          Find the point where the model predicts the optimal value.
        </p>
        
        <div>
          <label className="block text-xs font-medium mb-1.5">Optimization Goal</label>
          <div className="flex gap-2">
            <button
              onClick={() => setOptimumGoal('maximize')}
              disabled={!modelTrained}
              className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                optimumGoal === 'maximize'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground hover:bg-muted/80'
              }`}
            >
              Maximize
            </button>
            <button
              onClick={() => setOptimumGoal('minimize')}
              disabled={!modelTrained}
              className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                optimumGoal === 'minimize'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground hover:bg-muted/80'
              }`}
            >
              Minimize
            </button>
          </div>
        </div>
        
        <button
          onClick={handleFindOptimum}
          disabled={!modelTrained || findOptimum.isPending}
          className="w-full bg-secondary text-secondary-foreground px-3 py-2 rounded-md text-xs hover:bg-secondary/90 disabled:opacity-50 disabled:cursor-not-allowed font-medium flex items-center justify-center gap-2"
        >
          <Target className="w-3.5 h-3.5" />
          {findOptimum.isPending ? 'Finding Optimum...' : 'Find Model Optimum'}
        </button>
        
        <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-md">
          <p className="text-xs text-amber-700 dark:text-amber-500">
            <strong>Note:</strong> This relies entirely on the model's prediction, not on acquisition 
            functions that balance exploration and exploitation. Use when confident in model accuracy.
          </p>
        </div>
        
        {/* Display optimum result */}
        {findOptimum.isSuccess && findOptimum.data && (
          <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
            <h4 className="font-semibold text-sm mb-3 text-green-700 dark:text-green-500">
              Model Optimum Found
            </h4>
            <div className="space-y-2">
              <div className="p-2.5 bg-background rounded border text-xs">
                <div className="font-medium mb-1">Optimal Point:</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {Object.entries(findOptimum.data.optimum).map(([key, value]) => (
                    <div key={key}>
                      <span className="text-muted-foreground">{key}:</span>{' '}
                      <span className="font-medium">
                        {typeof value === 'number' ? value.toFixed(3) : value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="p-2 bg-background rounded border">
                  <div className="text-xs text-muted-foreground">Predicted Value</div>
                  <div className="font-semibold">{findOptimum.data.predicted_value.toFixed(4)}</div>
                </div>
                {findOptimum.data.predicted_std !== null && (
                  <div className="p-2 bg-background rounded border">
                    <div className="text-xs text-muted-foreground">Uncertainty (±)</div>
                    <div className="font-semibold">{findOptimum.data.predicted_std.toFixed(4)}</div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
