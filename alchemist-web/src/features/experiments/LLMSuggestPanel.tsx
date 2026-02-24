/**
 * LLMSuggestPanel — "Suggest effects with AI" expandable section.
 *
 * Appears inside the Custom Effects mode of OptimalDesignPanel.
 * Supports OpenAI (Responses API) and Ollama, with optional
 * Edison Scientific literature search as a first stage.
 */
import { useState, useEffect } from 'react';
import { ChevronDown, ChevronRight, Loader2, Sparkles, CheckCircle, XCircle } from 'lucide-react';
import { useLLMSuggest } from '../../hooks/api/useLLMSuggest';
import { getLLMConfig, saveLLMConfig, listOllamaModels } from '../../api/endpoints/llm';
import type { LLMProvider, EdisonJobType, SuggestEffectsRequest, LLMSavedConfig } from '../../api/types';

interface AvailableEffects {
  mainEffects: string[];
  interactions: string[];
  quadratics: string[];
}

interface LLMSuggestPanelProps {
  sessionId: string;
  availableEffects: AvailableEffects;
  onEffectsSuggested: (effects: string[]) => void;
}

export function LLMSuggestPanel({ sessionId, availableEffects, onEffectsSuggested }: LLMSuggestPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  // Provider config state
  const [provider, setProvider] = useState<LLMProvider>('openai');
  const [model, setModel] = useState('gpt-4o');
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState('http://localhost:11434/v1');
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [ollamaLoading, setOllamaLoading] = useState(false);

  // Edison config state
  const [useEdison, setUseEdison] = useState(false);
  const [edisonApiKey, setEdisonApiKey] = useState('');
  const [edisonJobType, setEdisonJobType] = useState<EdisonJobType>('literature');

  // System context
  const [systemContext, setSystemContext] = useState('');

  // Results expand state
  const [showReasoning, setShowReasoning] = useState(false);
  const [showLiterature, setShowLiterature] = useState(false);
  const [showSources, setShowSources] = useState(false);

  const { suggest, cancel, status, progressMessage, result, error } = useLLMSuggest(sessionId);

  // Load saved config once on mount
  useEffect(() => {
    getLLMConfig()
      .then((cfg: LLMSavedConfig) => {
        if (cfg.openai?.api_key) setApiKey(cfg.openai.api_key);
        if (cfg.ollama?.base_url) setBaseUrl(cfg.ollama.base_url);
        if (cfg.edison?.api_key) setEdisonApiKey(cfg.edison.api_key);
      })
      .catch(() => {}); // No saved config is fine
  }, []);

  // Auto-load Ollama models when switching to Ollama provider
  useEffect(() => {
    if (provider !== 'ollama') {
      if (provider === 'openai') setModel('gpt-4o');
      return;
    }
    fetchOllamaModels(baseUrl);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider]);

  const fetchOllamaModels = async (url: string) => {
    setOllamaLoading(true);
    try {
      const models = await listOllamaModels(url);
      setOllamaModels(models);
      if (models.length > 0) setModel(prev => models.includes(prev) ? prev : models[0]);
    } catch {
      setOllamaModels([]);
    } finally {
      setOllamaLoading(false);
    }
  };

  const handleSuggest = async () => {
    const request: SuggestEffectsRequest = {
      structuring_provider: {
        provider,
        model,
        ...(provider === 'openai' && apiKey ? { api_key: apiKey } : {}),
        ...(provider === 'ollama' ? { base_url: baseUrl } : {}),
      },
      system_context: systemContext,
      ...(useEdison && edisonApiKey ? {
        edison_config: { api_key: edisonApiKey, job_type: edisonJobType },
      } : {}),
    };
    await suggest(request);
    // Expand reasoning section automatically on success
    setShowReasoning(true);
  };

  const handleSaveConfig = async () => {
    const cfg: LLMSavedConfig = {
      ...(apiKey ? { openai: { api_key: apiKey } } : {}),
      ollama: { base_url: baseUrl },
      ...(edisonApiKey ? { edison: { api_key: edisonApiKey } } : {}),
    };
    try {
      await saveLLMConfig(cfg);
    } catch {
      // Non-critical
    }
  };

  const handleApply = () => {
    if (!result) return;
    const allAvailable = new Set([
      ...availableEffects.mainEffects,
      ...availableEffects.interactions,
      ...availableEffects.quadratics,
    ]);
    const matched = result.effects.filter(e => allAvailable.has(e));
    onEffectsSuggested(matched);
  };

  const allAvailableSet = new Set([
    ...availableEffects.mainEffects,
    ...availableEffects.interactions,
    ...availableEffects.quadratics,
  ]);

  return (
    <div className="border rounded bg-purple-50/30 dark:bg-purple-950/10 border-purple-200/50 dark:border-purple-800/30">
      {/* Header toggle */}
      <button
        onClick={() => setIsOpen(v => !v)}
        className="w-full flex items-center justify-between px-2.5 py-1.5 text-xs font-medium text-purple-700 dark:text-purple-400 hover:bg-purple-50/50 dark:hover:bg-purple-950/20 rounded"
      >
        <span className="flex items-center gap-1.5">
          <Sparkles className="h-3.5 w-3.5" />
          Suggest effects with AI
        </span>
        {isOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
      </button>

      {isOpen && (
        <div className="px-2.5 pb-2.5 space-y-2.5 border-t border-purple-200/50 dark:border-purple-800/30 pt-2">

          {/* ── Provider Config ─────────────────────────────────── */}
          <div className="space-y-1.5">
            <label className="text-xs text-muted-foreground font-medium">Provider</label>
            <div className="flex gap-3">
              {(['openai', 'ollama'] as LLMProvider[]).map(p => (
                <label key={p} className="inline-flex items-center gap-1.5 text-xs cursor-pointer">
                  <input
                    type="radio"
                    checked={provider === p}
                    onChange={() => setProvider(p)}
                    className="h-3 w-3"
                  />
                  {p === 'openai' ? 'OpenAI' : 'Ollama (local)'}
                </label>
              ))}
            </div>

            {provider === 'openai' ? (
              <>
                <div>
                  <label className="text-xs text-muted-foreground block mb-0.5">Model</label>
                  <input
                    type="text"
                    value={model}
                    onChange={e => setModel(e.target.value)}
                    placeholder="gpt-4o"
                    className="w-full px-2 py-1 text-xs border rounded bg-background"
                  />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground block mb-0.5">API Key</label>
                  <input
                    type="password"
                    value={apiKey}
                    onChange={e => setApiKey(e.target.value)}
                    placeholder="sk-..."
                    className="w-full px-2 py-1 text-xs border rounded bg-background"
                  />
                </div>
              </>
            ) : (
              <>
                <div className="flex gap-1.5 items-end">
                  <div className="flex-1">
                    <label className="text-xs text-muted-foreground block mb-0.5">Base URL</label>
                    <input
                      type="text"
                      value={baseUrl}
                      onChange={e => setBaseUrl(e.target.value)}
                      placeholder="http://localhost:11434/v1"
                      className="w-full px-2 py-1 text-xs border rounded bg-background"
                    />
                  </div>
                  <button
                    onClick={() => fetchOllamaModels(baseUrl)}
                    disabled={ollamaLoading}
                    className="px-2 py-1 text-xs border rounded hover:bg-accent disabled:opacity-50 whitespace-nowrap"
                  >
                    {ollamaLoading ? '…' : 'Refresh'}
                  </button>
                </div>
                <div>
                  <label className="text-xs text-muted-foreground block mb-0.5">Model</label>
                  {ollamaModels.length > 0 ? (
                    <select
                      value={model}
                      onChange={e => setModel(e.target.value)}
                      className="w-full px-2 py-1.5 text-xs border rounded bg-background"
                    >
                      {ollamaModels.map(m => <option key={m} value={m}>{m}</option>)}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={model}
                      onChange={e => setModel(e.target.value)}
                      placeholder="llama3.2 (Ollama not reachable)"
                      className="w-full px-2 py-1 text-xs border rounded bg-background"
                    />
                  )}
                </div>
              </>
            )}

            <button
              onClick={handleSaveConfig}
              className="text-xs text-muted-foreground hover:text-foreground underline underline-offset-2"
            >
              Save provider settings
            </button>
          </div>

          {/* ── Edison Scientific (optional) ─────────────────────── */}
          <div className="border-t pt-2 space-y-1.5">
            <label className="inline-flex items-center gap-2 text-xs cursor-pointer select-none">
              <input
                type="checkbox"
                checked={useEdison}
                onChange={e => setUseEdison(e.target.checked)}
                className="h-3 w-3"
              />
              <span className="font-medium">Edison Scientific</span>
              <span className="text-muted-foreground">(literature search)</span>
            </label>

            {useEdison && (
              <div className="pl-4 space-y-1.5 border-l border-purple-200/50 dark:border-purple-700/30">
                <div>
                  <label className="text-xs text-muted-foreground block mb-0.5">Edison API Key</label>
                  <input
                    type="password"
                    value={edisonApiKey}
                    onChange={e => setEdisonApiKey(e.target.value)}
                    placeholder="Edison API key"
                    className="w-full px-2 py-1 text-xs border rounded bg-background"
                  />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground block mb-0.5">Search mode</label>
                  <select
                    value={edisonJobType}
                    onChange={e => setEdisonJobType(e.target.value as EdisonJobType)}
                    className="w-full px-2 py-1.5 text-xs border rounded bg-background"
                  >
                    <option value="literature">Literature (standard)</option>
                    <option value="literature_high">Literature (high accuracy)</option>
                    <option value="precedent">Precedent</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* ── System context ──────────────────────────────────── */}
          <div className="border-t pt-2">
            <label className="text-xs text-muted-foreground block mb-0.5">
              Experiment description{' '}
              <span className="text-muted-foreground/60">(used to generate search query)</span>
            </label>
            <textarea
              value={systemContext}
              onChange={e => setSystemContext(e.target.value)}
              placeholder={
                'Describe what you are studying.\n' +
                'e.g. I am optimizing the yield of biochar from biomass pyrolysis. ' +
                'Variables include temperature (300–700 °C), residence time, and feedstock type.'
              }
              rows={3}
              className="w-full px-2 py-1 text-xs border rounded bg-background resize-none leading-relaxed"
            />
          </div>

          {/* ── Suggest / Cancel buttons ────────────────────────── */}
          <div className="flex gap-2">
            <button
              onClick={handleSuggest}
              disabled={status === 'loading' || !systemContext.trim()}
              className="flex-1 flex items-center justify-center gap-1.5 bg-purple-600 text-white text-xs px-3 py-1.5 rounded hover:bg-purple-700 disabled:opacity-50 transition-colors"
            >
              {status === 'loading' ? (
                <><Loader2 className="h-3 w-3 animate-spin" /> Analyzing…</>
              ) : (
                <><Sparkles className="h-3 w-3" /> Suggest effects</>
              )}
            </button>
            {status === 'loading' && (
              <button
                onClick={cancel}
                className="text-xs px-2.5 py-1.5 border rounded hover:bg-accent text-muted-foreground"
              >
                Cancel
              </button>
            )}
          </div>

          {/* ── Progress message ─────────────────────────────────── */}
          {status === 'loading' && progressMessage && (
            <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Loader2 className="h-3 w-3 animate-spin shrink-0" />
              {progressMessage}
            </p>
          )}

          {/* ── Error ────────────────────────────────────────────── */}
          {status === 'error' && error && (
            <div className="flex items-start gap-1.5 text-xs text-red-600 dark:text-red-400 border border-red-200 dark:border-red-800 rounded p-2 bg-red-50/50 dark:bg-red-950/20">
              <XCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {/* ── Results ──────────────────────────────────────────── */}
          {status === 'success' && result && (
            <div className="space-y-2 border-t pt-2">
              {/* Success banner */}
              <div className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400 font-medium">
                <CheckCircle className="h-3.5 w-3.5 shrink-0" />
                {result.effects.length} effect{result.effects.length !== 1 ? 's' : ''} suggested
              </div>

              {/* Effect chips */}
              <div className="flex flex-wrap gap-1">
                {result.effects.map(eff => {
                  const inSpace = allAvailableSet.has(eff);
                  return (
                    <span
                      key={eff}
                      title={inSpace ? 'Matches a variable in your design space' : 'Not in current variable set'}
                      className={`px-1.5 py-0.5 rounded text-xs border ${
                        inSpace
                          ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 border-purple-200 dark:border-purple-700'
                          : 'bg-muted/40 text-muted-foreground border-dashed'
                      }`}
                    >
                      {eff}{!inSpace && ' ⚠'}
                    </span>
                  );
                })}
              </div>

              {/* Apply button */}
              <button
                onClick={handleApply}
                className="w-full flex items-center justify-center gap-1.5 text-xs bg-purple-600 text-white px-3 py-1.5 rounded hover:bg-purple-700 transition-colors"
              >
                Apply to checkboxes
              </button>

              {/* Reasoning collapsible */}
              {result.reasoning.length > 0 && (
                <div>
                  <button
                    onClick={() => setShowReasoning(v => !v)}
                    className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                  >
                    {showReasoning ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                    Reasoning
                  </button>
                  {showReasoning && (
                    <div className="mt-1 space-y-1 pl-3 border-l border-purple-200/50 dark:border-purple-700/30">
                      {result.reasoning.map((r, i) => (
                        <div key={i} className="text-xs">
                          <span className="font-medium">{r.effect}:</span>{' '}
                          <span className="text-muted-foreground">{r.reason}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Literature context collapsible */}
              {result.literature_context && (
                <div>
                  <button
                    onClick={() => setShowLiterature(v => !v)}
                    className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                  >
                    {showLiterature ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                    Literature context (Edison)
                  </button>
                  {showLiterature && (
                    <div className="mt-1 pl-3 border-l border-purple-200/50 dark:border-purple-700/30 text-xs text-muted-foreground whitespace-pre-wrap leading-relaxed max-h-36 overflow-y-auto">
                      {result.literature_context}
                    </div>
                  )}
                </div>
              )}

              {/* Sources collapsible */}
              {result.sources.length > 0 && (
                <div>
                  <button
                    onClick={() => setShowSources(v => !v)}
                    className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                  >
                    {showSources ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                    Sources ({result.sources.length})
                  </button>
                  {showSources && (
                    <ul className="mt-1 pl-3 border-l border-purple-200/50 dark:border-purple-700/30 space-y-0.5">
                      {result.sources.map((src, i) => (
                        <li key={i} className="text-xs text-muted-foreground break-words">{src}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              {/* Disclaimer */}
              {result.disclaimer && (
                <p className="text-xs text-muted-foreground/60 italic">{result.disclaimer}</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
