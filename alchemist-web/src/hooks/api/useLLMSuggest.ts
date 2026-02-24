/**
 * Hook for the LLM suggest-effects SSE stream.
 *
 * Uses fetch + ReadableStream (EventSource only supports GET; we need POST).
 * Exposes status, progress message, final result, and a cancel function.
 */

import { useCallback, useRef, useState } from 'react';
import type { SuggestEffectsRequest, SuggestedEffectsResult, SuggestEffectsEvent } from '../../api/types';

export type SuggestStatus = 'idle' | 'loading' | 'success' | 'error';

export interface UseLLMSuggestReturn {
  suggest: (request: SuggestEffectsRequest) => Promise<void>;
  cancel: () => void;
  status: SuggestStatus;
  progressMessage: string;
  trajectoryUrl: string | null;
  /** True when the current result was served from the per-session Edison cache. */
  fromCache: boolean;
  result: SuggestedEffectsResult | null;
  error: string | null;
}

export function useLLMSuggest(sessionId: string): UseLLMSuggestReturn {
  const [status, setStatus] = useState<SuggestStatus>('idle');
  const [progressMessage, setProgressMessage] = useState('');
  const [trajectoryUrl, setTrajectoryUrl] = useState<string | null>(null);
  const [fromCache, setFromCache] = useState(false);
  const [result, setResult] = useState<SuggestedEffectsResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const suggest = useCallback(
    async (request: SuggestEffectsRequest) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setStatus('loading');
      setProgressMessage('Starting…');
      setTrajectoryUrl(null);
      setFromCache(false);
      setError(null);
      setResult(null);

      try {
        const response = await fetch(
          `/api/v1/llm/suggest-effects/${sessionId}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
            signal: controller.signal,
          }
        );

        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || `HTTP ${response.status}`);
        }

        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          // SSE events are separated by double newlines
          const chunks = buffer.split('\n\n');
          buffer = chunks.pop() ?? '';

          for (const chunk of chunks) {
            const line = chunk.trim();
            if (!line.startsWith('data: ')) continue;
            let event: SuggestEffectsEvent;
            try {
              event = JSON.parse(line.slice(6));
            } catch {
              continue;
            }

            if (event.status === 'complete') {
              setResult(event.result);
              setStatus('success');
              setProgressMessage('');
            } else if (event.status === 'error') {
              setError(event.message);
              setStatus('error');
              setProgressMessage('');
            } else {
              // progress events: searching_literature, searching_literature_progress,
              // literature_complete, literature_warning, literature_error, structuring
              if ('message' in event) setProgressMessage(event.message);

              // Extract trajectory_url from any event that carries it
              // (covers: searching_literature*, literature_complete, literature_warning)
              if ('trajectory_url' in event && event.trajectory_url) {
                setTrajectoryUrl(event.trajectory_url);
              }

              // Mark whether the result came from the per-session cache
              if ('cached' in event && event.cached) {
                setFromCache(true);
              }
            }
          }
        }
      } catch (err: any) {
        if (err?.name === 'AbortError') {
          setStatus('idle');
          setProgressMessage('');
        } else {
          setError(err?.message ?? String(err));
          setStatus('error');
          setProgressMessage('');
        }
      }
    },
    [sessionId]
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setStatus('idle');
    setProgressMessage('');
    setTrajectoryUrl(null);
    setFromCache(false);
  }, []);

  return { suggest, cancel, status, progressMessage, trajectoryUrl, fromCache, result, error };
}
