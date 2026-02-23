/**
 * API client functions for LLM / AI-assisted design endpoints.
 */

import { apiClient } from '../client';
import type { LLMSavedConfig } from '../types';

/** Read saved provider config from ~/.alchemist/config.json (via backend). */
export async function getLLMConfig(): Promise<LLMSavedConfig> {
  const response = await apiClient.get<LLMSavedConfig>('/llm/config');
  return response.data;
}

/** Persist provider config to ~/.alchemist/config.json (via backend). */
export async function saveLLMConfig(config: LLMSavedConfig): Promise<void> {
  await apiClient.post('/llm/config', config);
}

/** List locally available Ollama models. Returns [] if Ollama is unreachable. */
export async function listOllamaModels(
  baseUrl = 'http://localhost:11434'
): Promise<string[]> {
  const response = await apiClient.get<{ models: string[]; error?: string }>(
    `/llm/ollama/models?base_url=${encodeURIComponent(baseUrl)}`
  );
  return response.data.models;
}
