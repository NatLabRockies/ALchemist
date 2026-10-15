/**
 * Variables API endpoints
 */
import { apiClient } from '../client';
import type { Variable, VariableDetail, VariablesListResponse, APIVariable, APIVariableType } from '../types';

/**
 * Convert UI variable type to API type
 */
const toAPIType = (type: string): APIVariableType => {
  if (type === 'continuous') return 'real';
  if (type === 'discrete') return 'integer';
  if (type === 'discrete_numeric') return 'discrete';
  return 'categorical';
};

/**
 * Convert UI Variable to API format
 */
export const toAPIVariable = (variable: Variable): APIVariable => {
  const apiVar: APIVariable = {
    name: variable.name,
    type: toAPIType(variable.type),
  };

  // Only include unit and description if they have values
  if (variable.unit) {
    apiVar.unit = variable.unit;
  }
  if (variable.description) {
    apiVar.description = variable.description;
  }

  if (variable.type === 'continuous' || variable.type === 'discrete') {
    if (variable.bounds) {
      apiVar.min = variable.bounds[0];
      apiVar.max = variable.bounds[1];
    }
  } else if (variable.type === 'categorical') {
    apiVar.categories = variable.categories;
  } else if (variable.type === 'discrete_numeric') {
    apiVar.allowed_values = variable.allowed_values;
  }

  if (variable.role === 'observed') {
    apiVar.role = 'observed';
  } else if (variable.changes && variable.changes !== 'easy') {
    apiVar.changes = variable.changes;
  }

  return apiVar;
};

/**
 * Add a variable to the search space
 */
export const createVariable = async (
  sessionId: string,
  variable: Variable
): Promise<Variable> => {
  const apiVariable = toAPIVariable(variable);
  const response = await apiClient.post<Variable>(
    `/sessions/${sessionId}/variables`,
    apiVariable
  );
  return response.data;
};

/**
 * Update an existing variable in the search space
 */
export const updateVariable = async (
  sessionId: string,
  variableName: string,
  variable: Variable
): Promise<Variable> => {
  const apiVariable = toAPIVariable(variable);
  const response = await apiClient.put<Variable>(
    `/sessions/${sessionId}/variables/${variableName}`,
    apiVariable
  );
  return response.data;
};

/**
 * Get all variables in the search space
 */
export const getVariables = async (sessionId: string): Promise<VariablesListResponse> => {
  const response = await apiClient.get<VariablesListResponse>(
    `/sessions/${sessionId}/variables`
  );
  return response.data;
};

/**
 * Get a specific variable by name
 */
export const getVariable = async (
  sessionId: string,
  variableName: string
): Promise<VariableDetail> => {
  const response = await apiClient.get<VariableDetail>(
    `/sessions/${sessionId}/variables/${variableName}`
  );
  return response.data;
};

/**
 * Delete a variable from the search space
 */
export const deleteVariable = async (sessionId: string, variableName: string): Promise<void> => {
  await apiClient.delete(`/sessions/${sessionId}/variables/${variableName}`);
};

/**
 * Load variables from a JSON file
 */
export const loadVariablesFromFile = async (
  sessionId: string,
  file: File
): Promise<{ message: string; n_variables: number }> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await apiClient.post(`/sessions/${sessionId}/variables/load`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

/**
 * Export the session's variables as a JSON file's contents.
 *
 * A session with input constraints is written as `{variables, constraints}`,
 * the shape Load File reads back with its constraints; one without keeps the
 * bare variable list older tools load. Asking for the bare list always (W35)
 * dropped the constraints and, through a linked file, overwrote them.
 */
export const exportVariables = async (sessionId: string): Promise<Blob> => {
  const response = await apiClient.get<{ variables: unknown[]; constraints: unknown[] }>(
    `/sessions/${sessionId}/variables/export`,
    { params: { include_constraints: true } },
  );
  const { variables, constraints } = response.data;
  const content = constraints.length > 0 ? { variables, constraints } : variables;
  return new Blob([JSON.stringify(content, null, 2)], { type: 'application/json' });
};

/**
 * Load variables from JSON (future enhancement)
 */
export const loadVariablesFromJSON = async (
  sessionId: string,
  jsonData: Variable[]
): Promise<{ message: string; count: number }> => {
  const response = await apiClient.post(
    `/sessions/${sessionId}/variables/load`,
    { variables: jsonData }
  );
  return response.data;
};
