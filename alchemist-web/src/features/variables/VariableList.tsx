/**
 * Variable List Component - Mimics desktop variable rows display
 * Shows table of defined variables with edit/delete actions
 */
import type { VariableDetail } from '../../api/types';
import { useDeleteVariable } from '../../hooks/api/useDeleteVariable';

interface VariableListProps {
  variables: VariableDetail[];
  sessionId: string;
  onEdit: (variable: VariableDetail) => void;
}

export function VariableList({ variables, sessionId, onEdit }: VariableListProps) {
  const deleteVariable = useDeleteVariable(sessionId);

  const handleDelete = async (variableName: string) => {
    if (window.confirm(`Delete variable "${variableName}"?`)) {
      try {
        await deleteVariable.mutateAsync(variableName);
      } catch (error) {
        console.error('Delete failed:', error);
      }
    }
  };

  const formatBounds = (variable: VariableDetail): string => {
    if (variable.type === 'categorical' && variable.categories) {
      return variable.categories.join(', ');
    }
    if (variable.type === 'discrete' && variable.allowed_values && variable.allowed_values.length > 0) {
      return variable.allowed_values.join(', ');
    }
    if (variable.bounds && Array.isArray(variable.bounds) && variable.bounds.length === 2) {
      return `${variable.bounds[0]} to ${variable.bounds[1]}`;
    }
    return '-';
  };

  const getTypeDisplay = (type: string): string => {
    if (type === 'real') return 'Real';
    if (type === 'integer') return 'Integer';
    if (type === 'categorical') return 'Categorical';
    if (type === 'discrete') return 'Discrete';
    return type;
  };

  return (
    <div className="border rounded-lg overflow-hidden">
      {/* Header - mimics desktop UI header */}
      <div className="bg-muted/50 border-b px-4 py-2 grid grid-cols-12 gap-3 font-medium text-xs">
        <div className="col-span-2">Name</div>
        <div className="col-span-2">Type</div>
        <div className="col-span-3">Range/Values</div>
        <div className="col-span-1">Unit</div>
        <div className="col-span-2">Description</div>
        <div className="col-span-2 text-right">Actions</div>
      </div>

      {/* Variable Rows */}
      <div className="divide-y">
        {variables.map((variable, index) => (
          <div
            key={`${variable.name}-${index}`}
            className="px-4 py-2.5 grid grid-cols-12 gap-3 items-center hover:bg-accent/50 transition-colors"
          >
            {/* Variable Name */}
            <div className="col-span-2 font-medium text-xs">
              {variable.name}
            </div>

            {/* Type */}
            <div className="col-span-2 text-xs">
              {getTypeDisplay(variable.type)}
            </div>

            {/* Parameters/Range */}
            <div className="col-span-3 text-xs text-muted-foreground">
              {formatBounds(variable)}
            </div>

            {/* Unit */}
            <div className="col-span-1 text-xs text-muted-foreground">
              {variable.unit || '-'}
            </div>

            {/* Description */}
            <div className="col-span-2 text-xs text-muted-foreground truncate" title={variable.description || ''}>
              {variable.description || '-'}
            </div>

            {/* Actions */}
            <div className="col-span-2 flex justify-end gap-2">
              <button
                onClick={() => onEdit(variable)}
                className="text-xs text-primary hover:text-primary/80"
                title="Edit variable"
              >
                Edit
              </button>
              <button
                onClick={() => handleDelete(variable.name)}
                className="text-xs text-destructive hover:text-destructive/80"
                title="Delete variable"
                disabled={deleteVariable.isPending}
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Footer Summary - mimics desktop summary */}
      <div className="bg-muted/30 border-t px-4 py-2 text-sm text-muted-foreground">
        {variables.length} variable{variables.length !== 1 ? 's' : ''} defined
      </div>
    </div>
  );
}
