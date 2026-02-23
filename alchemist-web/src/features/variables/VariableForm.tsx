/**
 * Variable Form Component - Mimics desktop SpaceVariableRow
 * Shows inline form for adding/editing variables
 */
import { useState, useEffect } from 'react';
import type { Variable, VariableType, VariableDetail } from '../../api/types';
import { useCreateVariable, useUpdateVariable } from '../../hooks/api/useVariables';
import { toast } from 'sonner';

interface VariableFormProps {
  sessionId: string;
  onClose: () => void;
  editingVariable?: VariableDetail | null;
}

export function VariableForm({ sessionId, onClose, editingVariable }: VariableFormProps) {
  const isEditing = !!editingVariable;
  
  // Convert API type to UI type
  const getUIType = (apiType: string): VariableType => {
    if (apiType === 'real') return 'continuous';
    if (apiType === 'integer') return 'discrete';
    if (apiType === 'discrete') return 'discrete_numeric';
    if (apiType === 'categorical') return 'categorical';
    return 'continuous'; // safe fallback
  };

  const [name, setName] = useState('');
  const [type, setType] = useState<VariableType>('continuous');
  const [minValue, setMinValue] = useState('');
  const [maxValue, setMaxValue] = useState('');
  const [categories, setCategories] = useState<string[]>([]);
  const [categoryInput, setCategoryInput] = useState('');
  const [allowedValuesInput, setAllowedValuesInput] = useState('');
  const [unit, setUnit] = useState('');
  const [description, setDescription] = useState('');

  // Initialize form with editing data
  useEffect(() => {
    if (editingVariable) {
      setName(editingVariable.name);
      setType(getUIType(editingVariable.type));
      if (editingVariable.bounds) {
        setMinValue(editingVariable.bounds[0].toString());
        setMaxValue(editingVariable.bounds[1].toString());
      }
      if (editingVariable.categories) {
        setCategories(editingVariable.categories);
      }
      if (editingVariable.allowed_values) {
        setAllowedValuesInput(editingVariable.allowed_values.join(', '));
      }
      setUnit(editingVariable.unit || '');
      setDescription(editingVariable.description || '');
    }
  }, [editingVariable]);

  const createVariable = useCreateVariable(sessionId);
  const updateVariable = useUpdateVariable(sessionId);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validation
    if (!name.trim()) {
      toast.error('Variable name is required');
      return;
    }

    const variable: Variable = {
      name: name.trim(),
      type,
      unit: unit.trim() || undefined,
      description: description.trim() || undefined,
    };

    if (type === 'continuous' || type === 'discrete') {
      const min = parseFloat(minValue);
      const max = parseFloat(maxValue);

      if (isNaN(min) || isNaN(max)) {
        toast.error('Min and Max values must be numbers');
        return;
      }
      if (min >= max) {
        toast.error('Min must be less than Max');
        return;
      }

      variable.bounds = [min, max];
    } else if (type === 'categorical') {
      if (categories.length === 0) {
        toast.error('At least one category is required');
        return;
      }
      variable.categories = categories;
    } else if (type === 'discrete_numeric') {
      const parsed = allowedValuesInput
        .split(',')
        .map(s => parseFloat(s.trim()))
        .filter(v => !isNaN(v));
      if (parsed.length < 2) {
        toast.error('At least 2 numeric allowed values are required (comma-separated)');
        return;
      }
      if (new Set(parsed).size !== parsed.length) {
        toast.error('Allowed values must not contain duplicates');
        return;
      }
      variable.allowed_values = parsed;
    }

    try {
      if (isEditing && editingVariable) {
        // If name changed, need to delete old and create new
        if (variable.name !== editingVariable.name) {
          // This will be handled by deleting the old variable first
          const { deleteVariable } = await import('../../api/endpoints/variables');
          await deleteVariable(sessionId, editingVariable.name);
          await createVariable.mutateAsync(variable);
        } else {
          // Same name, just update
          await updateVariable.mutateAsync({
            variableName: editingVariable.name,
            variable
          });
        }
      } else {
        // Create new variable
        await createVariable.mutateAsync(variable);
      }
      onClose();
    } catch (error: any) {
      // Error handled by mutation
      console.error('Error saving variable:', error);
    }
  };

  const handleAddCategory = () => {
    const trimmed = categoryInput.trim();
    if (trimmed && !categories.includes(trimmed)) {
      setCategories([...categories, trimmed]);
      setCategoryInput('');
    }
  };

  const handleRemoveCategory = (index: number) => {
    setCategories(categories.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">
          {isEditing ? 'Edit Variable' : 'Add Variable'}
        </h3>
        <button
          onClick={onClose}
          className="text-muted-foreground hover:text-foreground"
        >
          ✕
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Variable Name */}
        <div>
          <label className="block text-sm font-medium mb-1">
            Variable Name *
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g., temperature"
            className="w-full px-3 py-2 border border-input rounded-md bg-background"
            required
          />
        </div>

        {/* Type Selection */}
        <div>
          <label className="block text-sm font-medium mb-1">
            Type *
          </label>
          <select
            value={type}
            onChange={(e) => setType(e.target.value as VariableType)}
            className="w-full px-3 py-2 border border-input rounded-md bg-background"
          >
            <option value="continuous">Real (Continuous)</option>
            <option value="discrete">Integer (Range)</option>
            <option value="discrete_numeric">Discrete (Specific Values)</option>
            <option value="categorical">Categorical</option>
          </select>
        </div>

        {/* Conditional Fields based on Type */}
        {(type === 'continuous' || type === 'discrete') && (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                Min *
              </label>
              <input
                type="number"
                step={type === 'continuous' ? 'any' : '1'}
                value={minValue}
                onChange={(e) => setMinValue(e.target.value)}
                placeholder="Min value"
                className="w-full px-3 py-2 border border-input rounded-md bg-background"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">
                Max *
              </label>
              <input
                type="number"
                step={type === 'continuous' ? 'any' : '1'}
                value={maxValue}
                onChange={(e) => setMaxValue(e.target.value)}
                placeholder="Max value"
                className="w-full px-3 py-2 border border-input rounded-md bg-background"
                required
              />
            </div>
          </div>
        )}

        {type === 'categorical' && (
          <div>
            <label className="block text-sm font-medium mb-1">
              Categories *
            </label>
            <div className="flex gap-2 mb-2">
              <input
                type="text"
                value={categoryInput}
                onChange={(e) => setCategoryInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), handleAddCategory())}
                placeholder="Enter category value"
                className="flex-1 px-3 py-2 border border-input rounded-md bg-background"
              />
              <button
                type="button"
                onClick={handleAddCategory}
                className="px-4 py-2 bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80"
              >
                Add
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {categories.map((cat, index) => (
                <div
                  key={index}
                  className="flex items-center gap-1 px-3 py-1 bg-primary/10 rounded-md"
                >
                  <span className="text-sm">{cat}</span>
                  <button
                    type="button"
                    onClick={() => handleRemoveCategory(index)}
                    className="text-destructive hover:text-destructive/80 ml-1"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
            {categories.length === 0 && (
              <p className="text-sm text-muted-foreground mt-2">
                Add at least one category value
              </p>
            )}
          </div>
        )}

        {type === 'discrete_numeric' && (
          <div>
            <label className="block text-sm font-medium mb-1">
              Allowed Values * <span className="text-muted-foreground font-normal">(comma-separated)</span>
            </label>
            <input
              type="text"
              value={allowedValuesInput}
              onChange={(e) => setAllowedValuesInput(e.target.value)}
              placeholder="e.g., 80, 180, 280"
              className="w-full px-3 py-2 border border-input rounded-md bg-background"
            />
            <p className="text-sm text-muted-foreground mt-1">
              Numeric values this variable can take (at least 2, no duplicates)
            </p>
          </div>
        )}

        {/* Unit (Optional) */}
        <div>
          <label className="block text-sm font-medium mb-1">
            Unit (optional)
          </label>
          <input
            type="text"
            value={unit}
            onChange={(e) => setUnit(e.target.value)}
            placeholder="e.g., °C, bar, mol/L"
            className="w-full px-3 py-2 border border-input rounded-md bg-background"
          />
        </div>

        {/* Description (Optional) */}
        <div>
          <label className="block text-sm font-medium mb-1">
            Description (optional)
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Brief description of this variable"
            rows={2}
            className="w-full px-3 py-2 border border-input rounded-md bg-background resize-none"
          />
        </div>

        {/* Form Actions */}
        <div className="flex gap-2 pt-4">
          <button
            type="submit"
            disabled={createVariable.isPending || updateVariable.isPending}
            className="flex-1 bg-primary text-primary-foreground px-4 py-2 rounded-md hover:bg-primary/90 disabled:opacity-50"
          >
            {(createVariable.isPending || updateVariable.isPending) 
              ? 'Saving...' 
              : isEditing 
                ? 'Update Variable' 
                : 'Add Variable'}
          </button>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 border border-input rounded-md hover:bg-accent"
          >
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
}
