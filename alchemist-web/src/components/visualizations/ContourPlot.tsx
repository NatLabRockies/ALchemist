/**
 * Contour Plot - 2D surface plot of model predictions
 * Mirrors desktop UI contour plot from visualizations.py with side controls
 * Uses Plotly.js for professional contour visualization
 */
import { useState, useMemo, useEffect, useCallback } from 'react';
import { Loader2 } from 'lucide-react';
import Plot from 'react-plotly.js';
import type { Data, Layout, Config } from 'plotly.js';
import { useContourData } from '../../hooks/api/useVisualizations';
import { useVariables } from '../../hooks/api/useVariables';
import type { VariableDetail } from '../../api/types';
import { useTheme } from '../../hooks/useTheme';

interface ContourPlotProps {
  sessionId: string;
}

interface FixedValue {
  value: number | string;
  min?: number;
  max?: number;
  categories?: string[];
  allowed_values?: number[];
  type: 'Real' | 'Integer' | 'Categorical' | 'Discrete';
}

export function ContourPlot({ sessionId }: ContourPlotProps) {
  const { data: variables } = useVariables(sessionId);
  const { theme } = useTheme();

  // Get continuous (Real) variables only
  const realVariables = useMemo(() => {
    if (!variables) return [];
    return variables.variables.filter((v: VariableDetail) => v.type === 'real');
  }, [variables]);

  const [xAxis, setXAxis] = useState<string>('');
  const [yAxis, setYAxis] = useState<string>('');
  const [fixedValues, setFixedValues] = useState<Record<string, FixedValue>>({});
  const [committedFixedValues, setCommittedFixedValues] = useState<Record<string, FixedValue>>({});
  const [gridResolution, setGridResolution] = useState(50);
  const [committedGridResolution, setCommittedGridResolution] = useState(50);
  const [showExperiments, setShowExperiments] = useState(false);
  const [showNextPoint, setShowNextPoint] = useState(false);
  const [colormap, setColormap] = useState<string>('Viridis');

  // Initialize axes when variables load
  useEffect(() => {
    if (realVariables.length >= 2 && !xAxis) {
      setXAxis(realVariables[0].name);
      setYAxis(realVariables[1].name);
    }
  }, [realVariables, xAxis]);

  // Update fixed values when axes change
  useEffect(() => {
    if (!variables) return;

    const newFixed: Record<string, FixedValue> = {};
    variables.variables.forEach((variable: VariableDetail) => {
      if (variable.name === xAxis || variable.name === yAxis) return;

      if (variable.type === 'real' && variable.bounds) {
        const min = variable.bounds[0];
        const max = variable.bounds[1];
        newFixed[variable.name] = {
          value: (min + max) / 2,
          min,
          max,
          type: 'Real',
        };
      } else if (variable.type === 'integer' && variable.bounds) {
        const min = variable.bounds[0];
        const max = variable.bounds[1];
        newFixed[variable.name] = {
          value: Math.floor((min + max) / 2),
          min,
          max,
          type: 'Integer',
        };
      } else if (variable.type === 'categorical' && variable.categories) {
        newFixed[variable.name] = {
          value: variable.categories[0],
          categories: variable.categories,
          type: 'Categorical',
        };
      } else if (variable.type === 'discrete' && variable.allowed_values && variable.allowed_values.length > 0) {
        const av = variable.allowed_values;
        newFixed[variable.name] = {
          value: av[Math.floor(av.length / 2)],
          allowed_values: av,
          type: 'Discrete',
        };
      }
    });

    setFixedValues(newFixed);
    setCommittedFixedValues(newFixed); // Also commit on initialization
  }, [xAxis, yAxis, variables]);

  // Fetch contour data (use committedFixedValues - only updates when slider is released)
  const contourRequest = useMemo(() => {
    if (!xAxis || !yAxis) return null;

    // Build fixed values object (exclude x/y axes). It's valid for this
    // object to be empty when the search space only contains the two
    // selected variables — that should still produce a valid contour.
    const fixed: Record<string, number | string> = {};
    Object.entries(committedFixedValues).forEach(([key, val]) => {
      if (key !== xAxis && key !== yAxis) {
        fixed[key] = val.value;
      }
    });

    const request = {
      x_var: xAxis,
      y_var: yAxis,
      fixed_values: fixed,
      grid_resolution: committedGridResolution,
      include_experiments: showExperiments,
      include_suggestions: showNextPoint,
    };

    // Prepared request (debug log intentionally removed in cleanup)

    return request;
  }, [xAxis, yAxis, committedFixedValues, committedGridResolution, showExperiments, showNextPoint]);

  const {
    data: contourApiData,
    isLoading,
    error,
  } = useContourData(sessionId, contourRequest!, !!contourRequest);

  // Handler for updating fixed values (updates local state while dragging)
  const handleFixedValueChange = useCallback((varName: string, value: number | string) => {
    setFixedValues((prev) => ({
      ...prev,
      [varName]: { ...prev[varName], value },
    }));
  }, []);

  // Handler for committing fixed values (triggers API request when slider is released)
  const handleFixedValueCommit = useCallback((varName: string, value: number | string) => {
    setCommittedFixedValues((prev) => ({
      ...prev,
      [varName]: { ...prev[varName], value },
    }));
  }, []);

  if (realVariables.length < 2) {
    return (
      <div className="flex items-center justify-center h-96">
        <p className="text-muted-foreground">
          Need at least two Real (continuous) variables for contour plotting
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full gap-2">
      {/* Main plot area */}
      <div className="flex-1 flex flex-col min-h-[550px]">
        {isLoading && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">Generating contour plot...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <p className="text-destructive font-medium">Error loading contour plot</p>
              <p className="text-sm text-muted-foreground mt-2">{error.message}</p>
            </div>
          </div>
        )}

        {!isLoading && !error && contourApiData && (
          <div className="flex-1 min-h-0">
            <ContourPlotly
              data={contourApiData}
              xAxis={xAxis}
              yAxis={yAxis}
              showExperiments={showExperiments}
              showNextPoint={showNextPoint}
              colormap={colormap}
              theme={theme}
            />
          </div>
        )}
      </div>

      {/* Controls below plot */}
      <div className="bg-muted/30 rounded-lg p-3 flex-shrink-0">
        <div className="grid grid-cols-5 gap-3">
          {/* Column 1: Axis Selection */}
          <div className="space-y-2">
            <h3 className="font-semibold text-xs uppercase tracking-wide text-muted-foreground border-b pb-1.5">Axes</h3>
            <div>
              <label className="text-xs font-medium block mb-1">X-Axis:</label>
              <select
                value={xAxis}
                onChange={(e) => setXAxis(e.target.value)}
                className="w-full px-2.5 py-1.5 text-xs bg-background border border-input rounded-md"
              >
                {realVariables.map((v: VariableDetail) => (
                  <option key={v.name} value={v.name}>
                    {v.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs font-medium block mb-1">Y-Axis:</label>
              <select
                value={yAxis}
                onChange={(e) => setYAxis(e.target.value)}
                className="w-full px-2.5 py-1.5 text-xs bg-background border border-input rounded-md"
              >
                {realVariables.map((v: VariableDetail) => (
                  <option key={v.name} value={v.name}>
                    {v.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Column 2: Fixed Values */}
          <div className="space-y-2">
            <h3 className="font-semibold text-xs uppercase tracking-wide text-muted-foreground border-b pb-1.5">Fixed Values</h3>
            <div className="space-y-2 max-h-[120px] overflow-y-auto pr-2">
              {Object.entries(fixedValues).map(([varName, varInfo]) => (
                <div key={varName}>
                  <label className="text-xs font-medium block mb-1">{varName}:</label>
                  {varInfo.type === 'Real' && (
                    <div>
                      <input
                        type="range"
                        min={varInfo.min}
                        max={varInfo.max}
                        step={(varInfo.max! - varInfo.min!) / 100}
                        value={varInfo.value as number}
                        onChange={(e) => handleFixedValueChange(varName, parseFloat(e.target.value))}
                        onMouseUp={(e) => handleFixedValueCommit(varName, parseFloat((e.target as HTMLInputElement).value))}
                        onTouchEnd={(e) => handleFixedValueCommit(varName, parseFloat((e.target as HTMLInputElement).value))}
                        className="w-full"
                      />
                      <div className="text-xs text-muted-foreground mt-1">
                        {(varInfo.value as number).toFixed(3)}
                      </div>
                    </div>
                  )}
                  {varInfo.type === 'Integer' && (
                    <input
                      type="number"
                      min={varInfo.min}
                      max={varInfo.max}
                      step={1}
                      value={varInfo.value as number}
                      onChange={(e) => {
                        const val = parseInt(e.target.value);
                        handleFixedValueChange(varName, val);
                        handleFixedValueCommit(varName, val);
                      }}
                      className="w-full px-2 py-1 text-xs bg-background border border-input rounded"
                    />
                  )}
                  {varInfo.type === 'Categorical' && (
                    <select
                      value={varInfo.value as string}
                      onChange={(e) => {
                        handleFixedValueChange(varName, e.target.value);
                        handleFixedValueCommit(varName, e.target.value);
                      }}
                      className="w-full px-2 py-1 text-xs bg-background border border-input rounded"
                    >
                      {varInfo.categories?.map((cat) => (
                        <option key={cat} value={cat}>
                          {cat}
                        </option>
                      ))}
                    </select>
                  )}
                  {varInfo.type === 'Discrete' && (
                    <select
                      value={varInfo.value as number}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        handleFixedValueChange(varName, val);
                        handleFixedValueCommit(varName, val);
                      }}
                      className="w-full px-2 py-1 text-xs bg-background border border-input rounded"
                    >
                      {varInfo.allowed_values?.map((v) => (
                        <option key={v} value={v}>
                          {v}
                        </option>
                      ))}
                    </select>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Column 3: Display Options */}
          <div className="space-y-2">
            <h3 className="font-semibold text-xs uppercase tracking-wide text-muted-foreground border-b pb-1.5">Display</h3>
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={showExperiments}
                  onChange={(e) => setShowExperiments(e.target.checked)}
                  className="w-3.5 h-3.5 rounded border-gray-300"
                />
                <span>Show Experimental Points</span>
              </label>
              <label className="flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={showNextPoint}
                  onChange={(e) => setShowNextPoint(e.target.checked)}
                  className="w-3.5 h-3.5 rounded border-gray-300"
                />
                <span>Show Next Point</span>
              </label>
            </div>
          </div>

          {/* Column 4: Colormap */}
          <div className="space-y-2">
            <h3 className="font-semibold text-xs uppercase tracking-wide text-muted-foreground border-b pb-1.5">Colormap</h3>
            <div>
              <select
                value={colormap}
                onChange={(e) => setColormap(e.target.value)}
                className="w-full px-2 py-1 text-xs bg-background border border-input rounded"
              >
                <option value="Viridis">Viridis</option>
                <option value="Plasma">Plasma</option>
                <option value="Inferno">Inferno</option>
                <option value="Magma">Magma</option>
                <option value="Cividis">Cividis</option>
                <option value="Jet">Jet</option>
                <option value="Hot">Hot</option>
                <option value="Cool">Cool</option>
                <option value="RdBu">Red-Blue</option>
                <option value="YlOrRd">Yellow-Orange-Red</option>
              </select>
            </div>
          </div>

          {/* Column 5: Grid Resolution */}
          <div className="space-y-2">
            <h3 className="font-semibold text-xs uppercase tracking-wide text-muted-foreground border-b pb-1.5">Resolution</h3>
            <div>
              <label className="text-xs font-medium block mb-1">Grid Resolution:</label>
              <input
                type="range"
                min={30}
                max={150}
                step={10}
                value={gridResolution}
                onChange={(e) => setGridResolution(parseInt(e.target.value))}
                onMouseUp={(e) => setCommittedGridResolution(parseInt((e.target as HTMLInputElement).value))}
                onTouchEnd={(e) => setCommittedGridResolution(parseInt((e.target as HTMLInputElement).value))}
                className="w-full"
              />
              <div className="text-xs text-muted-foreground mt-1">{gridResolution} × {gridResolution}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Plotly-based contour plot renderer
 * Provides professional, interactive contour visualization matching desktop UI
 */
interface ContourPlotlyProps {
  data: {
    x_var: string;
    y_var: string;
    x_grid: number[][];
    y_grid: number[][];
    predictions: number[][];
    uncertainties: number[][];
    experiments?: {
      x: number[];
      y: number[];
      output: number[];
    } | null;
    suggestions?: {
      x: number[];
      y: number[];
    } | null;
    x_bounds: number[];
    y_bounds: number[];
    colorbar_bounds: number[];
  };
  xAxis: string;
  yAxis: string;
  showExperiments: boolean;
  showNextPoint: boolean;
  colormap: string;
  theme: 'light' | 'dark';
}

function ContourPlotly({ 
  data, 
  xAxis, 
  yAxis, 
  showExperiments, 
  showNextPoint,
  colormap,
  theme
}: ContourPlotlyProps) {
  // Extract 1D arrays from 2D meshgrids for Plotly
  // x_grid is constant along columns, y_grid is constant along rows
  const xValues = data.x_grid[0]; // First row contains all unique x values
  const yValues = data.y_grid.map(row => row[0]); // First column contains all unique y values

  // Build traces array
  const traces: Data[] = [];

  // Main contour trace
  const contourTrace: Data = {
    type: 'contour',
    x: xValues,
    y: yValues,
    z: data.predictions,
    colorscale: colormap,
    colorbar: {
      title: {
        text: 'Prediction',
        side: 'right'
      },
      thickness: 20,
      len: 0.7,
    },
    contours: {
      coloring: 'heatmap',
    },
    hovertemplate: 
      `${xAxis}: %{x:.3f}<br>` +
      `${yAxis}: %{y:.3f}<br>` +
      'Prediction: %{z:.3f}<br>' +
      '<extra></extra>',
  };
  traces.push(contourTrace);

  // Add experimental points if requested and available
  if (showExperiments && data.experiments && data.experiments.x.length > 0) {
    const experimentTrace: Data = {
      type: 'scatter',
      mode: 'markers',
      x: data.experiments.x,
      y: data.experiments.y,
      marker: {
        color: 'white',
        size: 8,
        line: {
          color: 'black',
          width: 2
        },
        symbol: 'circle'
      },
      name: 'Experiments',
      hovertemplate: 
        `${xAxis}: %{x:.3f}<br>` +
        `${yAxis}: %{y:.3f}<br>` +
        'Output: %{text}<br>' +
        '<extra></extra>',
      text: data.experiments.output.map(v => v.toFixed(3)),
    };
    traces.push(experimentTrace);
  }

  // Add next suggested point if requested and available
  if (showNextPoint && data.suggestions && data.suggestions.x.length > 0) {
    const suggestionTrace: Data = {
      type: 'scatter',
      mode: 'markers',
      x: data.suggestions.x,
      y: data.suggestions.y,
      marker: {
        color: 'red',
        size: 12,
        line: {
          color: 'darkred',
          width: 2
        },
        symbol: 'diamond'
      },
      name: 'Next Point',
      hovertemplate: 
        `${xAxis}: %{x:.3f}<br>` +
        `${yAxis}: %{y:.3f}<br>` +
        '<extra></extra>',
    };
    traces.push(suggestionTrace);
  }

  // Layout configuration matching desktop UI - reactive to theme changes
  const layout: Partial<Layout> = useMemo(() => ({
    title: {
      text: 'Contour Plot of Model Predictions',
      font: {
        size: 16,
        family: 'Arial, sans-serif'
      }
    },
    xaxis: {
      title: {
        text: xAxis,
        font: { size: 14 }
      },
      range: data.x_bounds,
      showgrid: true,
      zeroline: false,
    },
    yaxis: {
      title: {
        text: yAxis,
        font: { size: 14 }
      },
      range: data.y_bounds,
      showgrid: true,
      zeroline: false,
    },
    autosize: true,
    margin: {
      l: 70,
      r: 110,
      t: 60,
      b: 50
    },
    hovermode: 'closest',
    showlegend: Boolean((showExperiments && data.experiments) || (showNextPoint && data.suggestions)),
    legend: {
      x: 1.05,
      y: 1,
      xanchor: 'left',
      yanchor: 'top',
      bgcolor: theme === 'dark' ? 'rgba(40, 46, 56, 0.9)' : 'rgba(255, 255, 255, 0.9)',
      bordercolor: theme === 'dark' ? 'hsl(220 13% 24%)' : '#ccc',
      borderwidth: 1,
      font: {
        color: theme === 'dark' ? 'hsl(210 40% 98%)' : 'hsl(222.2 84% 4.9%)'
      }
    },
    plot_bgcolor: theme === 'dark' ? 'hsl(220 13% 13%)' : '#fafafa',
    paper_bgcolor: theme === 'dark' ? 'hsl(220 13% 16%)' : 'white',
    font: {
      color: theme === 'dark' ? 'hsl(210 40% 98%)' : 'hsl(222.2 84% 4.9%)'
    },
  }), [data.x_bounds, data.y_bounds, xAxis, yAxis, showExperiments, showNextPoint, data.experiments, data.suggestions, theme]);

  // Plotly configuration
  const config: Partial<Config> = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    toImageButtonOptions: {
      format: 'png',
      filename: `contour_${xAxis}_${yAxis}`,
      height: 600,
      width: 800,
      scale: 2
    }
  };

  return (
    <div className="w-full h-full">
      <Plot
        data={traces}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
    </div>
  );
}
