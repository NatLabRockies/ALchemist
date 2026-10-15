/**
 * Contour Plot - 2D surface plot of model predictions
 * Mirrors desktop UI contour plot from visualizations.py with side controls
 * Uses Plotly.js for professional contour visualization
 */
import { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import { Loader2 } from 'lucide-react';
import Plot from 'react-plotly.js';
import type { Data, Layout, Config } from 'plotly.js';
import { useContourData } from '../../hooks/api/useVisualizations';
import { useVariables } from '../../hooks/api/useVariables';
import type { VariableDetail } from '../../api/types';
import { useTheme } from '../../hooks/useTheme';
import { FixedRealControl } from './FixedRealControl';
import { plotlyFonts, usePlotTheme } from './plotTheme';
import { CONTOUR_MARGINS, contourFigureSize, growMargins, type FigureSize, type Margins } from './contourSizing';
import {
  contourZRange,
  readContourColorScale,
  storeContourColorScale,
  usableRange,
  type ContourColorScale,
} from './contourColorScale';

// Quiet period after the last fixed-value change before the contour is requested.
const FIXED_VALUE_DEBOUNCE_MS = 150;

/**
 * Size of the box the contour sits in: the parent of `root`, which fills the visible part of the
 * visualization column. `root` is state set by a callback ref, so the observer attaches whenever
 * the element mounts, including after an early return on the first render. Null until measured (and in test environments without ResizeObserver);
 * a hidden tab reports 0 and is ignored so the last real size survives.
 */
function useColumnBox(root: HTMLElement | null): FigureSize | null {
  const [box, setBox] = useState<FigureSize | null>(null);

  useEffect(() => {
    const el = root?.parentElement;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const measure = () => {
      if (el.clientWidth === 0) return;
      const next = { width: el.clientWidth, height: el.clientHeight };
      setBox((prev) => (prev && prev.width === next.width && prev.height === next.height ? prev : next));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [root]);

  return box;
}

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
  const [rootEl, setRootEl] = useState<HTMLDivElement | null>(null);
  const columnBox = useColumnBox(rootEl);

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
  const [colorScale, setColorScale] = useState<ContourColorScale>(readContourColorScale);
  const chooseColorScale = useCallback((scale: ContourColorScale) => {
    setColorScale(scale);
    storeContourColorScale(scale);
  }, []);

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

  // Fetch contour data (use committedFixedValues - they settle after the last fixed-value change)
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
    isFetching,
    error,
  } = useContourData(sessionId, contourRequest!, !!contourRequest);

  // Pending debounced commits, one per variable
  const commitTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  const cancelCommit = useCallback((varName: string) => {
    clearTimeout(commitTimers.current[varName]);
    delete commitTimers.current[varName];
  }, []);

  // A new fixed-value set (axes changed) replaces the pending ones; leaving drops them.
  useEffect(() => {
    const timers = commitTimers.current;
    return () => {
      Object.values(timers).forEach((timer) => clearTimeout(timer));
    };
  }, [xAxis, yAxis, variables]);

  // Handler for updating fixed values (updates local state while dragging)
  const handleFixedValueChange = useCallback((varName: string, value: number | string) => {
    setFixedValues((prev) => ({
      ...prev,
      [varName]: { ...prev[varName], value },
    }));
  }, []);

  // Handler for committing fixed values (triggers the API request)
  const handleFixedValueCommit = useCallback((varName: string, value: number | string) => {
    cancelCommit(varName);
    setCommittedFixedValues((prev) => ({
      ...prev,
      [varName]: { ...prev[varName], value },
    }));
  }, [cancelCommit]);

  // A change that may be followed by more (slider drag, arrow keys): show it now, request the
  // contour once the value has stopped changing.
  const handleFixedValueScrub = useCallback((varName: string, value: number) => {
    handleFixedValueChange(varName, value);
    cancelCommit(varName);
    commitTimers.current[varName] = setTimeout(() => {
      delete commitTimers.current[varName];
      handleFixedValueCommit(varName, value);
    }, FIXED_VALUE_DEBOUNCE_MS);
  }, [handleFixedValueChange, handleFixedValueCommit, cancelCommit]);

  // The fixed scale needs the observed output range; without one (no span to colour across) the
  // slice's own range is used, and the remembered choice is left as it is.
  const outputRange = usableRange(contourApiData?.output_range);
  const fixedAvailable = !contourApiData || outputRange !== null;
  const effectiveScale: ContourColorScale = fixedAvailable ? colorScale : 'per-slice';

  // The figure sets its own height from the margins Plotly used; the loading and error states hold the space.
  const showFigure = !isLoading && !error && Boolean(contourApiData);

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
    <div ref={setRootEl} className="flex flex-col gap-3">
      {/* Main plot area: sized from the column's width, so the controls follow directly below */}
      <div
        className="flex flex-col"
        style={showFigure ? undefined : { minHeight: columnBox ? contourFigureSize(columnBox.width, columnBox.height).height : 360 }}
      >
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
          <div className="relative">
            {/* The previous figure stays up while the next one loads. */}
            {isFetching && (
              <div
                role="status"
                aria-label="Updating contour plot"
                className="absolute top-2 right-2 z-10 rounded-full bg-background/80 p-1"
              >
                <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
              </div>
            )}
            {/* Dimmed while the kept figure is out of date, so it reads as loading. */}
            <div className={`transition-opacity ${isFetching ? 'opacity-60' : ''}`}>
              <ContourPlotly
                data={contourApiData}
                xAxis={contourApiData.x_var}
                yAxis={contourApiData.y_var}
                showExperiments={showExperiments}
                showNextPoint={showNextPoint}
                colormap={colormap}
                colorScale={effectiveScale}
                theme={theme}
                columnBox={columnBox}
              />
            </div>
          </div>
        )}
      </div>

      {/* Controls below plot */}
      <div className="bg-muted/30 rounded-lg p-3 flex-shrink-0">
        {/* Wraps to as many columns as the column is wide; Fixed Values takes the full first row. */}
        <div className="grid grid-cols-[repeat(auto-fit,minmax(10rem,1fr))] gap-x-4 gap-y-3">
          {/* Fixed Values: used most, so it gets the first, full-width row directly under the figure */}
          <div className="space-y-2 col-span-full">
            <h3 className="font-semibold text-xs uppercase tracking-wide text-muted-foreground border-b pb-1.5">Fixed Values</h3>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(10rem,1fr))] gap-x-4 gap-y-2 max-h-[11rem] overflow-y-auto pr-2">
              {Object.entries(fixedValues).map(([varName, varInfo]) => (
                <div key={varName}>
                  {varInfo.type !== 'Real' && (
                    <label className="text-xs font-medium block mb-1">{varName}:</label>
                  )}
                  {varInfo.type === 'Real' && (
                    <FixedRealControl
                      name={varName}
                      min={varInfo.min!}
                      max={varInfo.max!}
                      value={varInfo.value as number}
                      onScrub={(v) => handleFixedValueScrub(varName, v)}
                      onCommit={(v) => handleFixedValueCommit(varName, v)}
                    />
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
            <ColorScaleControl
              value={effectiveScale}
              onChange={chooseColorScale}
              fixedAvailable={fixedAvailable}
              outputRange={outputRange}
            />
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

const COLOR_SCALE_OPTIONS: { value: ContourColorScale; label: string }[] = [
  { value: 'fixed', label: 'Fixed' },
  { value: 'per-slice', label: 'Per slice' },
];

const rangeFormat = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 3 });

interface ColorScaleControlProps {
  value: ContourColorScale;
  onChange: (scale: ContourColorScale) => void;
  /** False when the session has no observed output range to fix the scale to. */
  fixedAvailable: boolean;
  outputRange: [number, number] | null;
}

/**
 * Fixed / Per slice, as two joined buttons under the colormap: the colour bar either keeps the
 * observed output range for every slice or fits each slice's own min and max.
 */
function ColorScaleControl({ value, onChange, fixedAvailable, outputRange }: ColorScaleControlProps) {
  let hint: string;
  if (!fixedAvailable) hint = 'No observed output range to fix the scale to.';
  else if (value === 'per-slice') hint = "Colors span this slice's own min and max.";
  else if (outputRange) hint = `Observed outputs, ${rangeFormat.format(outputRange[0])} to ${rangeFormat.format(outputRange[1])}, on every slice.`;
  else hint = 'Observed output range, on every slice.';

  return (
    <div>
      <span id="contour-color-scale-label" className="text-xs font-medium block mb-1">Scale:</span>
      <div
        role="group"
        aria-labelledby="contour-color-scale-label"
        aria-describedby="contour-color-scale-hint"
        className="grid grid-cols-2 rounded border border-input overflow-hidden text-xs"
      >
        {COLOR_SCALE_OPTIONS.map((option, i) => {
          const selected = value === option.value;
          const disabled = option.value === 'fixed' && !fixedAvailable;
          return (
            <button
              key={option.value}
              type="button"
              aria-pressed={selected}
              disabled={disabled}
              onClick={() => onChange(option.value)}
              className={`px-2 py-1 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50 ${
                i > 0 ? 'border-l border-input' : ''
              } ${
                selected
                  ? 'bg-primary text-primary-foreground font-medium'
                  : 'bg-background text-muted-foreground hover:bg-accent hover:text-foreground disabled:hover:bg-background disabled:hover:text-muted-foreground'
              }`}
            >
              {option.label}
            </button>
          );
        })}
      </div>
      <p id="contour-color-scale-hint" className="text-xs text-muted-foreground mt-1">{hint}</p>
    </div>
  );
}

/**
 * Plotly-based contour plot renderer
 * Provides professional, interactive contour visualization matching desktop UI
 */
/** The parts of Plotly's graph div that the margin check reads (`_fullLayout` is Plotly-internal). */
interface PlotlyGraphDiv {
  _fullLayout?: { width: number; height: number; _size?: { l: number; t: number; w: number; h: number } };
}

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
    output_range?: number[] | null;
  };
  xAxis: string;
  yAxis: string;
  showExperiments: boolean;
  showNextPoint: boolean;
  colormap: string;
  /** Fixed: the colours span the observed output range on every slice; per-slice: this slice's range. */
  colorScale: ContourColorScale;
  theme: 'light' | 'dark';
  /** The column the figure sits in; null until it has been measured. */
  columnBox: FigureSize | null;
}

function ContourPlotly({ 
  data, 
  xAxis, 
  yAxis, 
  showExperiments, 
  showNextPoint,
  colormap,
  colorScale,
  theme,
  columnBox
}: ContourPlotlyProps) {
  // Plotly widens a margin when the colorbar's labels or title need it, which would make the plot
  // area wider or narrower than the sizing assumed. Read the margins it used and size from those.
  const [margins, setMargins] = useState<Margins>(CONTOUR_MARGINS);
  const size = useMemo(
    () => (columnBox ? contourFigureSize(columnBox.width, columnBox.height, margins) : null),
    [columnBox, margins],
  );
  const graphDivRef = useRef<PlotlyGraphDiv | null>(null);
  const readMargins = useCallback(() => {
    const full = graphDivRef.current?._fullLayout;
    if (!full?._size) return;
    const { l, t, w, h } = full._size;
    setMargins((prev) => growMargins(prev, { l, t, r: full.width - l - w, b: full.height - t - h }));
  }, []);
  // Read after every draw: the one that creates the plot, one that updates it, and afterplot,
  // which also covers Plotly's own follow-up redraws.
  const captureGraphDiv = useCallback((_figure: unknown, graphDiv: unknown) => {
    graphDivRef.current = graphDiv as PlotlyGraphDiv;
    readMargins();
  }, [readMargins]);

  // The plot's text is set in the UI's font and size scale (see plotTheme)
  const { fontFamily } = usePlotTheme();
  const textColor = theme === 'dark' ? 'hsl(210 40% 98%)' : 'hsl(222.2 84% 4.9%)';
  const fonts = useMemo(() => plotlyFonts(fontFamily, textColor), [fontFamily, textColor]);

  // Extract 1D arrays from 2D meshgrids for Plotly
  // x_grid is constant along columns, y_grid is constant along rows
  const xValues = data.x_grid[0]; // First row contains all unique x values
  const yValues = data.y_grid.map(row => row[0]); // First column contains all unique y values

  // Build traces array
  const traces: Data[] = [];

  // Fixed: pin the colours (and Plotly's automatic contour levels, which follow zmin/zmax) to the
  // observed outputs. A prediction beyond them saturates at the end colour; hover shows its value.
  const zRange = contourZRange(colorScale, data.output_range);

  // Main contour trace
  const contourTrace: Data = {
    type: 'contour',
    x: xValues,
    y: yValues,
    z: data.predictions,
    colorscale: colormap,
    ...(zRange ? { zauto: false, zmin: zRange.zmin, zmax: zRange.zmax } : { zauto: true }),
    colorbar: {
      title: {
        text: 'Prediction',
        side: 'right',
        font: fonts.axisTitle,
      },
      thickness: 14,
      len: 0.7,
      outlinewidth: 0,
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
      font: fonts.title,
    },
    xaxis: {
      title: {
        text: xAxis,
        font: fonts.axisTitle
      },
      range: data.x_bounds,
      showgrid: true,
      zeroline: false,
    },
    yaxis: {
      title: {
        text: yAxis,
        font: fonts.axisTitle
      },
      range: data.y_bounds,
      showgrid: true,
      zeroline: false,
    },
    // An explicit size keeps the plot area no taller than wide (see contourSizing); autosize
    // only applies before the column has been measured.
    autosize: !size,
    ...(size ? { width: size.width, height: size.height } : {}),
    // The right margin holds the colorbar, its tick labels and its title only; the legend sits
    // above the plot, so the plot area keeps its width in a narrow visualization column.
    margin: margins,
    hovermode: 'closest',
    showlegend: Boolean((showExperiments && data.experiments) || (showNextPoint && data.suggestions)),
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.02,
      xanchor: 'left',
      yanchor: 'bottom',
      bgcolor: theme === 'dark' ? 'rgba(40, 46, 56, 0.9)' : 'rgba(255, 255, 255, 0.9)',
      bordercolor: theme === 'dark' ? 'hsl(220 13% 24%)' : '#ccc',
      borderwidth: 1,
      font: fonts.legend,
    },
    plot_bgcolor: theme === 'dark' ? 'hsl(220 13% 13%)' : '#fafafa',
    paper_bgcolor: theme === 'dark' ? 'hsl(220 13% 16%)' : 'white',
    font: fonts.base,
    hoverlabel: { font: fonts.hover },
  }), [fonts, data.x_bounds, data.y_bounds, xAxis, yAxis, showExperiments, showNextPoint, data.experiments, data.suggestions, theme, size, margins]);

  // Plotly configuration
  const config: Partial<Config> = {
    // The ResizeObserver on the column resizes the figure; Plotly's own resize would refill it.
    responsive: false,
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
    <div className="w-full">
      <Plot
        data={traces}
        layout={layout}
        config={config}
        style={size ? { width: size.width, height: size.height } : { width: '100%', height: 360 }}
        useResizeHandler={!size}
        onInitialized={captureGraphDiv}
        onUpdate={captureGraphDiv}
        onAfterPlot={readMargins}
      />
    </div>
  );
}
