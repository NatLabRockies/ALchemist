import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithQuery } from '../../../test/queryWrapper';
import * as useQueue from '../../../hooks/api/useQueue';
import { ConfigTab } from './ConfigTab';
import { LiveTab } from './LiveTab';

// Stub heavy reused panels/plots so tabs render in isolation.
vi.mock('../../../features/variables/VariablesPanel', () => ({ VariablesPanel: () => <div>VariablesPanel</div> }));
vi.mock('../../../features/models/GPRPanel', () => ({ GPRPanel: () => <div>GPRPanel</div> }));
vi.mock('../../../features/acquisition/AcquisitionPanel', () => ({ AcquisitionPanel: () => <div>AcquisitionPanel</div> }));
vi.mock('../../../features/experiments/InitialDesignPanel', () => ({ InitialDesignPanel: () => <div>InitialDesignPanel</div> }));
vi.mock('../../../components/visualizations/MetricsPlot', () => ({ MetricsPlot: () => <div>MetricsPlot</div> }));
vi.mock('../../../components/visualizations/ParityPlot', () => ({ ParityPlot: () => <div>ParityPlot</div> }));

describe('monitor tabs', () => {
  it('ConfigTab shows the live-tuning banner and reused panels', () => {
    renderWithQuery(<ConfigTab sessionId="s1" isRunning={true} />);
    expect(screen.getByText(/does not initiate cycles/i)).toBeInTheDocument();
    expect(screen.getByText('VariablesPanel')).toBeInTheDocument();
    expect(screen.getByText('GPRPanel')).toBeInTheDocument();
  });

  it('LiveTab renders queue + plots', () => {
    vi.spyOn(useQueue, 'useExperimentQueue').mockReturnValue({
      data: { items: [], n_pending: 0, n_running: 0, n_done: 0, n_failed: 0 },
      isLoading: false,
    } as any);
    renderWithQuery(<LiveTab sessionId="s1" objectiveLabel="area (a.u.)" goal="maximize" />);
    expect(screen.getByText('MetricsPlot')).toBeInTheDocument();
    expect(screen.getByText('ParityPlot')).toBeInTheDocument();
  });
});
