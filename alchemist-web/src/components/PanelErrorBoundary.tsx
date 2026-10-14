/**
 * Keeps a crash inside one panel from blanking the whole app.
 *
 * Without a boundary, an exception while rendering any panel unmounts the
 * entire React tree, header included, so the user can't even save (W25). The
 * fallback names the panel and offers a retry; the header stays usable.
 */
import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  /** Shown in the fallback, e.g. "Model". */
  name: string;
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export class PanelErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`The ${this.props.name} panel crashed:`, error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" className="m-4 rounded-md border border-destructive/30 bg-destructive/5 p-4 text-sm">
        <p className="font-medium text-destructive">The {this.props.name} panel hit an error.</p>
        <p className="mt-1 text-xs text-muted-foreground break-words">{this.state.error.message}</p>
        <p className="mt-2 text-xs text-muted-foreground">
          The rest of the app still works, and your session is safe on the server.
        </p>
        <button
          type="button"
          onClick={() => this.setState({ error: null })}
          className="mt-3 rounded border border-input px-3 py-1.5 text-xs hover:bg-accent"
        >
          Try again
        </button>
      </div>
    );
  }
}
