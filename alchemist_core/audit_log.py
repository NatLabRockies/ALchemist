"""
Audit Log - Append-only logging for reproducible optimization workflows.

This module provides structured logging of optimization decisions to ensure
research reproducibility and traceability. The audit log captures:
- Experimental data lock-ins
- Model training decisions
- Acquisition function choices

Users can explore freely without spamming the log; only explicit "lock-in"
actions create audit entries.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict, field
import hashlib
import json
import uuid
import pandas as pd


@dataclass
class SessionMetadata:
    """
    Session metadata for user-friendly session management.
    
    Attributes:
        session_id: Unique session identifier (UUID)
        name: User-friendly session name
        created_at: ISO timestamp of session creation
        last_modified: ISO timestamp of last modification
        description: Optional detailed description
        author: Optional author name
        tags: Optional list of tags for organization
    """
    session_id: str
    name: str
    created_at: str
    last_modified: str
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    
    @staticmethod
    def create(name: str = "Untitled Session", description: str = "", 
               tags: Optional[List[str]] = None) -> 'SessionMetadata':
        """
        Create new session metadata.
        
        Args:
            name: User-friendly session name
            description: Optional description
            tags: Optional tags for organization
            
        Returns:
            SessionMetadata instance
        """
        now = datetime.now().isoformat()
        return SessionMetadata(
            session_id=str(uuid.uuid4()),
            name=name,
            created_at=now,
            last_modified=now,
            description=description,
            author="",
            tags=tags or []
        )
    
    def update_modified(self):
        """Update last_modified timestamp to now."""
        self.last_modified = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SessionMetadata':
        """Import from dictionary."""
        return SessionMetadata(**data)


@dataclass
class AuditEntry:
    """
    Single audit log entry.
    
    Attributes:
        timestamp: ISO timestamp of entry creation
        entry_type: Type of decision ('data_locked', 'model_locked', 'acquisition_locked')
        parameters: Complete snapshot of decision parameters
        hash: Reproducibility checksum (SHA256 of parameters)
        notes: Optional user notes
    """
    timestamp: str
    entry_type: str
    parameters: Dict[str, Any]
    hash: str
    notes: str = ""
    
    @staticmethod
    def create(entry_type: str, parameters: Dict[str, Any], 
               notes: str = "") -> 'AuditEntry':
        """
        Create new audit entry with auto-generated timestamp and hash.
        
        Args:
            entry_type: Type of entry ('data_locked', 'model_locked', 'acquisition_locked')
            parameters: Parameters to log
            notes: Optional user notes
            
        Returns:
            AuditEntry instance
        """
        timestamp = datetime.now().isoformat()
        
        # Create reproducibility hash
        # Sort keys for deterministic hashing
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        hash_val = hashlib.sha256(param_str.encode()).hexdigest()[:16]
        
        return AuditEntry(
            timestamp=timestamp,
            entry_type=entry_type,
            parameters=parameters,
            hash=hash_val,
            notes=notes
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AuditEntry':
        """Import from dictionary."""
        return AuditEntry(**data)


class AuditLog:
    """
    Append-only audit log for optimization decisions.
    
    This class maintains a complete, immutable history of optimization decisions
    to ensure reproducibility and traceability. Only explicit "lock-in" actions
    add entries, preventing log spam from exploration activities.
    
    The audit log is structured with:
    - Search space definition (set once)
    - Experimental data table (updated with each lock)
    - Optimization iterations (model + acquisition per iteration)
    """
    
    def __init__(self):
        """Initialize empty audit log."""
        self.entries: List[AuditEntry] = []
        self.search_space_definition: Optional[Dict[str, Any]] = None
        self.experiment_data: Optional['pd.DataFrame'] = None
    
    def set_search_space(self, variables: List[Dict[str, Any]]):
        """
        Set the search space definition (should only be called once).
        
        Args:
            variables: List of variable definitions
        """
        if self.search_space_definition is None:
            self.search_space_definition = {'variables': variables}
    
    def lock_data(self, experiment_data: 'pd.DataFrame', notes: str = "", extra_parameters: Optional[Dict[str, Any]] = None) -> AuditEntry:
        """
        Lock in experimental data snapshot.
        
        Args:
            experiment_data: DataFrame with all experimental data including Iteration and Reason
            notes: Optional user notes
            
        Returns:
            Created AuditEntry
        """
        # Store the experiment data snapshot for markdown/export
        self.experiment_data = experiment_data.copy()

        # Create hash of data for verification
        data_str = experiment_data.to_json()
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]

        params: Dict[str, Any] = {
            'n_experiments': len(experiment_data),
            'data_hash': data_hash
        }

        # Merge any extra parameters (e.g., initial design method/count)
        if extra_parameters:
            params.update(extra_parameters)

        entry = AuditEntry.create(
            entry_type='data_locked',
            parameters=params,
            notes=notes
        )
        self.entries.append(entry)
        return entry
    
    def lock_model(self, backend: str, kernel: str, 
                   hyperparameters: Dict[str, Any], 
                   cv_metrics: Optional[Dict[str, float]] = None,
                   iteration: Optional[int] = None,
                   notes: str = "") -> AuditEntry:
        """
        Lock in trained model configuration.
        
        Args:
            backend: Model backend ('sklearn', 'botorch')
            kernel: Kernel type
            hyperparameters: Learned hyperparameters
            cv_metrics: Cross-validation metrics (optional)
            iteration: Iteration number (optional)
            notes: Optional user notes
            
        Returns:
            Created AuditEntry
        """
        params = {
            'backend': backend,
            'kernel': kernel,
            'hyperparameters': hyperparameters
        }
        if cv_metrics is not None:
            params['cv_metrics'] = cv_metrics
        if iteration is not None:
            params['iteration'] = iteration
        
        entry = AuditEntry.create(
            entry_type='model_locked',
            parameters=params,
            notes=notes
        )
        self.entries.append(entry)
        return entry
    
    def lock_acquisition(self, strategy: str, parameters: Dict[str, Any],
                        suggestions: List[Dict[str, Any]], 
                        iteration: Optional[int] = None,
                        notes: str = "") -> AuditEntry:
        """
        Lock in acquisition function decision.
        
        Args:
            strategy: Acquisition strategy name
            parameters: Acquisition function parameters
            suggestions: Suggested next experiments
            iteration: Iteration number (optional)
            notes: Optional user notes
            
        Returns:
            Created AuditEntry
        """
        params = {
            'strategy': strategy,
            'parameters': parameters,
            'suggestions': suggestions
        }
        if iteration is not None:
            params['iteration'] = iteration
        
        entry = AuditEntry.create(
            entry_type='acquisition_locked',
            parameters=params,
            notes=notes
        )
        self.entries.append(entry)
        return entry
    
    def log_event(self, entry_type: str, parameters: Dict[str, Any],
                  notes: str = "") -> AuditEntry:
        """Append a generic audit entry.

        Unlike the lock_* helpers this records an arbitrary event type. Used for
        cross-cutting provenance events such as objective-label changes.
        """
        entry = AuditEntry.create(entry_type=entry_type, parameters=parameters, notes=notes)
        self.entries.append(entry)
        return entry

    def log_config_change(self, component: str, old: Dict[str, Any],
                          new: Dict[str, Any],
                          iteration: Optional[int] = None) -> AuditEntry:
        """Record a mid-campaign optimizer-config change for provenance.

        component: 'model' or 'acquisition' (opaque label; ALchemist-generic).
        old/new: config snapshots before/after the applied change.
        iteration: ALchemist iteration counter (no reactor-cycle concept here).
        """
        parameters: Dict[str, Any] = {"component": component, "old": old, "new": new}
        if iteration is not None:
            parameters["iteration"] = iteration
        return self.log_event(
            entry_type="config_changed",
            parameters=parameters,
            notes=f"{component} config changed",
        )

    def get_entries(self, entry_type: Optional[str] = None) -> List[AuditEntry]:
        """
        Get audit entries, optionally filtered by type.
        
        Args:
            entry_type: Optional filter ('data_locked', 'model_locked', 'acquisition_locked')
            
        Returns:
            List of AuditEntry objects
        """
        if entry_type is None:
            return self.entries.copy()
        return [e for e in self.entries if e.entry_type == entry_type]
    
    def get_latest(self, entry_type: str) -> Optional[AuditEntry]:
        """
        Get most recent entry of specified type.
        
        Args:
            entry_type: Entry type to find
            
        Returns:
            Latest AuditEntry or None if not found
        """
        entries = self.get_entries(entry_type)
        return entries[-1] if entries else None
    
    def clear(self):
        """
        Clear all entries (use with caution - breaks immutability contract).
        
        This should only be used when starting a completely new optimization
        campaign within the same session.
        """
        self.entries = []
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """
        Export audit log to dictionary format.
        
        Returns:
            Dictionary with search_space, experiment_data, and entries
        """
        result = {
            'entries': [entry.to_dict() for entry in self.entries]
        }
        
        if self.search_space_definition is not None:
            result['search_space'] = self.search_space_definition
        
        if self.experiment_data is not None:
            result['experiment_data'] = self.experiment_data.to_dict(orient='records')
        
        return result
    
    def from_dict(self, data: Union[List[Dict[str, Any]], Dict[str, Any]]):
        """
        Import audit log from dictionary format.
        
        Args:
            data: Dictionary with entries (and optionally search_space and experiment_data)
                  or legacy list of entry dictionaries
        """
        # Handle legacy format (list of entries)
        if isinstance(data, list):
            self.entries = [AuditEntry.from_dict(entry) for entry in data]
            return
        
        # New format (dict with entries, search_space, experiment_data)
        if 'entries' in data:
            self.entries = [AuditEntry.from_dict(entry) for entry in data['entries']]
        
        if 'search_space' in data:
            self.search_space_definition = data['search_space']
        
        if 'experiment_data' in data:
            self.experiment_data = pd.DataFrame(data['experiment_data'])
    
    def to_markdown(self, session_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Export audit log to markdown format for publications.

        Args:
            session_metadata: Optional dictionary of session metadata (name, description, tags, created_at, etc.)

        Returns:
            Markdown-formatted audit trail with session metadata, search space, data table, and iterations
        """
        lines = ["# Optimization Audit Trail\n"]

        # If session metadata provided, include a small metadata section
        if session_metadata:
            lines.append("## Session Metadata\n")
            name = session_metadata.get('name', 'Untitled Session')
            created = session_metadata.get('created_at', '')
            last_mod = session_metadata.get('last_modified', '')
            description = session_metadata.get('description', '')
            tags = session_metadata.get('tags', [])

            lines.append(f"- **Name**: {name}")
            if created:
                lines.append(f"- **Created At**: {created}")
            if last_mod:
                lines.append(f"- **Last Modified**: {last_mod}")
            if description:
                lines.append(f"- **Description**: {description}")
            if tags:
                if isinstance(tags, (list, tuple)):
                    tags_str = ', '.join(map(str, tags))
                else:
                    tags_str = str(tags)
                lines.append(f"- **Tags**: {tags_str}")

            lines.append("")
        
        # Section 1: Search Space Definition
        if self.search_space_definition:
            lines.append("## Search Space Definition\n")
            for var in self.search_space_definition['variables']:
                var_type = var['type']
                name = var['name']
                
                if var_type in ['real', 'integer']:
                    lines.append(f"- **{name}** ({var_type}): [{var.get('min', 'N/A')}, {var.get('max', 'N/A')}]")
                else:  # categorical
                    values = ', '.join(map(str, var.get('values', [])))
                    lines.append(f"- **{name}** (categorical): {{{values}}}")
            lines.append("")
        
        # Section 2: Experimental Data Table
        if self.experiment_data is not None and len(self.experiment_data) > 0:
            lines.append("## Experimental Data\n")
            
            # Generate markdown table
            df = self.experiment_data.copy()
            
            # Reorder columns: Iteration, Reason, then variables, then Output
            col_order = []
            if 'Iteration' in df.columns:
                col_order.append('Iteration')
            if 'Reason' in df.columns:
                col_order.append('Reason')
            
            # Add variable columns (exclude metadata columns)
            metadata_cols = {'Iteration', 'Reason', 'Output', 'Noise', 'ProvenanceId'}
            var_cols = [col for col in df.columns if col not in metadata_cols]
            col_order.extend(var_cols)
            
            # Add Output
            if 'Output' in df.columns:
                col_order.append('Output')
            
            # Reorder DataFrame
            df = df[[col for col in col_order if col in df.columns]]
            
            # Create markdown table
            lines.append("| " + " | ".join(df.columns) + " |")
            lines.append("|" + "|".join(['---'] * len(df.columns)) + "|")
            
            for _, row in df.iterrows():
                row_vals = []
                for val in row:
                    if isinstance(val, float):
                        row_vals.append(f"{val:.4f}")
                    else:
                        row_vals.append(str(val))
                lines.append("| " + " | ".join(row_vals) + " |")
            
            lines.append("")
        
        # Section 3: Optimization Iterations
        if len(self.entries) > 0:
            lines.append("## Optimization Iterations\n")
            
            # Group entries by iteration and track timestamps
            iterations: Dict[Union[int, str], Dict[str, Any]] = {}

            for entry in self.entries:
                # Prefer explicit iteration in parameters when available
                iteration = entry.parameters.get('iteration', None)

                # Special-case data_locked entries that include initial-design metadata
                if entry.entry_type == 'data_locked' and 'initial_design_method' in entry.parameters:
                    iteration = 0

                if iteration is None:
                    iteration_key = 'N/A'
                else:
                    iteration_key = iteration

                if iteration_key not in iterations:
                    iterations[iteration_key] = {
                        'model': None,
                        'acquisition': None,
                        'data': None,
                        'timestamp': entry.timestamp
                    }

                if entry.entry_type == 'model_locked':
                    iterations[iteration_key]['model'] = entry
                elif entry.entry_type == 'acquisition_locked':
                    iterations[iteration_key]['acquisition'] = entry
                elif entry.entry_type == 'data_locked':
                    iterations[iteration_key]['data'] = entry
            
            # Sort iterations: numeric iteration keys first (ascending), then 'N/A'
            def sort_key(item):
                iter_num, data = item
                is_na = (iter_num == 'N/A')
                # Primary: whether N/A (False comes before True), secondary: iteration number or large sentinel
                num_key = iter_num if isinstance(iter_num, int) else 999999
                # Use the stored timestamp as tie-breaker
                return (is_na, num_key, data.get('timestamp', ''))
            
            # Output each iteration (skip N/A entries if they have no data)
            for iter_num, iter_data in sorted(iterations.items(), key=sort_key):
                # Skip N/A iteration if it has no model or acquisition
                if iter_num == 'N/A' and not iter_data['model'] and not iter_data['acquisition']:
                    continue
                
                lines.append(f"### Iteration {iter_num}\n")
                
                # Model information
                if iter_data.get('model'):
                    entry = iter_data['model']
                    params = entry.parameters

                    lines.append(f"**Timestamp**: {entry.timestamp}")
                    lines.append("")

                    # Build kernel string with nu parameter if Matern
                    kernel = params.get('kernel', 'N/A')
                    kernel_str = f"{kernel} kernel"
                    hyperparams = params.get('hyperparameters', {})

                    # Try common keys for matern nu
                    matern_nu = None
                    if kernel == 'Matern':
                        matern_nu = hyperparams.get('matern_nu') or hyperparams.get('nu')
                        if matern_nu is None:
                            # Also try params top-level hyperparameters representation
                            matern_nu = params.get('hyperparameters', {}).get('matern_nu')

                    if kernel == 'Matern' and matern_nu is not None:
                        kernel_str = f"{kernel} kernel (ν={matern_nu})"

                    lines.append(f"**Model**: {params.get('backend', 'N/A')}, {kernel_str}")
                    lines.append("")

                    if 'cv_metrics' in params and params['cv_metrics']:
                        metrics = params['cv_metrics']
                        r2 = metrics.get('r2', 0)
                        rmse = metrics.get('rmse', 0)
                        lines.append(f"**Metrics**: R²={r2:.4f}, RMSE={rmse:.4f}")
                    else:
                        lines.append(f"**Metrics**: Not available")

                    # Display input/output scaling if provided in hyperparameters
                    input_scale = hyperparams.get('input_scaling') or hyperparams.get('input_transform_type')
                    output_scale = hyperparams.get('output_scaling') or hyperparams.get('output_transform_type')
                    if input_scale is not None or output_scale is not None:
                        lines.append("")
                        lines.append(f"**Input Scaling**: {input_scale if input_scale is not None else 'none'}")
                        lines.append(f"**Output Scaling**: {output_scale if output_scale is not None else 'none'}")

                    if entry.notes:
                        lines.append("")
                        lines.append(f"**Notes**: {entry.notes}")

                    lines.append("")
                
                # Acquisition information
                if iter_data.get('acquisition'):
                    entry = iter_data['acquisition']
                    params = entry.parameters

                    lines.append(f"**Acquisition**: {params.get('strategy', 'N/A')}")
                    lines.append("")

                    if 'parameters' in params and params['parameters']:
                        acq_params = params['parameters']
                        param_str = ', '.join([f"{k}={v}" for k, v in acq_params.items()])
                        lines.append(f"**Parameters**: {param_str}")
                        lines.append("")

                    if 'suggestions' in params and params['suggestions']:
                        suggestions = params['suggestions']
                        lines.append(f"**Suggested Next**: {suggestions}")

                    if entry.notes:
                        lines.append("")
                        lines.append(f"**Notes**: {entry.notes}")

                    lines.append("")

                # Data information (e.g., initial design)
                if iter_data.get('data'):
                    entry = iter_data['data']
                    params = entry.parameters
                    # If initial design metadata present, print it clearly
                    method = params.get('initial_design_method')
                    n_points = params.get('initial_design_n_points')
                    if method:
                        lines.append(f"**Initial Design**: {method} ({n_points if n_points is not None else params.get('n_experiments', 'N/A')} points)")
                        lines.append("")
                    # Optionally include notes for data lock
                    if entry.notes:
                        lines.append(f"**Notes**: {entry.notes}")
                        lines.append("")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        """Return number of entries."""
        return len(self.entries)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"AuditLog({len(self.entries)} entries)"
