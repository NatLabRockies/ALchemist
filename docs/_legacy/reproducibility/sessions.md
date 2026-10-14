# Session Management

**Sessions** are your optimization project containers. Each session stores your variables, experimental data, trained models, and complete history. Sessions can be saved, loaded, and shared across the desktop and web applications.

---

## Using Sessions in the Application

### Creating a New Session

**Desktop or Web App:**

1. Click "New Session" (or File → New Session on desktop)
2. Enter a project name (e.g., "Catalyst Screening Dec 2025")
3. Optionally add a description
4. Click Create

Your session is now ready. Start by defining your variables.

### Opening an Existing Session

**Desktop App:**

- File → Open Session → Select your `.json` file

- Or File → Recent Sessions to quickly reopen

**Web App:**

- Click "Load Session"

- Select from your available sessions

- Session loads with all your data intact

### Saving Your Session

**Desktop App:**

- File → Save Session (or Cmd+S / Ctrl+S)

- Save after adding data, training models, or generating suggestions

**Web App:**

- Sessions auto-save as you work

- No manual save needed

**Where sessions are stored:** `cache/sessions/` directory as `.json` files

### Sharing Sessions

Sessions are just JSON files you can share with colleagues:

1. Find your session file in `cache/sessions/`
2. Share it via email, Dropbox, Git, etc.
3. Colleague places it in their `cache/sessions/` folder
4. They can open it in desktop or web app

**Sessions work across both apps** - create in web, open in desktop, or vice versa.



---

## Advanced: Programmatic Session Management

For users writing Python scripts or integrating ALchemist into analysis pipelines:

### Creating Sessions Programmatically

```python
from alchemist_core import OptimizationSession

# Create new session
session = OptimizationSession()
session.metadata.name = "Catalyst Screening"
session.metadata.description = "Pd catalyst optimization"

# Define search space
session.add_variable('temperature', 'real', bounds=(20, 100), unit='°C')
session.add_variable('catalyst_loading', 'real', bounds=(0.1, 5.0), unit='mol%')
session.add_variable('solvent', 'categorical', categories=['THF', 'DMF', 'toluene'])

# Add experimental data
session.add_experiment({
    'temperature': 60,
    'catalyst_loading': 2.5,
    'solvent': 'THF'
}, output=85.3)

# Save session
session.save_session('cache/sessions/')
```

### Loading and Using Sessions

```python
# Load existing session
session = OptimizationSession.load_session('cache/sessions/abc123.json')

# Access session state
print(f"Session: {session.metadata.name}")
print(f"Variables: {len(session.search_space.variables)}")
print(f"Experiments: {len(session.experiment_manager.data)}")

# Continue optimization
session.train_model(backend='botorch')
next_point = session.suggest_next(strategy='EI', goal='maximize')

# Save updates
session.save_session()
```

### Session File Structure

Sessions are stored as JSON files with this structure:

```json
{
  "metadata": {
    "session_id": "abc123-def456-...",
    "name": "Catalyst Screening",
    "created_at": "2025-12-12T10:00:00",
    "last_modified": "2025-12-12T15:30:00"
  },
  "search_space": {
    "variables": [...]
  },
  "experiments": {
    "data": [...]
  },
  "model": {
    "backend": "botorch",
    "hyperparameters": {...}
  },
  "audit_log": {
    "entries": [...]
  }
}
```

The JSON format is human-readable and can be inspected with any text editor
git commit -m "Add catalyst screening session"
git push

# Collaborator pulls
git pull
# Session now available in their ALchemist
```

### Best Practices

**Version control**:

- Use Git LFS for large session files (> 1 MB)

- Commit session files at milestones

- Tag important versions

**Naming conventions**:

- Use descriptive names: "Catalyst_Screen_2025-12"

- Avoid special characters

- Include date or version in name

**Documentation**:

- Use description field extensively

- Add tags for organization

- Include contact info in author field

---

## Session Organization

### Tagging System

**Purpose**: Organize sessions by project, topic, date, or priority

**Examples**:

- Project tags: `"project-alpha"`, `"project-beta"`

- Topic tags: `"catalysis"`, `"materials"`, `"process-optimization"`

- Date tags: `"2025"`, `"Q4-2025"`

- Status tags: `"active"`, `"completed"`, `"on-hold"`

- Priority tags: `"high-priority"`, `"exploratory"`

**Usage**:
```python
session.metadata.tags = [
    "catalysis",
    "suzuki",
    "2025",
    "high-priority"
]
```

### Directory Organization

**Recommended Structure**:
```
cache/sessions/
├── active/
│   ├── catalyst_screen_2025-12.json
│   └── process_optimization_2025-11.json
├── completed/
│   ├── materials_discovery_2025-10.json
│   └── kinetics_study_2025-09.json
└── archive/
    ├── old_project_2024.json
    └── failed_experiment_2024.json
```

**Implementation**:
```python
# Save to organized directory
session.save_session('cache/sessions/active/')
```

---

## Session State Management

### What's Saved

**Always Saved**:

- Variable space definition

- All experimental data

- Session metadata

- Audit log entries

**Optionally Saved**:

- Trained model state (if trained)

- Model hyperparameters

- Training metrics (CV results)

- Acquisition suggestions (last batch)

### What's Not Saved

**Excluded from session files**:

- Visualization plots (regenerated on demand)

- Large intermediate arrays

- UI preferences (stored separately)

- Temporary scratch data

**Why**: Keep session files lightweight and portable

### State Consistency

**Validation on load**:

- Variable space matches experimental data

- Data integrity checks

- Model compatibility verification

- Audit log hash validation

**Error Handling**:
```python
try:
    session = OptimizationSession.load_session('path/to/session.json')
except ValueError as e:
    print(f"Session validation failed: {e}")
    # Handle corrupted or incompatible session
```

