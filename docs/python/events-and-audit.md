# Events and audit log

A session announces what it does through events, and records the decisions behind a campaign in an audit log. Events are for your code; the audit log is for people to read afterwards, as a methods section or a record of what was run and why.

## Events

An event is a name and a dictionary. Subscribe with `session.on(name, callback)`, and the callback runs each time the session emits that name, with the dictionary as its one argument. This listener counts what a short campaign does.

```python
from alchemist_core import OptimizationSession

session = OptimizationSession()
heard = []

for name in ("variable_added", "initial_design_generated", "experiment_added",
             "training_completed", "acquisition_completed"):
    session.on(name, lambda data, name=name: heard.append((name, sorted(data))))

session.add_variable("temperature", "real", bounds=(200.0, 300.0))
session.add_variable("pressure", "real", bounds=(20.0, 80.0))
for p in session.generate_initial_design("lhs", n_points=6, random_seed=1):
    y = -((p["temperature"] - 250) / 50) ** 2 - ((p["pressure"] - 50) / 30) ** 2
    session.add_experiment(p, output=y, reason="Initial Design")
session.train_model(backend="sklearn")
suggestion = session.suggest_next(strategy="EI", goal="maximize")

print(len(heard))
print(heard[-1])
```

`session.on` is shorthand for `session.events.on`; both register on the session's `EventEmitter`, which also has `once`, `off`, `remove_all_listeners`, `listener_count` and `event_names`. Adding the same callback twice for one name has no effect.

!!! warning "Known issue"
    A32: when a `once` listener fires, the next listener registered for that name is skipped for that emit, and `off` cannot remove a `once` listener before it fires. Use `on` and remove the listener yourself.

Listeners run synchronously, on the thread that made the call, in the order they were added. A listener that raises is logged and skipped, and the session call carries on. A name nothing emits is accepted without complaint, so a misspelt name never fires; check it against the table below.

### The events a session emits

| Event | When | Payload keys |
|---|---|---|
| `variable_added` | `add_variable` | `name`, `type` |
| `derived_variable_added` | `add_derived_variable` | `name`, `input_cols` |
| `search_space_loaded` | `load_search_space` | `filepath` |
| `data_loaded` | `load_data` | `n_experiments`, `filepath` |
| `experiment_added` | `add_experiment`, and each recorded queue item | `inputs`, `output` |
| `experiment_staged`, `staged_experiments_cleared` | `add_staged_experiment`; `clear_staged_experiments` and `move_staged_to_experiments` when they removed any | `inputs`; `count` |
| `queue_item_updated` | An item is staged, started, completed or failed | `item_id`, `status`, `reason`, `output`, `error` |
| `queue_updated` | The queue changed in any way, including delete and purge | `n_pending`, `n_running`, `n_done`, `n_failed` |
| `initial_design_generated` | `generate_initial_design`, `generate_split_plot_design`, `generate_optimal_design` | `method`, `n_points`; an optimal design adds `criterion` and `D_eff` |
| `training_started` | `train_model` begins | `backend`, `kernel` |
| `training_completed` | `train_model` succeeds | `backend`, `kernel`, `hyperparameters`, `metrics`, `success`, `n_objectives` |
| `acquisition_started` | `suggest_next` begins | `strategy`, `goal` |
| `acquisition_completed` | `suggest_next` returns | `suggestion`, the first suggested point only |
| `data_locked`, `model_locked`, `acquisition_locked` | `lock_data`, `lock_model`, `lock_acquisition` | `entry`, the audit entry as a dictionary |
| `session_saved` | `save_session` | `filepath` |
| `session_loaded` | `load_session` | `filepath` |
| `metadata_updated` | `update_metadata` | The metadata fields: `session_id`, `name`, `created_at`, `last_modified`, `description`, `author`, `tags` |
| `model_retrained`, `model_retrain_failed` | The retrain step of `load_session` | `backend`, or `error` |

`complete_experiment` adds a row, so it emits `experiment_added` as well as the queue events. A failed training emits `training_started` and no `training_completed`. Nothing emits `model_trained` or `acquisition_generated`.

!!! warning "Known issue"
    A30: `model_retrained` and `model_retrain_failed` are emitted on the session that `load_session` builds internally, so a listener you registered never receives them. With the instance form, `session.load_session(path)`, `session_loaded` is the only load event you hear. The class form, `OptimizationSession.load_session(path)`, returns a new session, so there is nothing to subscribe to beforehand. Call `train_model` yourself after loading if you need to know that the model was rebuilt.

The REST server runs sessions in the same process, so these events fire there too, but nothing forwards them; the server's WebSocket sends messages of its own, such as `experiments_updated`. To give several sessions one emitter, pass `event_emitter=` to `OptimizationSession`; the work queue always uses its own session's emitter.

## The audit log

The audit log holds the decisions a campaign was built on. Exploring writes nothing to it; an entry appears when something is committed, which keeps the log short enough to read. Nothing edits or removes an entry once it is written, except `AuditLog.clear()`, which starts a new campaign in the same session.

Each entry has a `timestamp`, an `entry_type`, a `parameters` dictionary that differs by type, free-text `notes`, and a `hash`: the first 16 hexadecimal characters of the SHA-256 of `parameters`, serialized as JSON with sorted keys.

### Entry types

| `entry_type` | Written by | `parameters` |
|---|---|---|
| `data_locked` | `lock_data`, and by itself when you generate an initial, optimal or split-plot design | `n_experiments` and `data_hash`; a design adds `initial_design_method` and `initial_design_n_points`, and an optimal design `criterion` and `D_eff` |
| `model_locked` | `lock_model` | `backend`, `kernel`, `hyperparameters`, `cv_metrics` when available, and `iteration` |
| `acquisition_locked` | `lock_acquisition` | `strategy`, `parameters`, `suggestions` and `iteration` |
| `experiment_completed` | `complete_experiment`, and the REST routes that record a result, manual ones included | The [provenance](../automation/provenance.md) record: suggested and actual conditions, their difference and the result |
| `config_changed` | The REST routes that train a model and ask for a suggestion, which the web app uses | `component` (`model` or `acquisition`), `old`, `new` and a running count as `iteration`; `old` is always empty |
| `objective_label_changed` | `set_objective_metadata`, when a label or unit actually changed | `old` and `new` |
| `control_requested`, `control_reported` | The [control channel](../guide/more/monitoring.md), when the value changed; heartbeats are not logged | `requested` or `reported`, `by`, and `detail` for a report |
| Any other string | `AuditLog.log_event`, or `POST .../audit/event` | Whatever you pass |

`lock_acquisition` adds one to the session's iteration counter and records the new number. `lock_model` records the number the next acquisition will take, so a model locked just before an acquisition shares its iteration. `suggest_next` locks nothing in Python; the REST route that suggests locks the acquisition itself, so a web session has an entry per suggestion.

### Lock a decision and read it back

```python
def announce(data):
    entry = data["entry"]
    print("locked", entry["entry_type"], entry["hash"])

session.on("acquisition_locked", announce)

session.lock_data(notes="Initial design and first results")
session.lock_model(notes="Default Matern kernel")
session.lock_acquisition(
    strategy="EI",
    parameters={"goal": "maximize"},
    suggestions=suggestion.to_dict("records"),
    notes="Round 1",
)

log = session.audit_log
print(len(log), [entry.entry_type for entry in log.get_entries()])
latest = log.get_latest("acquisition_locked")
print(latest.parameters["iteration"], latest.parameters["suggestions"][0].keys())
```

`get_entries(entry_type=None)` returns `AuditEntry` objects, all or one type, and `get_latest(entry_type)` the most recent or `None`. `session.get_audit_log()` returns the whole log as a dictionary, which is what the session file stores. `lock_model` raises a `ValueError` without a trained model.

The timestamp and notes are not hashed. The hash shows that an entry has not changed since it was written, and you can recompute it by hashing the entry's `parameters` the same way. It is a checksum and not a signature: anyone who can edit the session file can edit an entry and its hash together.

### Add your own entries

`log_event(entry_type, parameters, notes="")` writes an entry of any type, for a fact the campaign depends on that ALchemist does not know, such as a catalyst batch. The type is any string, up to 64 characters over REST, and ALchemist never interprets it.

```python
log.log_event(
    "sample_prepared",
    {"batch": "B-17", "furnace_ramp_C_per_min": 5},
    notes="From the lab notebook",
)
print(log.get_latest("sample_prepared").parameters)
```

The entry is saved with the session and returned by `GET .../audit`. Over REST the same call is:

```http
POST /api/v1/sessions/{session_id}/audit/event
Content-Type: application/json

{"entry_type": "sample_prepared", "parameters": {"batch": "B-17"}, "notes": "From the lab notebook"}
```

## Export the audit log as Markdown

`session.export_audit_markdown()` returns the session's metadata and the log together as one Markdown document for a methods section; `AuditLog.to_markdown(session_metadata=None)` underneath leaves the metadata part out unless you pass it. Write the result to a file next to the session file.

```python
with open("campaign_audit_log.md", "w") as handle:
    handle.write(session.export_audit_markdown())
print(session.export_audit_markdown().splitlines()[0])
```

The parts of the document are described under [Sessions](../guide/more/sessions.md#3-export-the-audit-log) and shown in [File formats](../reference/file-formats.md#audit-log-markdown).

The iteration blocks are built from the `data_locked`, `model_locked` and `acquisition_locked` entries. A `data_locked` entry with no design metadata has no block of its own; the experiment table shows the data it locked. Entries of other types, your own included, are in the session file and `GET .../audit` and print no content in the Markdown.

!!! warning "Known issue"
    W23: the search-space part lists a discrete variable as `(categorical): {}`, with its values missing; the values are correct in the session file and the data table. Also, every entry whose `parameters` carry an `iteration`, `config_changed` (a running count) and `experiment_completed` included, opens an iteration group, so the Markdown can show empty "Iteration N" headings.

## Over REST

The audit routes, which list entries, lock a decision, add an event, and export the Markdown, are described in the [REST API reference](../reference/rest-api.md) under Sessions, and `GET .../audit/config-changes` is listed under Experiments.

!!! info "How ALchemist does it"
    The session creates one `EventEmitter` and hands it to its work queue, so queue events and session events arrive on the same object. `AuditLog` in `alchemist_core/audit_log.py` appends `AuditEntry` objects and makes no other change to them; the `lock_*` methods on the session add the search-space snapshot and the iteration number before they call it.
