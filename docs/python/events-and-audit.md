# Events and audit log

A session announces what it does through events, and it records the decisions behind a campaign in an audit log. Events are for your code to react to. The audit log is for people to read afterwards, as a methods section or a record of what was run and why.

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
for name, keys in heard[2:4] + heard[-2:]:
    print(name, keys)
```

`session.on` is shorthand for `session.events.on`; both register on the session's `EventEmitter`. The emitter has a few more calls.

| Call | Does |
|---|---|
| `on(name, callback)` | Adds a listener. Adding the same callback twice for one name has no effect. |
| `once(name, callback)` | Adds a listener that removes itself after its first call. |
| `off(name, callback)` | Removes a listener. |
| `remove_all_listeners(name=None)` | Removes the listeners for one name, or for every name. |
| `listener_count(name)`, `event_names()` | How many listeners a name has, and the names that have any. |

Listeners run synchronously, on the thread that made the call, in the order they were added. A listener that raises is logged and skipped: the session call carries on and the other listeners still run. A name nothing emits is accepted without complaint, so a misspelt name just never fires; check it against the table below.

```python
session.events.once("session_saved", lambda data: print("saved", data["filepath"]))
session.save_session("campaign.json")
session.save_session("campaign_copy.json")
print(session.events.listener_count("session_saved"))
```

### The events a session emits

| Event | When | Payload keys |
|---|---|---|
| `variable_added` | `add_variable` | `name`, `type` |
| `derived_variable_added` | `add_derived_variable` | `name`, `input_cols` |
| `search_space_loaded` | `load_search_space` | `filepath` |
| `data_loaded` | `load_data` | `n_experiments`, `filepath` |
| `experiment_added` | `add_experiment`, and each recorded queue item | `inputs`, `output` |
| `experiment_staged` | `add_staged_experiment` | `inputs` |
| `staged_experiments_cleared` | `clear_staged_experiments`, `move_staged_to_experiments`, when it removed any | `count` |
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

`complete_experiment` adds a row, so it emits `experiment_added` as well as the two queue events. A failed training emits `training_started` and no `training_completed`; call `train_model` inside a `try` to learn why. Nothing emits `model_trained` or `acquisition_generated`; use the `training_` and `acquisition_` pairs above.

`model_retrained` and `model_retrain_failed` are emitted on the session that `load_session` builds internally, not on yours, so a listener you registered never receives them. Of the load events, `session_loaded` is the only one you can hear.

The REST server runs sessions in the same process, so these events fire there too, but nothing forwards them. Its WebSocket sends its own messages, such as `experiments_updated`, which are not these events.

### Share one emitter

Pass `event_emitter=` to give several sessions the same emitter, for example to log them all in one place. The work queue always uses its session's emitter.

```python
from alchemist_core.events import EventEmitter

bus = EventEmitter()
bus.on("training_completed", lambda data: print("trained", data["backend"]))

sessions = [OptimizationSession(event_emitter=bus) for _ in range(2)]
for s in sessions:
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    for i in range(5):
        s.add_experiment({"x": i / 4}, output=float(i % 3))
    s.train_model(backend="sklearn")
```

## The audit log

The audit log holds the decisions a campaign was built on. Exploring writes nothing to it: you can plot, predict and ask for suggestions as often as you like. An entry appears when something is committed, which keeps the log short enough to read. Nothing in ALchemist edits or removes an entry once it is written, except `AuditLog.clear()`, which exists to start a new campaign in the same session.

Each entry has five fields.

| Field | Content |
|---|---|
| `timestamp` | When the entry was made, as an ISO string. |
| `entry_type` | What kind of decision it records; see below. |
| `parameters` | The details, as a dictionary. This is the part that differs by type. |
| `hash` | The first 16 hexadecimal characters of the SHA-256 of `parameters`, serialized as JSON with sorted keys. |
| `notes` | Free text that you or the calling code supplied. |

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

`lock_acquisition` adds one to the session's iteration counter and records the new number, so each acquisition is one iteration. `lock_model` records the number the next acquisition will take, so a model locked just before an acquisition shares its iteration. `session.suggest_next` locks nothing in Python. The REST route that suggests does lock the acquisition itself, and a web session therefore has an entry per suggestion without anyone locking it.

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
print(log.get_entries("model_locked")[0].parameters["cv_metrics"].keys())
```

`get_entries(entry_type=None)` returns a list of `AuditEntry` objects, all of them or one type. `get_latest(entry_type)` returns the most recent, or `None`. `session.get_audit_log()` returns the whole log as a dictionary, with `entries`, `search_space` and `experiment_data`, which is also what the session file stores. `lock_model` needs a trained model and raises a `ValueError` without one.

The hash covers `parameters` only, not the timestamp or notes. It shows that an entry has not changed since it was written, and you can recompute it. It is a checksum and not a signature: anyone who can edit the session file can edit an entry and its hash together.

```python
import hashlib
import json

def recompute(entry):
    text = json.dumps(entry.parameters, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]

print(all(recompute(entry) == entry.hash for entry in log.get_entries()))
```

### Add your own entries

`log_event(entry_type, parameters, notes="")` writes an entry of any type, for a fact the campaign depends on that ALchemist does not know, such as a catalyst batch or a furnace ramp rate. The type is an opaque string, up to 64 characters over REST, and ALchemist never interprets it.

```python
log.log_event(
    "sample_prepared",
    {"batch": "B-17", "furnace_ramp_C_per_min": 5},
    notes="From the lab notebook",
)
print(log.get_latest("sample_prepared").parameters)
```

The entry is saved with the session and returned by `GET .../audit`, and it does not appear in the Markdown export. Over REST the same call is:

```http
POST /api/v1/sessions/{session_id}/audit/event
Content-Type: application/json

{"entry_type": "sample_prepared", "parameters": {"batch": "B-17"}, "notes": "From the lab notebook"}
```

## Export the audit log as Markdown

`session.export_audit_markdown()` returns the log as Markdown for a methods section. Write it to a file next to the session file.

```python
with open("campaign_audit_log.md", "w") as handle:
    handle.write(session.export_audit_markdown())
print(session.export_audit_markdown().splitlines()[0])
```

The document has four parts: session metadata, the search space, the experiment table, and one block per iteration. A block holds the model's backend, kernel, scaling and cross-validation metrics, the acquisition strategy and parameters with the suggested points, and the initial design, each with its notes. [Sessions](../guide/more/sessions.md#3-export-the-audit-log) describes the parts from the web app's side, and [File formats](../reference/file-formats.md#audit-log-markdown) shows an example.

The export keeps only the `data_locked`, `model_locked` and `acquisition_locked` entries. A `data_locked` entry with no design metadata has no block of its own; the experiment table shows the data it locked. Configuration changes, completed experiments, label changes, control entries and your own entries are in the session file and `GET .../audit` but not in the Markdown.

!!! warning "Known issue"
    W23: the search-space part lists a discrete variable as `(categorical): {}`, with its values missing. The values are correct in the session file and the data table.

`AuditLog.to_markdown(session_metadata=None)` is the method underneath; the session method passes the session's metadata for the first part, and the log alone leaves it out.

## Over REST

| Request | Returns |
|---|---|
| `GET /api/v1/sessions/{session_id}/audit` | The entries as JSON, with `n_entries`; add `?entry_type=model_locked` to filter. |
| `POST /api/v1/sessions/{session_id}/audit/lock` | Locks a decision. The body has `lock_type` (`data`, `model` or `acquisition`) and `notes`; an acquisition lock also needs `strategy`, `parameters` and `suggestions`. |
| `POST /api/v1/sessions/{session_id}/audit/event` | Adds an entry of any type, as above. |
| `GET /api/v1/sessions/{session_id}/audit/export` | The Markdown document, as a file named `audit_log_<session ID>.md`. |
| `GET /api/v1/sessions/{session_id}/audit/config-changes` | The `config_changed` entries as `changes`, each with `timestamp`, `component`, `old`, `new` and `iteration`. |

!!! info "How ALchemist does it"
    `EventEmitter` in `alchemist_core/events.py` keeps a list of callbacks per name and calls them in a loop, which is why a listener runs before the session call returns. The session creates one emitter and hands it to its work queue, so queue events and session events arrive on the same object. `AuditLog` in `alchemist_core/audit_log.py` appends `AuditEntry` objects and makes no other change to them; the `lock_*` methods on the session add the search-space snapshot and the iteration number before they call it.
