# Changelog

All notable changes to ALchemist are documented here.

---

## [Unreleased]

---

## [0.4.0] — 2026-10-15

0.4.0 makes designs of experiments respect your constraints and your hardware. Linear input constraints now work in every design method, hard-to-change variables and split-plot designs are supported, and an experiment work queue records what was suggested against what was actually run. A live monitor follows a running campaign, and the documentation moves to a new Zensical site. Sessions autosave after every change, load lazily, and rebuild their model on reload, alongside many correctness fixes. The desktop app is deprecated and now installs only through the `[desktop]` extra.

Work surfaced while deploying ALchemist on an inductive RCC campaign (real
non-`Output` objective column, batch qEI acquisitions, hardware-adjusted
conditions), and a pass making linear input constraints work everywhere they
were already advertised.

### Breaking Changes

All four come from this work. The first three are direct consequences of the
constrained-DoE feature; the fourth is a side effect of hardening the shared
variable-registration path the feature depends on. **Unconstrained behavior is
unchanged at every seed**, locked by golden tests over every DoE method.

- A constrained **classical** design (`ccd`, `box_behnken`, `full_factorial`,
  `fractional_factorial`, `plackett_burman`, `gsd`) that loses structural
  points now raises `DesignNotEstimableError` when the surviving points can no
  longer estimate the design's implied model. It previously returned the
  degraded remnant with a log warning — a rank-deficient design that still
  looked like the design you asked for. The message names the terms that
  became inestimable. Pass `allow_infeasible=True` for the old behavior —
  in Python, and as a request field on `POST /initial-design`, where the
  response then reports `"estimability": "waived"`.
  Designs that lose only harmless points (a replicated center point) still
  return normally. Over REST this is a `400` with
  `"error_type": "DesignNotEstimableError"`.
- A constrained **optimal** design returns **different points for the same
  seed**, because it now selects from a candidate set that is filtered *and*
  augmented on the feasible boundary rather than from a plain filtered
  lattice. This is the fix, not a regression.
- `SearchSpace.add_constraint` / `session.add_input_constraint` now **reject
  non-numeric variables** (`categorical`, `context`) at registration, along
  with non-finite `rhs` and coefficients. Such constraints were already
  non-functional — they failed later, inside feasibility filtering.
- **Variable bounds are now validated, so documents that used to load are now
  refused.** `SearchSpace.add_variable` — and therefore `POST /variables`,
  `PUT /variables/{name}` and `/variables/load` — reject a non-finite `min` or
  `max` (`inf`, `NaN`) and a `real` span that is not float64-representable
  even though each endpoint is. None of these guards existed before; all three
  values were previously accepted and produced a variable that was unusable
  downstream (a `NaN` bound broke every subsequent export; an overflowing span
  sampled to a single distinct value). The guards live on the shared core
  path, so a stored search-space **or session** file carrying one of these
  values no longer loads **anywhere** — `SearchSpace.load_from_json` (the
  desktop loader), `OptimizationSession.load_session`, and every REST route
  all refuse it. A file written by an earlier version is only affected if it
  actually carries such a bound.
- **`suggest_next` refuses a batch from a one-point acquisition (A1).** `EI`,
  `LogEI`, `PI`, `LogPI` and `UCB` with `n_suggestions > 1` now raise
  `ValueError` (REST 400) instead of crashing inside BoTorch. The scikit-learn
  backend now rejects any `n_suggestions > 1`, `gp_hedge` included, instead of
  silently returning one row. For a batch, use BoTorch with `qEI`, `qUCB` or
  `qIPV`.
- **`suggest_next` refuses unknown keyword arguments (A5).** Arguments such as
  `n_candidates` now raise `TypeError` instead of being dropped with a warning.
- The desktop app is deprecated and no longer installed by default. Install it with `pip install "alchemist-nrel[desktop]"`; `alchemist` prints a deprecation notice on start and an install hint if the extra is missing. The web app (`alchemist-web`) is the supported interface. The extra carries `customtkinter`, `tksheet`, `ctkmessagebox`, `mplcursors` and `tabulate`, which only `ui/` imports.

### New Features
- **Suggested-vs-actual provenance.** Every experiment now records what the model
  suggested versus what was actually run, durable in the session file and audit
  log. Each dataset row carries a hidden `ProvenanceId` joining it to a full
  provenance record (suggested inputs, actual inputs, per-variable delta,
  strategy, acquisition params, output, noise, timestamp). Closes the loophole
  where actuals could deviate arbitrarily from a suggestion while still being
  labeled with an acquisition strategy. The web "Add Point" flow now routes
  through the work-queue `complete()` lifecycle; new endpoints
  `GET /experiments/provenance` and `/{id}` expose the records. `ProvenanceId`
  is excluded from the model input matrix at every site.
- **Read-only suggested + editable "actual" conditions in the Add Point dialog.**
  Suggested values are shown read-only (rounded per variable — integers to whole
  numbers, reals to 3 decimals, raw precision in a tooltip) alongside editable
  "actual" boxes pre-filled with the rounded suggestion. The recorded data point
  uses the actual (hardware-adjusted) values.
- **Acquisition suggestions now carry their iteration number** end to end, so the
  Add Point dialog shows the real iteration instead of `N/A`; a whole batch is
  recorded under one iteration.
- **Linear input constraints are now settable over REST.** `POST`/`GET`
  `/sessions/{id}/constraints` and `DELETE .../{name}`. Constraint names are
  unique and are the delete identity, so one call removes exactly one
  constraint. Previously constraints could only be set from Python, so no REST
  or web consumer could use the feature at all. New documentation page:
  *Constraining the Variable Space*.
- **Design responses carry a `feasibility` block.** `/initial-design` and
  `/optimal-design` report which constraints applied, how many structural
  points a classical design dropped, the optimal-design candidate accounting
  (candidates total, feasible, boundary and vertex points added), and an
  `estimability` verdict. `null` when no constraints are registered. This
  provenance previously existed only in a server log line.
- **`/variables/load` accepts the `{variables, constraints}` document** that
  `SearchSpace.save_to_json` writes, alongside the bare variable list. The dict
  form is validated atomically — a bad variable or constraint anywhere in the
  file rejects the whole document and leaves the session untouched — and
  loaded constraints pass the same validation as `POST /constraints`.
- **`/variables/export?include_constraints=true`** returns that same document,
  so `load → export → load` round-trips constraints. The default export stays a
  bare variable array, which is an existing cross-surface contract.
- **The web app links to the user guide.** A "User guide" link after the
  tagline in the header opens the documentation site in a new tab.
- **The web app saves to a file you choose and updates it in place** (Chrome
  and Edge, on https or localhost). The first Save asks where; later Saves
  (⌘S / Ctrl+S) rewrite that file, and Save As… (⇧⌘S / Ctrl+Shift+S) picks a
  new one. A session opened with Load Session saves back to the file it came
  from. The header names the linked file and shows a dot when there are unsaved
  changes. The variables file gets the same Load File / Save File linking; the
  audit log and DoE/OED CSVs open the Save dialog. Other browsers, and plain-http
  addresses such as `http://fscl2:8200`, keep downloading as before. The
  leave-page prompt now appears only when there are unsaved changes. No REST
  endpoint changed.
- **The contour plot's fixed values can be typed and stepped.** Each fixed real
  variable has a number box below its slider: type a value and press Enter (or
  leave the box) to set it, clamped to the variable's range, or press ↑ and ↓ to
  step it as the slider does. The slider's arrow keys update the plot too, and
  the plot is requested once, 150 ms after the last change. The previous contour
  stays on screen, dimmed, while the next one loads.
- **The contour plot keeps one colour scale across slices.** By default the
  colour bar now spans the observed outputs (the min and max of the experiments'
  output), so moving a fixed value no longer rescales it and two slices can be
  compared by colour: a slice that predicts lower values simply looks darker. A
  prediction outside the observed range takes the end colour; hover still shows
  its value. A Scale control under Colormap switches to **Per slice**, the old
  behaviour, which fits the colours to each slice's own min and max to show the
  structure within it. The browser remembers the choice. The contour response
  gains an `output_range` field (`null` when the outputs have no span).
- **The web app reopens the session you were working on (W30).** After a reload, or
  after quitting and relaunching the server or the desktop app, it opens the
  session this browser last created, loaded, restored or joined. The session
  comes back with its current data and still linked to its file, so the next
  Save writes there (Chrome asks once more for permission to write after a
  reload). `?session=` in the URL still takes precedence. A session the
  server no longer has is forgotten without an error. The remembered session
  is per browser, so several browsers sharing one server each reopen their
  own. The unsaved-changes dot now also survives a relaunch.

### Bug Fixes
- **A multi-objective session no longer loses its other objectives on load (NEW-1).**
  `load_session` rebuilt the results one row at a time with the first objective
  only, so a 2-objective file came back with the second objective empty on
  every row, the load-time retrain failed, and saving again wrote the loss to
  disk. Every objective is restored now, and the retrain and the next save
  keep them all.
- **`add_experiment` takes every objective on a multi-objective session (A18).**
  Pass a dict `{objective: value}` or a sequence in target order. A bare float,
  which used to fill the first objective and leave the rest empty with no
  error, is now refused with the missing objectives named.
- **Results added after `load_search_space` train again (A28).** The loaded
  search space was not passed on to the results table, so rows added with
  `add_experiment` afterwards had no inputs as far as training was concerned
  and `train_model` failed with "Found array with 0 feature(s)" or a
  state-dict error.
- **Outcome constraints survive save and load (A29).** `save_session` never
  wrote them, so a reloaded session suggested unconstrained. The session file
  now carries an `outcome_constraints` key and `load_session` restores it.
- **Deleting a variable deletes the constraints that name it (A14).** They
  used to stay, and since feasibility sums only the columns present,
  `x1 + x2 <= 12` was silently enforced as `x1 <= 12` after `x2` was gone. The
  dropped constraints are named in a log warning. The REST route and the web
  app's variable deletion get the same behaviour.
- **`add_experiment` refuses a key that is not a variable (A4).** An unknown
  key used to become a column in the data with no error; the REST experiment
  endpoints now answer 400. A value outside a real or integer variable's bounds
  is still recorded, since observed variables and hardware-adjusted actuals
  may legitimately sit outside, but it is logged as a warning.
- **Session files are strict JSON (W13).** An empty cell, such as a result
  with no `ProvenanceId` or no `Noise`, was written as a bare `NaN`, which
  `JSON.parse` and other strict parsers refuse. Empty cells are `null` now,
  the writer refuses to emit `NaN`, and the loader reads `null` back as an
  empty cell.
- **scikit-learn output scaling works for every listed transform, and the
  noise follows it (T17, T18).** `minmax` and `robust` built no output scaler,
  so training ran unscaled while cross-validation scaled; both now build the
  scaler. With a Noise column and an output transform, the noise variance was
  passed to the GP in the output's raw units while `y` was scaled, so a noise
  of 4.0 on an output spanning thousands was weighted 5e4 times too heavily.
  The variance is now carried through the same scaling as `y`.
- **`halton` is a Halton sequence (T12).** It ran the Hammersly sampler and
  gave the same points as `hammersly`; it now runs scikit-optimize's `Halton`.
  The golden design fixture's `halton` entry was re-locked for this. A
  circumscribed CCD still returns the inscribed design and `detmax` /
  `modified_fedorov` still run the Fedorov exchange (T9, T10); both now log a
  warning saying so, and the docs say it plainly.
- **`move_staged_to_experiments(iteration=)` takes effect (A3).** The argument
  was accepted and ignored, so a batch recorded through it, or through
  `POST .../staged/complete`, took the next free iteration numbers one each.
- **BoTorch `input_transform_type="standardize"` standardizes (T6).** It built
  `Normalize`, which rescales to [0, 1]; it now builds `InputStandardize`.
- **`kappa` reaches UCB and qUCB on BoTorch (T13).** The REST API and the web
  app send the exploration weight as `kappa`; BoTorch's UCB reads `beta`, so
  the value was dropped and both ran at β = 0.5. `kappa` is now taken as
  `beta` when `beta` is not given. Note that the REST `kappa` default is 2.0,
  so BoTorch UCB and qUCB over REST now run at β = 2 unless the request sets
  it.
- **Batch acquisitions now honour Minimize (T7).** `qEI` and `qUCB` were built
  without a direction and always looked for high values, so a minimisation
  campaign using a batch got the worst points. They now negate the objective
  for `goal="minimize"`. EI and LogEI with an outcome constraint go through the
  same directed path, so they minimize correctly too, and EI no longer ignores
  the constraint (T20). The incumbent that improvement is measured against is
  now the best *feasible* training point when outcome constraints are set.
- **Open plots follow the header theme toggle.** The contour plot kept the theme it mounted with, and the parity, metrics, Q-Q, calibration and objective-trace plots drew light grids, white points and light tooltips in the dark theme. All of them now redraw when you switch.
- **The parity plot's "Predictions" legend text was invisible in the light theme.**
- **The desktop app no longer silently drops data when it syncs a session made in the web app or by the work queue (D1, D2).** If the sync would lose observed or hard-to-change variables, derived variables, input constraints, work-queue provenance or split-plot columns, or rows that have a result but an empty or non-numeric input, it refuses, lists what would be lost, and tells you to open the session in the web app. Nothing is changed, and Save, Save As, Train Model, the audit-log lock and the variable dialog stop instead of carrying on. Rows with no result yet are still dropped as before, and sessions built in the desktop app itself sync as they did.
- **Add Point works with nothing staged (W28).** It opens a blank point to type
  an experiment's conditions and result by hand, for a first result or an
  experiment ALchemist did not suggest. With points already staged, an "Add
  blank point" control in the dialog adds a hand-typed point alongside them.
- **Joining or loading a session no longer brings back already-recorded points as pending (W26).** The audit-log fallback now skips any suggestion that matches a row in the experiment table, comparing reals after the 3-decimal rounding the Add Point dialog records, integers as whole numbers, and categorical or discrete values as given. A point recorded with an edited actual value is also skipped, matched through its provenance record's suggested conditions.
- **Session load dropped all data when the objective column was not named
  `Output`.** A long-standing bug (present since ≤ v0.3.3): the loader hardcoded
  the target column, so any session whose objective was named something else
  (e.g. `Methane umol/g/Wh`) silently restored 0 experiments (rows were swallowed
  by a per-row `try/except`). The loader now resolves the target from a persisted
  `target_columns` key or infers it from the non-variable/non-metadata columns;
  `save_session` now records `target_columns` for self-describing files. Fully
  backward compatible across versions.
- **Batch acquisition points appeared identical in the Add Point dialog.** The
  dialog seeded form state from props at mount only, and stepping through a batch
  swapped the prop without remounting, so every point showed the first point's
  values. Fixed by keying the dialog per suggestion so it remounts with fresh
  state. (The underlying acquisition and staging were always correct — this was
  a display bug.)
- **Staged suggestions persisted across sessions ("ghost points").** The staged-
  suggestions restore effect only ever *added* to state and never cleared, so
  starting a new session left stale suggestions in the UI. It now always resolves
  to a definitive value (clearing when the session has none), with a race guard.
- **"Retrain model" was silently ignored when recording results via the work
  queue.** The queue-complete endpoint had no auto-train; it now honors the
  retrain control like the direct add-experiment path.
- **Regret / hypervolume plots showed gaps where a subset GP failed to fit.**
  The posterior overlay refits a fresh GP on each data prefix; small or
  ill-conditioned prefixes (notably categorical/`AdditiveKernel` models, where
  the hyperparameter-reuse fast path is skipped and every iteration
  re-optimizes) could raise `ModelFittingError`, leaving that iteration's
  prediction as NaN. Both the single-objective and MOBO paths now walk a
  fallback ladder — requested transforms, then transforms disabled, then
  escalating Cholesky jitter (1e-3, 1e-2) — and every value plotted remains a
  genuine prediction from an actually-fitted GP, never interpolated or
  fabricated. If the whole ladder fails, NaN is still the last resort.
- **Linear input constraints were honored in only two of four places.** They
  were respected by acquisition and by space-filling DoE, but **classical**
  designs generated their points ignoring constraints and post-hoc-dropped the
  infeasible ones with a log warning, leaving a design rank-deficient for the
  model it claimed to fit; and **optimal** designs had no constraint awareness
  at all, so the exchange algorithm spent its whole budget selecting points it
  would not be allowed to keep. Optimal designs now select from a candidate set
  that is filtered *and* augmented with points on the feasible region's
  boundary — a filtered lattice has none, and an optimal design wants precisely
  those extremes — so a constrained D/A/I-optimal design is now genuinely
  optimal over its region. An equality constraint over `real` variables is now
  satisfiable for the first time — optimal design is the only method that
  places points on the zero-volume slice such a constraint defines. Note the
  model must not contain the terms the equality makes collinear: with
  `x1 + x2 == rhs`, the intercept and the two main effects are exactly
  dependent, so pass an `effects` list that drops one of the tied variables
  rather than a `model_type` shortcut. The rank-deficiency error names this
  case and its remedy.
- **Non-model variables were spread straight through a constraint.** Variables
  absent from every model term were overwritten with a shuffled range *after*
  selection, undoing all feasibility work for those columns. They are now drawn
  from each row's feasible interval.
- **A `context` variable silently dropped a real variable from every design.**
  Design generation zipped the full variable list against samples drawn from
  dimensions that exclude `context`, so a context variable anywhere but last
  produced points missing an optimization variable, with its values relabeled
  onto the context column — no error, straight into the experiment table, the
  model and the acquisition function. The classical and optimal paths were
  affected more severely than space-filling. The desktop design table hit the
  same skew.
- **Editing or deleting a variable could silently corrupt the search space.**
  `PUT`/`DELETE` on a variable rebuilt dimensions by an index derived from a
  list that includes `context` entries, applied against one that does not. In
  range, `PUT` returned 200 and destroyed a different variable's dimension —
  invisible through the API's own read path, since `GET /variables` reads the
  other list. Both routes now go through one validated core path that replaces
  the dimension in place, so ordering is preserved — which is also how `PUT`
  picks up the new bound guards listed under Breaking Changes. Before this
  work no route validated bounds at all, and `PUT` built its dimension inline,
  so it would have bypassed the guards even once they existed.
- **Space-filling designs returned values that were not JSON-serializable.** An
  `integer` variable came back as `np.int64`, which `json.dumps` refuses, so the
  endpoint failed outright; `discrete` variables had been leaking `np.float64`
  silently for as long. Every variable type now returns a native scalar.
- **A provably infeasible constrained design ground for minutes and blocked the
  whole server.** The reject-and-resample loop escalated its batch size to
  4096× before giving up — measured at 480 s to a correct 400 — and the design
  endpoints ran that work directly on the ASGI event loop, so one such request
  starved every other request in the process, not just its own. Impossible
  regions are now proven empty up front and refused immediately with
  `InfeasibleRegionError` (`400`), and design generation runs in a threadpool
  like the other heavy endpoints.
- **`random_seed` was applied through process-global state.** Two concurrent
  seeded design requests interleaved and neither got the design its seed names.
  The seed is now honored per call.
- **Constraints were lost on server restart.** `SearchSpace.save_to_json`
  persisted them, but the *session* serializer did not, so every API session —
  and every crash-recovery backup — silently dropped its constraints on reload.
- **Constraint errors were indistinguishable from any other bad request.** Both
  are `ValueError` subclasses and so already returned `400`, but under
  `"error_type": "ValueError"`, alongside malformed bounds and unknown methods.
  They now report `DesignNotEstimableError` and `InfeasibleRegionError`, which
  carry different remedies.
- **Web notifications covered the session buttons and could not be dismissed.**
  They now appear bottom-right with a close button, and errors and warnings
  stay up for 10 s instead of 4 s so a long API message can be read. They also
  follow the app's light/dark theme instead of always rendering light.
- **Files saved from the web app are named after the session.** A session saves
  as `<title>.json`, with `<title>_variables.json` and `<title>_audit_log.md`
  alongside, instead of a fragment of the session id. Characters Windows or
  macOS forbid in filenames become `_`; an untitled session still falls back to
  its id. The REST endpoints are unchanged.
- **The web app's Edit Info dialog opened empty.** It read metadata from a
  session response that does not carry it; it now loads the session's current
  name, description and tags.
- **Opening `http://localhost:8000` showed JSON instead of the web app.**
  `alchemist-web` prints that address, but `/` always returned the API-info
  JSON. A browser now gets the web app there; scripts and other clients that
  don't ask for HTML still get the JSON, which also has its own route at
  `/api`. `/health` is unchanged.
- **Training a multi-objective model in the web app blanked the whole page
  (W25).** The model reports no cross-validation metrics for more than one
  objective, and the UI formatted the missing values. The Model panel now says
  how many objectives were trained and that their metrics aren't computed yet.
  Each panel also contains its own errors, so a crash in one shows a message
  there while the header (Save included) keeps working.
- **`session.load_session(path)` on an existing session now replaces all of its state and keeps the provenance (A31).** The instance form copied everything from the file except `provenance`, so `get_provenance()` was empty afterwards and the next `save_session` wrote `provenance: []` while the rows kept their provenance IDs. The class form and REST import were not affected.
- **The web app's variables Save File keeps input constraints (W35).** It asked for the bare variable list, so it dropped a session's constraints and, through a file linked by Load File, overwrote a `{variables, constraints}` file with the bare list. It now writes `{"variables": [...], "constraints": [...]}` when the session has constraints and the bare list otherwise.
- **`OptimizationSession.load_session(path, retrain_on_load=False)` on the class honours the keyword (A16).** It was ignored and the model was retrained anyway. The server's own loaders (reloading a saved session, REST import and upload) deliberately retrain; see Changed.
- Training on data where only some experiments carry a noise value now raises a clear `ValueError` (REST 400) naming the rows, instead of a NaN error from BoTorch or a "not positive definite" error from scikit-learn; an empty Noise column trains as no noise (A6).
- BoTorch models trained through the REST API or the web app without choosing transforms now normalize inputs and standardize outputs, as the docs describe; previously the web app trained unscaled, which can give an R² near zero on data with large ranges. Choosing `none` explicitly still trains unscaled (W2).
- **The recovery banner kept offering the same old backup.** Restoring a
  backup made a new session but left the original's backup in place, and
  saving the new session never cleared it, so every reload offered the old
  work again as an unsaved copy. A restore now hands the backup to the new
  session, and saving that session clears it. The banner also skips the
  session being reopened. Backups are now taken only while a session has
  changes not yet saved to its file, so saved work is no longer offered back
  as "unsaved".

### Hard-to-change variables

Variables can now say how they behave in an experiment: whether you control
them or only measure them, and how hard they are to change between runs. Holds
and split-plot designs build on this.

- **Variable roles.** Every variable has a role, `controllable` (the default) or
  `observed`, separate from its kind. An observed variable is measured, never
  optimized or sampled, and it can be categorical. `type: "context"` still works
  and means an observed real.
- **Change difficulty.** A controllable variable has a change difficulty:
  `easy` (the default), `hard` or `very_hard`.
- **Holds.** `suggest_next(hold=..., observed=...)` fixes variables for one call
  through BoTorch's `fixed_features`. `context=` remains an alias of `observed=`.
  A hold that leaves no feasible point raises `InfeasibleRegionError`. With
  anything held, the EI-family baselines (EI, LogEI, PI, LogPI, qEI) are the best
  value reachable in the held slice.
- **Knowledge ratio.** Each suggestion reports a knowledge ratio (0 where the data
  pins the function, 1 where the model is at its prior) in
  `session.last_suggestion_diagnostics`. A warning is added above 0.9 when
  something is held or observed.
- **Observed variables are never optimized or gridded.** `find_optimum(observed=...)`,
  prediction grids and plot slices condition on the supplied value, else on the
  most recent experiment's value.
- **Split-plot designs.** `generate_split_plot_design(...)` nests the existing DoE
  methods by change level (`very_hard`, then `hard`, then `easy`). It supports
  replicates, fresh or crossed inner designs, reference runs, run order and
  per-plot constraint reduction, and writes `WholePlot`, `Subplot` and `RunOrder`
  columns. Classical methods default to no center settings inside split-plot
  designs.
- **REST.** New `POST /sessions/{id}/split-plot-design`. Suggest gains `hold`,
  `observed` and `diagnostics`; find-optimum gains `observed`; variables gain
  `role` and `changes`.
- **Web.** Role and change difficulty in the variable form; a split-plot mode in
  Initial Design; setup-change banners and observed inputs in Add Point; a Hold
  list with "Hold current setup" in Suggest; integer formatting for the
  bookkeeping columns.
- **Dependencies.** `botorch>=0.18.1`, `gpytorch>=1.15.2`, `torch>=2.4`.

#### Breaking changes

- Categorical labels are encoded by their declared position, and a training value
  outside the declared values now raises.
- `WholePlot`, `Subplot` and `RunOrder` are reserved variable names.
- Sessions that contain an observed categorical variable do not load in earlier
  versions.
- Staging a point requires only the controllable variables.
- A legacy `context` variable now follows the both-or-neither bounds rule, and
  `None` bounds are dropped.

### Changed

- **The contour plot's layout is tidier.** The plot area is never taller than it is wide, its legend sits above the plot, and Fixed Values comes first, directly under the figure. In a narrow column the calibration table wraps under its chart.
- **The server saves a session after every request that changes it.** It used to write a session only when it was created, imported, locked or unlocked, and on `POST /sessions/{session_id}/save`, so results, queue changes and trained models made in between were lost if the server stopped. Every successful `POST`, `PUT`, `PATCH` or `DELETE` under `/api/v1/sessions/{session_id}/` now saves the session, except routes that only read it. The response carries `X-ALchemist-Saved: true` or `false`; a failed save is logged and does not turn a successful response into an error.
- **The server loads a session the first time its ID is used, not all of them at startup.** A server started in a folder with many saved sessions starts without reading them.
- **A reloaded or imported session gets its model back.** The server's loaders (lazy load, `POST /sessions/import`, `POST /sessions/upload`) rebuild the model by repeating the session's last training call, transforms included, and fall back to backend, kernel and kernel parameters for older files. If the retrain fails it is logged and the session loads without a model; the training call stays in the file (`model_config` with `fitted: false`), so later saves keep it and the next reload tries it again.
- **Session files are written atomically.** The server writes a temporary file and renames it over the session's file, so a crash during a save leaves the previous file whole.
- **`POST /sessions/{session_id}/save` returns 500 when the write fails.** It already returned 404 for an unknown session; it used to report success after a failed write.
- **A session file the server cannot read answers 503, not 404.** A session whose file exists but is corrupt or unreadable is no longer reported as not found, so a controller does not take it for deleted. `DELETE` still removes it.
- **Deleting a session that the server cannot remove from disk answers 500.** A delete of a loaded session used to return 204 even when the file stayed (a Windows scanner holding it), and the session came back after a restart. The server retries a held file, then reports the failure.
- **A session ID that differs only in letter case no longer loads a second copy.** On macOS and Windows `<ID>.json` and `<id>.json` are the same file; the server now requires the exact name, and the ID pattern no longer accepts a trailing newline.
- **Recovery backups no longer rewrite the session file, and lock and unlock write it once instead of twice.** The web app posts a backup every 30 seconds.

### Maintenance / Internal
- De-duplicated three copies of the model-input metadata-exclusion logic into a
  single `ExperimentManager.metadata_columns()` helper; routed the BoTorch
  evaluation and Pareto feature-selection paths through it as well.
- `BoTorchModel` now carries a per-instance `cholesky_jitter` attribute
  (defaulting to the module value). Callers that need to escalate jitter for one
  ill-conditioned fit set it on their own instance instead of rebinding the
  `_CHOLESKY_JITTER` module global, which would leak across concurrent API
  requests and every other GPyTorch consumer in the process.
- Production web build (`tsc -b`) no longer fails on `*.test.ts(x)` in fresh
  checkouts (test files excluded from `tsconfig.app.json`); vitest still
  type-checks them.
- New `alchemist_core.utils.constrained_region` module: the feasible-region
  primitives (candidate filtering, boundary and vertex augmentation, emptiness
  and measure-zero proofs) and `InfeasibleRegionError`.
  `DesignNotEstimableError` lives in `alchemist_core.utils.doe`. Both subclass
  `ValueError`, so existing `except ValueError` handlers keep working.
- The test suite no longer writes session files into the checkout's `cache/`: the API tests point the session store at a temporary directory.

---

## [0.3.4] — 2026-06-02

### Maintenance
- **Removed the unused `ax-platform` dependency.** It was no longer referenced by any model or acquisition backend (it had been dropped as a backend option but never pruned). It was also the only package forcing `ipywidgets` and the JupyterLab widgets extension into the install, whose very long asset paths triggered install failures on Windows (the 260-character `MAX_PATH` limit). Removing it fixes that failure class and trims `ipywidgets`, `plotly`, `sympy`, and `pyre-extensions` from the dependency tree. The BoTorch backend is unaffected.

### Documentation
- Removed internal planning/spec notes that had been inadvertently published to the docs site, and added `exclude_docs` / `.gitignore` guards to keep dev notes off the public docs going forward.

---

## [0.3.3] — 2026-03-20

### New Features
- **Full DoE suite** — classical RSM (CCD, Box-Behnken, Full/Fractional Factorial), screening (Plackett-Burman, GSD), and optimal designs (D/A/I-optimal with five exchange algorithms)
- **AI-assisted effect selection** — LLM-powered model term suggestions via OpenAI or local Ollama; optional Edison Scientific literature search integration
- **Discrete variable type** — numerical variables restricted to a specific set of allowed values (e.g., SAR ratios); full BO and DoE integration
- **IBNN kernel for BoTorch** — Infinite-Width Bayesian Neural Network kernel alongside Matern and RBF
- **Multi-objective variable role management** — explicit variable/target/drop column assignment on CSV upload; multi-target support in web UI, desktop, and Python API
- **3D surface and uncertainty surface plots** via `create_surface_plot()` and `create_uncertainty_surface_plot()`

### Improvements
- `pyDOE` pinned to 0.9.5 for reproducibility
- D-efficiency computation corrected for degenerate information matrices
- LLM panel: Beta badge added to toggle button; fixes for hallucinated citations and Ollama base URL normalization

### Documentation
- New pages: Classical & Screening Designs, Optimal Design, AI-Assisted Effect Selection, Multi-Objective Optimization, Staged Experiments (Python API), 3D Surface Plots, DoE theory background
- Updated: Variable Space setup (Discrete type), BoTorch Backend (IBNN kernel), Home feature list

---

## [0.3.2] — 2026-02-05

### New Features
- **Visualization module** (`alchemist_core/visualization/`) — pure plotting functions usable in notebooks, scripts, and the web API without UI dependencies
- **Staged experiments API** — `/api/v1/sessions/{id}/experiments/staged` endpoints for autonomous reactor-in-the-loop workflows
- **WebSocket-based session events** — lock status, experiment additions, and model training pushed to the frontend in real time (replaces polling)
- **Session reconnection fix** — URL `?session=` parameter now takes priority over stale recovery backups
- **Join Session UI** — paste a session ID on the landing page to connect to a running session
- Target column selection on CSV upload; flexible multi-objective preparation

### Improvements
- `BoTorchModel` updated to float64 tensors for numerical stability
- Corrected PI computation and sklearn acquisition functions for maximization
- `load_session()` now supports both static and instance usage patterns

### Infrastructure
- Repository migrated from NREL to NatLabRockies GitHub organization

---

## [0.3.1] — 2025-12-12

### New Features
- **WebSocket-based session locking** — replaces 5-second HTTP polling with instant (< 100 ms) lock status events; auto-reconnect with 5-second retry
- **Session lock REST endpoints** — UUID token-based lock/unlock with force-unlock recovery
- **Comprehensive unit test suite** — OptimizationSession, SklearnModel, AuditLog, EventEmitter, and multi-client scenarios

### Bug Fixes
- Multi-client iteration tracking: ExperimentManager now calculates iteration as `max(existing) + 1` automatically
- BoTorch/sklearn backend switching: automatic transform type mapping to prevent scaling errors
- Sklearn GP numerical stability: clamp predicted std to ≥ 1e-6; validate calibration factors before applying

### Documentation
- Auto-generated Python API reference using mkdocstrings
- Reorganized navigation; corrected endpoint paths, parameter names, and variable type names throughout

---

## [0.3.0] — 2025-11-24

### New Features
- **Production-ready packaging** — pre-built web UI bundled in the Python wheel; `pip install alchemist-nrel` includes both desktop and web apps
- **Entry points** — `alchemist` (desktop GUI) and `alchemist-web` (React + FastAPI) installed as CLI commands
- **Docker support** — production Dockerfile with multi-stage build, Docker Compose configuration, health checks, and volume mounting
- **Custom build hooks** — automatic React UI compilation during `python -m build`

### Improvements
- Flexible CORS via `ALLOWED_ORIGINS` environment variable
- Static file serving checks `api/static/` (production) before `alchemist-web/dist/` (development)

---

[Unreleased]: https://github.com/NatLabRockies/ALchemist/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/NatLabRockies/ALchemist/compare/v0.3.4...v0.4.0
[0.3.4]: https://github.com/NatLabRockies/ALchemist/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/NatLabRockies/ALchemist/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/NatLabRockies/ALchemist/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/NatLabRockies/ALchemist/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/NatLabRockies/ALchemist/releases/tag/v0.3.0
