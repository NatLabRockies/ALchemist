# Changelog

All notable changes to ALchemist are documented here.

---

## [Unreleased]

---

## [0.4.0] — 2026-10-15

0.4.0 makes designs of experiments respect your constraints and your hardware. Linear input constraints now work in every design method, variables can be marked hard to change and used in split-plot designs, and an experiment work queue records what was suggested next to what was actually run. A live monitor follows a running campaign, and the documentation moves to a new Zensical site. Sessions autosave after every change, load lazily and rebuild their model on reload. Two security holes in the web server are closed: the app's catch-all route could serve any file the server user could read, and two recovery routes accepted a wildcard session ID. Batch acquisitions (qEI, qUCB) now honour Minimize. Many smaller fixes are listed below. The desktop app is deprecated and installs only through the `[desktop]` extra.

### Breaking Changes

The first four come from making constraints work everywhere. Unconstrained designs are unchanged for every seed; golden tests cover every method.

- A constrained classical design (`ccd`, `box_behnken`, `full_factorial`,
  `fractional_factorial`, `plackett_burman`, `gsd`) that loses structural
  points now raises `DesignNotEstimableError` when the surviving points can no
  longer estimate the design's implied model. It used to return the degraded
  remnant with a log warning: a rank-deficient design that still looked like
  the design you asked for. The message names the terms that became
  inestimable. Pass `allow_infeasible=True` for the old behaviour, in Python
  and as a request field on `POST /initial-design`, where the response then
  reports `"estimability": "waived"`. Designs that lose only harmless points
  (a replicated center point) still return normally. Over REST this is a `400`
  with `"error_type": "DesignNotEstimableError"`.
- A constrained optimal design returns different points for the same seed,
  because it now selects from a candidate set that is filtered and then
  augmented with points on the feasible boundary, instead of a plain filtered
  lattice.
- `SearchSpace.add_constraint` and `session.add_input_constraint` reject
  non-numeric variables (`categorical`, `context`) at registration, along with
  a non-finite `rhs` or coefficient. Such constraints never worked; they failed
  later, inside feasibility filtering.
- Variable bounds are validated, so some documents that used to load are now
  refused. `SearchSpace.add_variable`, and therefore `POST /variables`,
  `PUT /variables/{name}` and `/variables/load`, reject a non-finite `min` or
  `max` (`inf`, `NaN`) and a `real` span that is not representable as a
  float64 even though each bound is. These values were accepted before and
  produced a variable that was unusable downstream (a `NaN` bound broke every
  export; an overflowing span sampled to a single value). The guards sit on
  the shared core path, so a stored search-space or session file carrying such
  a value no longer loads anywhere: `SearchSpace.load_from_json`,
  `OptimizationSession.load_session` and every REST route refuse it. A file
  written by an earlier version is affected only if it carries such a bound.
- `suggest_next` refuses a batch from a one-point acquisition (A1). `EI`,
  `LogEI`, `PI`, `LogPI` and `UCB` with `n_suggestions > 1` raise `ValueError`
  (REST 400) instead of crashing inside BoTorch. The scikit-learn backend
  rejects any `n_suggestions > 1`, `gp_hedge` included, instead of returning
  one row. For a batch, use BoTorch with `qEI`, `qUCB` or `qIPV`.
- `suggest_next` refuses unknown keyword arguments (A5). Arguments such as
  `n_candidates` raise `TypeError` instead of being dropped with a warning.
- `add_experiment` on a multi-objective session needs every objective (A18):
  a dict `{objective: value}` or a sequence in target order. A bare float,
  which used to fill the first objective and leave the rest empty, is refused
  with the missing objectives named.
- `add_variable` refuses a new variable once results are recorded, since the
  rows would have no value for it. Add every variable before recording
  results, or start a new session. A variable whose column is already in the
  data (results loaded from a CSV, then defined) is still accepted.
- The desktop app is deprecated and no longer installed by default. Install it with `pip install "alchemist-nrel[desktop]"`; `alchemist` prints a deprecation notice on start and an install hint if the extra is missing. The web app (`alchemist-web`) is the supported interface. The extra carries `customtkinter`, `tksheet`, `ctkmessagebox`, `mplcursors` and `tabulate`, which only `ui/` imports.

### New Features
- **Suggested-vs-actual provenance.** Every experiment records what the model
  suggested and what was actually run, in the session file and the audit log.
  Each data row carries a hidden `ProvenanceId` that joins it to a provenance
  record: suggested inputs, actual inputs, per-variable delta, strategy,
  acquisition parameters, output, noise and timestamp. Actuals can no longer
  drift from a suggestion while still being labelled with its strategy. The
  web app's Add Point goes through the work queue's `complete()` lifecycle;
  `GET /experiments/provenance` and `/{id}` return the records. `ProvenanceId`
  is never a model input.
- **Suggested and actual conditions in the Add Point dialog.** Suggested values
  are shown read-only (integers rounded to whole numbers, reals to 3 decimals;
  the tooltip shows the raw value) beside editable actual boxes pre-filled with
  the suggestion. The recorded point uses the actual, hardware-adjusted values.
- **Suggestions carry their iteration number** end to end, so the Add Point
  dialog shows the real iteration instead of `N/A`, and a batch is recorded
  under one iteration.
- **Linear input constraints over REST.** `POST` and `GET`
  `/sessions/{id}/constraints`, and `DELETE .../{name}`. Names are unique and
  are the delete identity, so one call removes exactly one constraint.
  Constraints could previously be set from Python only, so no REST or web
  client could use them.
- **Design responses carry a `feasibility` block.** `/initial-design` and
  `/optimal-design` report which constraints applied, how many structural
  points a classical design dropped, the optimal-design candidate accounting
  (candidates total, feasible, boundary and vertex points added) and an
  `estimability` verdict; `null` when no constraints are registered. This
  used to exist only in a server log line.
- **`/variables/load` accepts the `{variables, constraints}` document** that
  `SearchSpace.save_to_json` writes, as well as the bare variable list. Either
  form is validated whole: a bad variable or constraint anywhere in the file
  rejects the document and leaves the session untouched, and loaded
  constraints pass the same checks as `POST /constraints`.
- **`/variables/export?include_constraints=true`** returns that document, so
  load, export and load again round-trips constraints. The default export is
  still the bare variable list.
- **The web app links to the user guide.** A "User guide" link after the
  tagline in the header opens the documentation site in a new tab.
- **The web app saves to a file you choose and updates it in place** (Chrome
  and Edge, on https or localhost). The first Save asks where; later Saves
  (⌘S / Ctrl+S) rewrite that file, and Save As… (⇧⌘S / Ctrl+Shift+S) picks a
  new one. A session opened with Load Session saves back to the file it came
  from. The header names the linked file and shows a dot when there are unsaved
  changes. The variables file gets the same Load File and Save File linking;
  the audit log and the design CSVs open the Save dialog. Other browsers, and
  plain-http addresses such as `http://fscl2:8200`, download as before. The
  leave-page prompt appears only when there are unsaved changes. No REST
  endpoint changed.
- **The contour plot's fixed values can be typed and stepped.** Each fixed real
  variable has a number box below its slider: type a value and press Enter (or
  leave the box) to set it, clamped to the variable's range, or press ↑ and ↓ to
  step it as the slider does. The slider's arrow keys update the plot too, and
  the plot is requested once, 150 ms after the last change. The previous
  contour stays on screen, dimmed, while the next one loads.
- **The contour plot keeps one colour scale across slices.** By default the
  colour bar spans the observed outputs (the smallest and largest result), so
  moving a fixed value no longer rescales it and two slices can be compared by
  colour: a slice that predicts lower values looks darker. A prediction outside
  the observed range takes the end colour; hover still shows its value. A
  Scale control under Colormap switches to **Per slice**, the old behaviour,
  which fits the colours to each slice's own range. The browser remembers the
  choice. The contour response gains an `output_range` field (`null` when the
  outputs have no span).
- **The web app reopens the session you were working on (W30).** After a
  reload, or after quitting and relaunching the server or the desktop app, it
  opens the session this browser last created, loaded, restored or joined,
  with its current data and still linked to its file, so the next Save writes
  there (Chrome asks once more for permission to write after a reload).
  `?session=` in the URL still takes precedence. A session the server no longer
  has is forgotten without an error. The remembered session is per browser, so
  several browsers sharing one server each reopen their own. The
  unsaved-changes dot also survives a relaunch.
- **Sobol designs take a seed (W5).** The Initial Design panel shows
  **Seed (opt)** for Sobol as it does for LHS and Random, so a Sobol design can
  be reproduced.
- **Session metadata over REST (W27, W11, A7).** `GET /sessions/{id}` includes
  `metadata` (name, description, tags, author, last_modified) and
  `target_columns`, the objective column(s) of the data. `author` is accepted
  by `PATCH /metadata` and returned by both metadata routes and the session
  export; it used to be dropped. `POST /sessions` accepts an optional body
  with name, description, tags and author; it used to be ignored.
- **`GET /sessions/{id}/model` returns `training_config`**, the arguments of
  the last `train_model()` call, saved with the session. **`GET
  /sessions/{id}/state`** fills `model_backend`, `last_suggestion` and
  `last_acquisition_value`, which were always null (A27), and adds
  `last_goal` (`maximize`, `minimize`, or null before any suggestion; the first
  objective's on a multi-objective session).
- **`POST /model/train` takes `optimizer` for the scikit-learn backend (W42)**,
  and the web app's **Optimizer** menu sends it and restores it from the saved
  configuration. Every web and REST scikit-learn fit used `L-BFGS-B` whatever
  was chosen.

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
  pin the function, 1 where the model is at its prior) in
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

### Bug Fixes

**Security**

- The web app's catch-all route could serve any file the server user can read (A33). `GET /..%2f..%2f.alchemist%2fconfig.json` returned the LLM API-key store, and session files or `/etc/passwd` could be read the same way. The route now resolves every request against the static root and serves nothing outside it; such paths fall through to the app's index page. Update any server reachable by other people.
- `DELETE /sessions/{id}/backup` and `POST /recovery/{id}/restore` used the raw session ID as a file pattern (A34). With `*` as the ID the first deleted every user's crash backups and the second restored one and renamed them all. The ID must now be a plain session ID before any recovery file is looked up, and it is escaped in the pattern; anything else touches nothing.
- `GET /llm/ollama/models` accepts only `http` and `https` URLs without credentials as `base_url`.

**Data loss and persistence**

- Session load dropped every row when the objective column was not named `Output`. Since at least v0.3.3 the loader hardcoded the target column, so a session whose objective was named, say, `Methane umol/g/Wh` restored 0 experiments; a per-row `try/except` swallowed the rows. The loader now reads the target from the saved `target_columns` key or infers it from the columns that are neither variables nor bookkeeping, and `save_session` writes `target_columns`. Older files still load.
- A multi-objective session lost its other objectives on load (A35). `load_session` rebuilt the rows with the first objective only, so a two-objective file came back with the second objective empty on every row, the load-time retrain failed, and saving again wrote the loss to disk. Every objective is restored now.
- Constraints were lost on server restart. `SearchSpace.save_to_json` wrote them but the session serializer did not, so every API session, and every crash-recovery backup, dropped its constraints on reload.
- Outcome constraints survive save and load (A29). `save_session` never wrote them, so a reloaded session suggested unconstrained. The session file carries an `outcome_constraints` key and `load_session` restores it.
- `session.load_session(path)` on an existing session replaces all of its state and keeps the provenance (A31). The instance form copied everything from the file except `provenance`, so `get_provenance()` was empty afterwards and the next `save_session` wrote `provenance: []` while the rows kept their IDs. The class form and REST import were not affected.
- `OptimizationSession.load_session(path, retrain_on_load=False)` on the class honours the keyword (A16). It was ignored and the model was retrained anyway. The server's own loaders (reloading a saved session, REST import and upload) retrain on purpose; see Changed.
- Session files are strict JSON (W13). An empty cell, such as a result with no `ProvenanceId` or no `Noise`, was written as a bare `NaN`, which `JSON.parse` and other strict parsers refuse. Empty cells are `null`; the writer refuses to emit `NaN`, and the loader reads `null` back as an empty cell.
- Every session and recovery file is read and written as UTF-8 whatever the machine's locale. A session name with `α` or `≥` could not be saved on a cp1252 Windows box.
- A session lock survives a server restart. The lock lived in memory, so after a restart `GET /lock` reported the session unlocked, another client could take it mid-run, and the controller's own `DELETE /lock?lock_token=` answered 403. The lock is now part of the session file's store metadata.
- The desktop app no longer drops data when it syncs a session made in the web app or by the work queue (D1, D2). If the sync would lose observed or hard-to-change variables, derived variables, input constraints, work-queue provenance or split-plot columns, or rows that have a result but an empty or non-numeric input, it refuses, lists what would be lost and says to open the session in the web app. Nothing is changed, and Save, Save As, Train Model, the audit-log lock and the variable dialog stop. Rows with no result yet are still dropped as before, and sessions built in the desktop app itself sync as they did.
- The web app's variables Save File keeps input constraints (W35). It asked for the bare variable list, so it dropped a session's constraints and, through a file linked by Load File, overwrote a `{variables, constraints}` file with the bare list. It writes `{"variables": [...], "constraints": [...]}` when the session has constraints and the bare list otherwise.
- Recording one point into an empty session no longer loses the session (W1). The data summary of a single row produced a `NaN`, `GET /sessions/{id}` answered 400, and the web app showed "Session not found" and cleared the session.
- `GET /sessions/{id}` answers even when the data summary cannot be computed. A data frame missing a variable column made the route answer 400 and the web app lose the session; the `data` block now carries the row count and an `error` message, and the failure is logged.
- The recovery banner kept offering the same old backup. Restoring a backup made a new session but left the original's backup in place, and saving the new session never cleared it, so every reload offered the old work again as unsaved. A restore hands the backup to the new session, saving that session clears it, and the banner skips the session being reopened. Backups are taken only while a session has changes not yet saved to its file, so saved work is no longer offered back as unsaved. The "Unsaved work detected" banner is also offered only for sessions this browser worked on: the browser keeps the last 50 session IDs it created, joined, loaded or restored, so backups of other people's sessions on a shared server are not offered to a new profile.
- The web app warns when the server could not autosave the session file (W52). The server reports a failed write in the `X-ALchemist-Saved` header, which no client read, so a full disk or an unwritable cache directory went unnoticed. One warning per minute now says to use **Save** to keep a copy.
- `save_session` writes `kernel_params` (N1). It wrote `{}` even after training with `{"nu": 1.5}`, so a reload retrained with the default ν.

**Optimization engine**

- Batch acquisitions honour Minimize (T7). `qEI` and `qUCB` were built without a direction and always looked for high values, so a minimisation campaign using a batch got the worst points. They negate the objective for `goal="minimize"`. EI and LogEI with an outcome constraint go through the same directed path, so they minimize correctly too, and EI no longer ignores the constraint (T20). With outcome constraints set, improvement is measured against the best feasible training point.
- `kappa` reaches UCB and qUCB on BoTorch (T13). The REST API and the web app send the exploration weight as `kappa`; BoTorch's UCB reads `beta`, so the value was dropped and both ran at β = 0.5. `kappa` is taken as `beta` when `beta` is not given. `POST /acquisition/suggest` no longer fills in `xi = 0.01` and `kappa = 2.0` when the request leaves them out: the request model defaulted both and the route forwarded them, so a REST scikit-learn UCB ran at κ = 2.0 instead of 1.96, and with the alias above every REST UCB and qUCB would have run at β = 2.0. The core's defaults apply when the fields are omitted, and the audit's `acq_params` carry only what was sent (A26).
- Linear input constraints were honoured in only two of four places. Acquisition and space-filling designs respected them, but classical designs generated their points without them and dropped the infeasible ones afterwards with a log warning, which left a design rank-deficient for the model it claimed to fit, and optimal designs ignored them, so the exchange algorithm spent its budget on points it could not keep. Optimal designs now select from a candidate set that is filtered and then augmented with points on the feasible region's boundary, which a filtered lattice lacks and an optimal design needs, so a constrained D-, A- or I-optimal design covers its whole feasible region. An equality constraint over `real` variables is satisfiable for the first time; optimal design is the only method that places points on the zero-volume slice such a constraint defines. The model must not contain the terms the equality makes collinear: with `x1 + x2 == rhs` the intercept and the two main effects are exactly dependent, so pass an `effects` list that drops one of the tied variables rather than a `model_type` shortcut. The rank-deficiency error names this case and its remedy.
- Non-model variables were spread straight through a constraint. Variables absent from every model term were overwritten with a shuffled range after selection, undoing the feasibility work for those columns. They are now drawn from each row's feasible interval.
- A `context` variable dropped a real variable from every design. Design generation zipped the full variable list against samples drawn from dimensions that exclude `context`, so a context variable anywhere but last produced points missing an optimization variable, with its values relabelled onto the context column, and no error. The classical and optimal paths were affected more than space-filling; the desktop design table had the same skew.
- A provably infeasible constrained design ground for minutes and blocked the whole server. The reject-and-resample loop escalated its batch size 4096 times before giving up (480 s to a correct 400), and the design endpoints ran on the ASGI event loop, so one such request starved every other request in the process. Impossible regions are proven empty up front and refused at once with `InfeasibleRegionError` (`400`), and design generation runs in a thread pool like the other heavy endpoints.
- `random_seed` was applied through process-global state. Two concurrent seeded design requests interleaved and neither got the design its seed names. The seed is honoured per call.
- Constraint errors were indistinguishable from any other bad request. Both are `ValueError` subclasses and already returned `400`, but under `"error_type": "ValueError"`, next to malformed bounds and unknown methods. They now report `DesignNotEstimableError` and `InfeasibleRegionError`, which call for different remedies.
- Space-filling designs returned values that were not JSON-serializable. An `integer` variable came back as `np.int64`, which `json.dumps` refuses, so the endpoint failed; `discrete` variables had been leaking `np.float64`. Every variable type returns a native scalar.
- `halton` is a Halton sequence (T12). It ran the Hammersly sampler and gave the same points as `hammersly`; it runs scikit-optimize's `Halton`, and the golden design fixture's `halton` entry was re-locked. A circumscribed CCD still returns the inscribed design, and `detmax` and `modified_fedorov` still run the Fedorov exchange (T9, T10); both log a warning saying so, and the docs say it plainly.
- scikit-learn output scaling works for every listed transform, and the noise follows it (T17, T18). `minmax` and `robust` built no output scaler, so training ran unscaled while cross-validation scaled; both build the scaler now. With a Noise column and an output transform, the noise variance was passed to the GP in the output's raw units while `y` was scaled, so a noise of 4.0 on an output spanning thousands was weighted 5e4 times too heavily. The variance is carried through the same scaling as `y`.
- BoTorch `input_transform_type="standardize"` standardizes (T6). It built `Normalize`, which rescales to [0, 1]; it builds `InputStandardize`.
- BoTorch models trained through the REST API or the web app without choosing transforms normalize inputs and standardize outputs, as the docs describe (W2). The web app used to train unscaled, which can give an R² near zero on data with large ranges. Choosing `none` explicitly still trains unscaled.
- Training on data where only some experiments carry a noise value raises a `ValueError` (REST 400) naming the rows, instead of a NaN error from BoTorch or a "not positive definite" error from scikit-learn; an empty Noise column trains as no noise (A6).
- Noise is documented as a variance (T19). The `add_experiment`, `ExperimentManager` and docs wording says the Noise column is the noise variance (σ²), which is how both backends use it (BoTorch `train_Yvar`, scikit-learn `alpha`).
- Regret and hypervolume plots showed gaps where a subset GP failed to fit. The posterior overlay refits a GP on each data prefix; a small or ill-conditioned prefix (categorical or `AdditiveKernel` models, which skip the hyperparameter-reuse fast path and re-optimize every iteration) could raise `ModelFittingError` and leave that iteration as NaN. Both the single-objective and the multi-objective path now try the requested transforms, then no transforms, then Cholesky jitter of 1e-3 and 1e-2; every plotted value is a prediction from a fitted GP, and NaN is the last resort when all of that fails.
- Results added after `load_search_space` train again (A28). The loaded search space was not passed on to the results table, so rows added with `add_experiment` afterwards had no inputs as far as training was concerned and `train_model` failed with "Found array with 0 feature(s)" or a state-dict error.
- Deleting a variable deletes the constraints that name it (A14). They used to stay, and since feasibility sums only the columns present, `x1 + x2 <= 12` was enforced as `x1 <= 12` after `x2` was gone. The dropped constraints are named in a log warning. The REST route and the web app's variable deletion do the same.
- Editing or deleting a variable could corrupt the search space. `PUT` and `DELETE` on a variable rebuilt dimensions by an index taken from a list that includes `context` entries and applied to one that does not, so `PUT` could return 200 and destroy a different variable's dimension, invisibly, since `GET /variables` reads the other list. Both routes go through one validated core path that replaces the dimension in place, which is also how `PUT` picks up the bound guards under Breaking Changes.
- `add_experiment` refuses a key that is not a variable (A4). An unknown key used to become a column in the data with no error; the REST experiment endpoints answer 400. A value outside a real or integer variable's bounds is still recorded, since observed variables and hardware-adjusted actuals may sit outside, and is logged as a warning.
- `move_staged_to_experiments(iteration=)` takes effect (A3). The argument was accepted and ignored, so a batch recorded through it, or through `POST .../staged/complete`, took the next free iteration numbers one each.
- Load-time retrain events reach your listeners (A30). With `session.load_session(path)`, `model_retrained` and `model_retrain_failed` were emitted on an internal session; they are emitted on the session you called, before `session_loaded`.
- A `once` listener no longer skips the listener after it, and `off` removes a pending `once` (A32).
- Find Model Optimum respects input constraints over REST and in the desktop app; it used to search the whole box.

**REST API**

- An unknown `/api/...` path is a real 404 (A2). It answered 200 with `{"detail": "Not Found"}` in the body, so a controller with a typo in a route saw success.
- `POST /experiments/queue` checks every item before staging any (A23). A row missing a variable, or with an undeclared category, is refused with a 400 that names the item. Such a row used to be accepted as pending and then made `staged/complete` refuse every staged point until it was deleted by ID.
- `POST /experiments/queue/{item_id}/complete` records the row under the `iteration` it is sent (A22). The field was accepted and ignored.
- `POST /experiments/record-batch` records a request under one iteration when its entries give none. Each row took the next free number, so a 10-point design recorded without `iteration` got Iteration 0 to 9; entries that carry an `iteration` are recorded as before.
- A suggestion or an auto-retrain no longer stalls every other request. `POST /acquisition/suggest` and the auto-train after a recorded result ran on the server's event loop, so while a qEI or a retrain ran, `/health`, the web app's 30-second backup and a controller's polling all waited. They run in the thread pool, as `POST /model/train` already did.
- "Retrain model" was ignored when results were recorded through the work queue. The queue-complete endpoint had no auto-train; it honours the retrain control like the direct add-experiment path.
- Find-optimum on a multi-objective session is a 400 with a clear message, not a 500 (A19). The route read a single optimum from a result that holds a Pareto frontier.
- The parity, Q-Q and calibration routes work on a multi-objective model (A20). They take a `target` query parameter naming the objective (the first by default) and read that objective's cross-validation results; an unknown objective is a 400. They used to answer "Model does not have cached cross-validation results".
- Batch and CSV-upload rows get provenance records and a `ProvenanceId` like `POST /experiments` does (A24). They had none, so those rows were untraceable and the session export carried bare `NaN` in the column.
- A manual provenance record's `strategy` is the `reason` that was sent (A25). It was always `Manual`.
- `POST /experiments` without an `output` is a 422, not a 500 (F3). The field is required; stage a candidate with the work queue instead.
- A bare-list `/variables/load` is refused whole when any entry fails (W36). It added variables one at a time, so a duplicate at entry 3 left entries 1 and 2 added. The object form refuses a variable the recorded rows have no value for, with the same 400 message as `POST /variables`; it wrote to the search space directly and bypassed that guard. Replacing the variables the data already has still works.
- The AI effect-suggestion prompt carries each variable's range and categories (A15). It read `lower_bound`, `upper_bound` and `categories` where the search space stores `min`, `max` and `values`, so every real variable was rendered `?–?` and every categorical `[]`.
- Saving one LLM provider's settings no longer wipes the others (A17). `POST /llm/config` replaced the whole `llm` section; it merges per provider, so a key saved earlier stays unless a new one is sent.
- The contour plot works on a session whose CSV had extra non-variable columns. The prediction grid's features come from the trained model (`original_feature_names`), not from every data column that is not bookkeeping, which made an extra column a bogus model input and a 500.
- `api/example_client.py` runs against the real API (A40). It sent `type: continuous` with `bounds` (a 422), read an `output_range` the summary does not have and an `r2_score` the metrics do not have, and its unit test faked every call and passed anyway. The script sends `real` variables with `min` and `max` and reads `target_stats` and `r2`, and the test validates every request body against the real request models and builds every response from the real response models.
- Opening `http://localhost:8000` showed JSON instead of the web app. `alchemist-web` prints that address, but `/` always returned the API-info JSON. A browser gets the web app there; scripts and other clients that do not ask for HTML still get the JSON, which also has its own route at `/api`. `/health` is unchanged.
- `GET /download` removes its temporary file after sending it, and a WebSocket client connecting or leaving during a broadcast no longer raises inside the request that broadcast.

**Web app**

- Add Point works with nothing staged (W28). It opens a blank point to type an experiment's conditions and result by hand, for a first result or an experiment ALchemist did not suggest. With points already staged, an **Add blank point** control adds a hand-typed point alongside them.
- Joining or loading a session no longer brings back already-recorded points as pending (W26). The audit-log fallback skips any suggestion that matches a row in the experiment table, comparing reals after the 3-decimal rounding the Add Point dialog records, integers as whole numbers, and categorical or discrete values as given. A point recorded with an edited actual value is also skipped, matched through its provenance record's suggested conditions.
- Batch acquisition points appeared identical in the Add Point dialog. The dialog seeded its form from props at mount only, and stepping through a batch swapped the prop without remounting, so every point showed the first point's values. The dialog is keyed per suggestion, so it remounts with fresh state. Acquisition and staging were always correct; this was a display bug.
- Staged suggestions persisted across sessions ("ghost points"). The restore effect only ever added to state and never cleared it, so a new session started with stale suggestions. It now resolves to a definite value, clearing when the session has none, with a race guard.
- Add Point pre-fills each actual field with the full suggested value (W33). It used the value rounded to three decimals, so provenance recorded a deviation nobody caused (247.55 run against 247.5503 suggested). The read-only suggested cell keeps the rounded display.
- Add Point flags an actual value outside its variable's bounds. The field gets an amber "outside 200–300" note and the Save button's tooltip lists them; saving is still allowed, since hardware-adjusted conditions may sit slightly outside.
- Add Point knows the objective's name (W16, W43). When the server reports the session's `target_columns`, the result field and table column are labelled with the target's name instead of "Output", and a pasted CSV block headed with that name is recognised.
- Add Point refuses a point on a multi-objective session instead of recording it with the other objectives empty; load multi-objective results from a CSV.
- Add Variable refuses Min ≥ Max, and a non-numeric Min or Max, inline under the fields before any request. A variable the server rejects shows the server's reason: a 400 (a duplicate name, a session that already has recorded experiments) is an error toast with the server's detail, a 422 lists the fields, and an unreachable server says so.
- Open plots stay live (W20). The contour, parity, metrics, Q-Q and calibration plots redraw after new results, a retrain or a new suggestion instead of showing cached data; an open contour also picks up a CSV load, and Stage Suggestions refreshes the queue (FE-11).
- Open plots follow the header theme toggle. The contour plot kept the theme it mounted with, and the parity, metrics, Q-Q, calibration and objective-trace plots drew light grids, white points and light tooltips in the dark theme. All of them redraw when you switch. The parity plot's "Predictions" legend text was invisible in the light theme.
- Model Visualizations close when the session changes (W24). After **Clear** and a new or joined session, the panel still showed, and refetched, the old session's plots.
- The Model tab shows the settings a joined or loaded session was trained with (W31). It showed scikit-learn, RBF and no scaling whatever the session held, so one **Train Model** click switched backend. The tab copies backend, kernel, nu, scaling, optimizer and calibration from `training_config`; the controls stay editable.
- The retrain after Add Point uses your model settings (W3). It refitted scikit-learn RBF with normalize and standardize whatever the Model tab said; it reuses the session's last training call.
- Training a multi-objective model in the web app blanked the whole page (W25). The model reports no cross-validation metrics for more than one objective, and the UI formatted the missing values. The Model panel says how many objectives were trained and that their metrics are not computed yet. Each panel also contains its own errors, so a crash in one shows a message there while the header, Save included, keeps working.
- The Model tab no longer pops a native alert for fewer than 5 points. The **Train Model** button is disabled with the reason in its tooltip and the hint below it.
- A failed refresh no longer clears the session (W46). Any failed background fetch of the session (server restarting, laptop between networks) cleared it and forgot the stored ID, undoing reopen-last-session. Only a 404 clears it; other errors show a toast and the fetch retries.
- A failed request is not sent twice (W47, part of W22). The web app retried every failed write once, so a failing Find Model Optimum, train, CSV upload or batch add reached the server twice.
- Emptying the contour's Integer fixed-value box no longer blanks the plot (W48). It sent `null`, the server answered 422 and the figure was replaced by an error until a digit was typed; an out-of-range value was sent as typed. The box ignores an empty value and clamps to the variable's bounds, as the Real control does.
- The contour's Plotly toolbar appears on hover (W38), so it no longer covers the title in a narrow column.
- Select Target Column(s) pre-selects the recommended column (W6, dialog half). The selection was set once with no columns, so it showed the first column while nothing was selected and **Confirm** stayed disabled until you picked one. It pre-selects the server's recommendation, else the first column, each time it opens.
- The experiments table hides the `ProvenanceId` column and its sticky header is solid (W14). The column is still in the data and the export; rows no longer show through the header.
- Work Queue rows show numbers as the experiment table does (W39): reals to three decimals and bookkeeping integers whole, instead of `Temperature=271.17931449935946`.
- A page opened with `?mode=monitor` stays in the monitor when the controller releases the lock (W41).
- The monitor's best-so-far line follows the last suggestion's goal (W40). It was always the running maximum; **Objective so far** reads `last_goal` from the session state and maximizes when it is null.
- The header names the session: the session name beside the short ID and counts ("Methanol reactor · d80984c1 · 4V · 16E"), or "Untitled Session".
- Clear asks first when the session has unsaved changes. The session stays on the server, but the browser forgets it and the file it saves to.
- The Edit Info dialog opened empty. It read metadata from a session response that did not carry it; it loads the session's current name, description, tags and author.
- Web notifications covered the session buttons and could not be dismissed. They appear bottom-right with a close button, errors and warnings stay up for 10 s instead of 4 s so a long API message can be read, and they follow the app's light or dark theme.
- Files saved from the web app are named after the session. A session saves as `<title>.json`, with `<title>_variables.json` and `<title>_audit_log.md` beside it, instead of a fragment of the session ID. Characters Windows or macOS forbid in file names become `_`; an untitled session falls back to its ID. The REST endpoints are unchanged.

### Changed

- **The contour plot's layout is tidier.** The plot area is never taller than it is wide, its legend sits above the plot, and Fixed Values comes first, directly under the figure. In a narrow column the calibration table wraps under its chart.
- **The server saves a session after every request that changes it.** It used to write a session only when it was created, imported, locked or unlocked, and on `POST /sessions/{session_id}/save`, so results, queue changes and trained models made in between were lost if the server stopped. Every successful `POST`, `PUT`, `PATCH` or `DELETE` under `/api/v1/sessions/{session_id}/` saves the session, except routes that only read it. The response carries `X-ALchemist-Saved: true` or `false`; a failed save is logged and does not turn a successful response into an error.
- **The server loads a session the first time its ID is used, not all of them at startup.** A server started in a folder with many saved sessions starts without reading them.
- **A reloaded or imported session gets its model back.** The server's loaders (lazy load, `POST /sessions/import`, `POST /sessions/upload`) rebuild the model by repeating the session's last training call, transforms included, and fall back to backend, kernel and kernel parameters for older files. If the retrain fails it is logged and the session loads without a model; the training call stays in the file (`model_config` with `fitted: false`), so later saves keep it and the next reload tries it again.
- **Session files are written atomically.** The server writes a temporary file and renames it over the session's file, so a crash during a save leaves the previous file whole.
- **`POST /sessions/{session_id}/save` returns 500 when the write fails.** It already returned 404 for an unknown session; it used to report success after a failed write.
- **A session file the server cannot read answers 503, not 404.** A session whose file exists but is corrupt or unreadable is no longer reported as not found, so a controller does not take it for deleted. `DELETE` still removes it.
- **Deleting a session that the server cannot remove from disk answers 500.** A delete of a loaded session used to return 204 even when the file stayed (a Windows scanner holding it), and the session came back after a restart. The server retries a held file, then reports the failure.
- **A session ID that differs only in letter case no longer loads a second copy.** On macOS and Windows `<ID>.json` and `<id>.json` are the same file; the server requires the exact name, and the ID pattern no longer accepts a trailing newline.
- **Recovery backups no longer rewrite the session file, and lock and unlock write it once instead of twice.** The web app posts a backup every 30 seconds.

### Maintenance / Internal
- Three copies of the model-input metadata-exclusion logic became one
  `ExperimentManager.metadata_columns()` helper; the BoTorch evaluation and
  Pareto feature-selection paths use it too.
- `BoTorchModel` carries a per-instance `cholesky_jitter` attribute
  (defaulting to the module value). A caller that needs more jitter for one
  ill-conditioned fit sets it on its own instance instead of rebinding the
  `_CHOLESKY_JITTER` module global, which leaked across concurrent API
  requests and every other GPyTorch consumer in the process.
- The production web build (`tsc -b`) no longer fails on `*.test.ts(x)` in
  fresh checkouts (test files are excluded from `tsconfig.app.json`); vitest
  still type-checks them.
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
