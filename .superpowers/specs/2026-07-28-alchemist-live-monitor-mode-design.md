# ALchemist Live-Monitor Mode — Design

**Date:** 2026-07-28
**Status:** Design spec (pre-implementation). Awaiting human review before writing the implementation plan.
**Repo:** ALchemist (`alchemist-web/`, `alchemist_core/`, `api/`)
**Upstream decision doc:** `nanoparticle-hmi/docs/superpowers/specs/2026-07-27-autonomous-monitoring-control-ux-decision.md` (§2, §4 step 3, §4.5)
**Prerequisites (shipped, `main` `6e5e499`):**
- `docs/superpowers/specs/2026-07-28-experiment-work-queue-design.md` (real work queue: per-item IDs/status + `queue_item_updated`/`queue_updated` events)
- Opaque objective label/unit metadata (`objective_metadata`)

This is **step 3 of 5** in the autonomous monitoring & control UX program. It **unblocks step 4** (controller slimming): slimming removes generic optimizer functionality (DoE/model/kernel/acquisition/viz) from the controller, which must already have a home in ALchemist's web app first.

---

## 1. Problem

Monitor mode today is thin and brittle. `alchemist-web/src/features/monitoring/MonitoringDashboard.tsx` (189 lines) is a poll-and-render status-card view: session id, variable/experiment counts, a "model trained" flag, a summary block, and the last suggestion. It has **no live-campaign controls, no queue view, and no streaming visualization**. During a controller-driven run there is nothing that lets a human watch the loop turn or trust the surrogate is learning.

Meanwhile the pieces needed to fix this **already exist** and are unused by monitor mode:

- **Backend is ready, frontend is empty.** Step 1 shipped `/experiments/queue` endpoints and emits `queue_item_updated` / `queue_updated` over the existing WebSocket, but `alchemist-web` has **zero** queue consumption — `useSessionEvents.ts` does not even listen for those two events.
- **Config panels exist** (`VariablesPanel`, `GPRPanel`, `AcquisitionPanel`, `InitialDesignPanel`/`OptimalDesignPanel`/`LLMSuggestPanel`), all `sessionId`-driven and reusable.
- **Plot components exist** (`ParityPlot`, `MetricsPlot`, `CalibrationCurve`, `QQPlot`, `HyperparametersDisplay`, `ContourPlot`), all react-query-key-driven — so "live" = invalidate the key on a WebSocket event.

So this task is **not** a from-scratch build. It is: *compose the existing config panels into a robust live-monitor mode + build genuine real-time visualization off the step-1 work-queue stream.*

### Load-bearing constraints

- **Domain-agnostic (§2.1).** ALchemist serves multiple future consumers. **No** reactor/spectroscopy/MQTT/band/detector concepts may appear. The objective is an **opaque scalar** displayed via the step-2 `objective_metadata` label/unit — shown, never interpreted.
- **Non-commander (§4.5, Model A).** The external controller drives the physical loop and is the sole commander of the reactor. Monitor mode is an **observer + live-config surface**, never a hardware driver's seat. No control commands the reactor.
- **Reuse the existing aesthetic.** Reused panels/plots already carry the app's look; compose them as-is. The current `MonitoringDashboard` aesthetic is scrapped.
- **Embeddable (step 5).** The web app will be iframe-embedded on a lab monitor alongside a reactor-monitor and Grafana. Monitor mode must read well as a standalone pane.

---

## 2. What Monitor Mode Is (scope, from §4.5)

An **observer + live-config surface**. Three jobs:

1. **Configure & stage before handover** — set model/kernel/acquisition/DoE and stage an initial design into the work queue.
2. **Watch a controller-driven campaign live** — the work queue filling and items transitioning `pending→running→done/failed`, the model retraining, and metrics/parity/objective plots updating as results arrive.
3. **Tune config live between cycles** — adjust model/acquisition config mid-run; the controller picks up ALchemist's *current* config the next time it asks "suggest next".

**It is NOT** a hardware controller. Under Model A the controller asks ALchemist "suggest next", applies setpoints, measures, reduces to an opaque scalar, and calls `complete(item, output, objective_label)`; ALchemist retrains and monitor mode re-renders. The work queue is the **observability/coordination surface**, not a dispatch mechanism — monitor mode is the *reader* of that per-item stream.

### v1 vs deferred

- **v1 streaming views:** queue timeline, metrics-over-time, live parity, objective-so-far / best-value trace.
- **Deferred to v2:** contour-with-latest-point overlaid. Reason: it needs configurable X/Y axis selection and a rule for fixing the other dimensions (e.g. last point's values). It also fetches on-demand with `staleTime: Infinity` today, so streaming it is a distinct effort. Kept explicitly in our back pocket.

---

## 3. Layout & Component Structure

Replaces the current full-screen `MonitoringDashboard` branch in `App.tsx` (reusing the existing `isMonitoringMode` switch + full-screen slot). **Three persistent tabs** with a header strip showing the opaque objective label/unit and run status. **Default tab = Live**; a `?tab=` URL param is honored so an embedded viewer can force the live view.

### Tab 1 — Config (setup + live tuning, merged)

Reuses `VariablesPanel`, `GPRPanel`, `AcquisitionPanel`, and the DoE panels (`InitialDesignPanel` / `OptimalDesignPanel` / `LLMSuggestPanel`) **unchanged**.

- **Before handover:** stage an initial design via the existing staging flow (into the work queue).
- **During a run:** edits **auto-apply on save** to the session; a passive banner reads:
  > *"Applies to the next suggestion the controller requests — ALchemist does not initiate cycles."*

  No apply/commit button, no reactor controls. This matches the truth that ALchemist merely holds current config; the controller decides when to ask for the next point.

### Tab 2 — Live (the spine; the embed default)

- **Queue timeline (new).** A live list/timeline of `QueueItem`s: status (`pending→running→done/failed`), input point, per-item reason, and objective result as it lands. Failed items render distinctly with their `error` string. Driven by `queue_item_updated` / `queue_updated` + `GET /experiments/queue` resync.
- **Metrics-over-time** (`MetricsPlot`) and **live Parity** (`ParityPlot`) — reused as-is, refreshed on `model_trained`.
- **Objective-so-far / best-value trace (new).** Per-item objective value + cumulative-best envelope, derived client-side from `done` queue items, labeled with the opaque objective label/unit.
- **Pre-handover / empty state:** an explicit "no items yet — awaiting controller" panel (not a blank/broken view) — important for the embedded pane.

### Tab 3 — History

- **Config-change provenance timeline (new, §4.5).** Every applied config change — model backend/kernel/hyperparameters, acquisition strategy/params, objective label — old→new + cycle, from the audit log.
- **Deeper diagnostics**, reused as-is: `CalibrationCurve`, `QQPlot`, `HyperparametersDisplay`.

### New components (thin)

- `LiveMonitor` — the shell: tab bar, header strip, `?tab=` handling.
- `QueueTimeline` — renders the queue stream.
- `ObjectiveTrace` — computes/plots per-item + cumulative-best objective.
- `ConfigChangeTimeline` — renders provenance entries.

Everything else is existing panels/plots composed in.

---

## 4. Data Flow & Real-Time Wiring

**Transport: reuse the step-1 WebSocket.** No new backend events, no parallel mechanism. Extend the existing `useSessionEvents` client (`hooks/useSessionEvents.ts`) which already handles `experiments_updated`, `model_trained`, `lock_status_changed`. Add two branches:

- `queue_item_updated` → invalidate the queue query (one item changed).
- `queue_updated` → invalidate the queue query (coarse resync ping).

### Query invalidation map (react-query key → refetch)

| Event | Invalidates |
|---|---|
| `queue_item_updated` / `queue_updated` | `['experiments-queue', sessionId]` |
| `model_trained` | `['parity-data', …]`, `['metrics-data', …]`, `['calibration-…']`, `['qq-…']`, `['hyperparameters', …]` |
| `experiments_updated` | `['experiments', …]`, `['experiments-summary', …]`, `['session', …]` |

### New query hook

`useExperimentQueue(sessionId)` → `GET /experiments/queue` (backend already shipped in step 1). On WebSocket reconnect, one GET fully resyncs — the step-1 resync contract. Both `QueueTimeline` and `ObjectiveTrace` read from this hook.

### Objective trace

Computed in `ObjectiveTrace` from the queue's `done` items (each carries `output` + `completed_at`): plots per-item value and a cumulative-best envelope, labeled from `objective_metadata` (falls back to the raw column name when absent). No new endpoint.

### Contour (v2)

Untouched — stays manual/on-demand with its current request/axis UI.

### Config provenance (backend)

Add config-change logging in `alchemist_core/audit_log.py` via the existing generic `log_event` hook. When model or acquisition config is **applied** (or the objective label changes), write a **timestamped** entry, `entry_type: "config_changed"`, capturing `old`→`new` values + the current iteration. ("Cycle" in the decision doc §4.5 maps to ALchemist's existing **iteration** counter — ALchemist has no reactor-cycle concept, keeping it domain-agnostic.) Objective-label changes already log (`objective_label_changed`, step 1); this generalizes the pattern to model/acquisition config. A read endpoint exposes these entries to the History tab's `ConfigChangeTimeline`.

---

## 5. Error Handling & Edge Cases

- **WebSocket drop** → existing 5s auto-reconnect; on reconnect, `GET /experiments/queue` + query invalidation fully resync (no missed-event gap). Inherited from the step-1 resync contract.
- **`failed` queue items** → rendered distinctly with their `error` string; excluded from the dataset and the objective trace. The human sees *why* an item failed (the cross-boundary error string the controller sets).
- **Empty / pre-handover** → explicit "no items yet — awaiting controller" state on the Live tab.
- **Missing `objective_metadata`** (older sessions) → fall back to the raw column name, matching step-1 display behavior.
- **Config edit mid-suggestion** → auto-apply writes session state; the controller's *next* request picks it up. No locking; there is no hardware to race.

---

## 6. Provenance Integrity (§4.5)

Config-change audit entries are:

- **Timestamped server-side** at apply time (not client clock).
- **Old→new + cycle** so a run reconstructs faithfully across mid-campaign tuning (e.g. "acq EI→qEI at cycle 12"). ("Cycle" = ALchemist's iteration counter; see §4.)
- **Append-only** in the audit log.

This is required for the systems-PoC manuscript's reproducibility — the honest cost of allowing live tuning.

---

## 7. Testing (TDD)

**Backend:**
- `config_changed` audit entries written on model/acquisition apply — old→new, cycle, server timestamp.
- Objective-label change still logs (`objective_label_changed` regression).
- Read endpoint returns config-change entries in order; old sessions → empty timeline.

**Frontend (component/hook):**
- `useExperimentQueue` fetches `GET /experiments/queue` and updates on event-driven invalidation.
- `useSessionEvents` new branches (`queue_item_updated`, `queue_updated`) fire the correct invalidations; existing branches unchanged.
- `QueueTimeline` renders each status; `failed` shows `error`; per-item reason shown.
- `ObjectiveTrace` computes cumulative-best from `done` items and labels with the opaque objective label; falls back to column name when metadata absent.
- `ConfigChangeTimeline` renders provenance entries old→new + cycle.
- Tab default = Live; `?tab=` param honored.
- Reused panels/plots mount with `sessionId` inside the monitor shell.

**Resync test:** subscribe → drive transitions → drop WS → reconnect → `GET /experiments/queue` matches observed final state.

---

## 8. Back-Compat & Breaking Changes

- **`MonitoringDashboard` replaced.** The thin status-card view is removed; its props (`sessionId`, `pollingInterval`) are subsumed by `LiveMonitor`. The `isMonitoringMode` switch + full-screen slot in `App.tsx` are reused. This is an **internal web-app change — no ALchemist REST API breakage** and no impact on any API consumer (including the controller).
- **Backend additive:** a `config_changed` audit entry type + a read endpoint. No existing endpoint changes. Old sessions load fine (empty provenance timeline; `objective_metadata` absent → column-name fallback).

No changes to the work-queue API, the objective-metadata API, or the WebSocket event set — this step is purely a *consumer* of what step 1/2 shipped, plus the additive config-provenance audit.

---

## 9. Out of Scope (flagged, belongs elsewhere)

- **Any reactor/spectroscopy/band/detector/MQTT concept** — controller-side (§2.1).
- **Pause/stop / soft-stop / coordination signals** — ALchemist never commands the reactor and cannot halt the controller. Any control-plane handshake belongs to the controller / **step 4** (§4.5). v1 monitor mode is pure observer + live-config, with **no pause/stop button**.
- **Contour-with-latest-point** — deferred to v2 (axis selection + fixed-dimension rule).
- **Reactor-monitor surface, Grafana, iframe composition** — **step 5**.
- **What the objective scalar means / how it's reduced** — the consumer owns reduction and sets the opaque label.

---

## 10. Program Position

- **Consumes:** step 1 (work queue + WebSocket events) and step 2 (objective metadata).
- **Unblocks:** **step 4 (controller slimming)** — the generic optimizer functionality removed from the controller now has a home in monitor mode.
- **Precedes:** step 5 (monitoring/UX composition layer), which iframe-embeds this web app on the lab monitor.
