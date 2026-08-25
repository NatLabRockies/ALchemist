# ALchemist Live-Monitor Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild ALchemist's web-app monitor mode into a robust, observer-only live-monitor with a streaming work-queue timeline, live plots, an objective-so-far trace, mid-run config tuning, and config-change provenance.

**Architecture:** A three-tab (`Config` / `Live` / `History`) `LiveMonitor` React shell replaces the thin `MonitoringDashboard`, reusing existing config panels and plot components. Real-time updates ride the already-shipped step-1 work-queue WebSocket events (`queue_item_updated` / `queue_updated`) plus `model_trained` / `experiments_updated`, driven by react-query cache invalidation. On the backend, model/acquisition config applies gain timestamped `config_changed` audit entries exposed by a new read endpoint. ALchemist never commands the reactor.

**Tech Stack:** Backend — Python, FastAPI, pytest (`~/miniforge3/envs/alchemist-env/bin/python -m pytest`). Frontend — React/TypeScript, TanStack Query, axios (`apiClient`), Vitest + Testing Library (`npm test` in `alchemist-web/`).

**Spec:** `docs/superpowers/specs/2026-07-28-alchemist-live-monitor-mode-design.md`

---

## File Structure

**Backend (create/modify):**
- Modify `alchemist_core/audit_log.py` — add `log_config_change()` helper (thin wrapper over `log_event`).
- Modify `api/routers/models.py` — log a `config_changed` audit entry on `train_model`.
- Modify `api/routers/acquisition.py` — log a `config_changed` audit entry on `suggest`.
- Modify `api/routers/experiments.py` — add `GET /{session_id}/audit/config-changes`.
- Modify `api/models/responses.py` — add `ConfigChangeEntry` + `ConfigChangesResponse`.
- Tests: `tests/unit/core/test_config_change_audit.py`, `tests/integration/api/test_config_changes_router.py`.

**Frontend (create/modify), all under `alchemist-web/src/`:**
- Create `api/endpoints/queue.ts` — `getQueue`, `getConfigChanges` axios calls.
- Create `hooks/api/useQueue.ts` — `useExperimentQueue`, `useConfigChanges`.
- Modify `hooks/useSessionEvents.ts` — handle `queue_item_updated` / `queue_updated`.
- Create `features/monitoring/LiveMonitor.tsx` — tab shell (replaces `MonitoringDashboard` at the call site).
- Create `features/monitoring/QueueTimeline.tsx`.
- Create `features/monitoring/ObjectiveTrace.tsx`.
- Create `features/monitoring/ConfigChangeTimeline.tsx`.
- Create `features/monitoring/tabs/{ConfigTab,LiveTab,HistoryTab}.tsx`.
- Modify `App.tsx` — render `LiveMonitor` instead of `MonitoringDashboard`.
- Create `test/queryWrapper.tsx` — react-query test helper (none exists yet).
- Tests co-located `*.test.tsx`.

**Types shared (frontend):** a `QueueItem` / `QueueListResponse` / `ConfigChangeEntry` TS type set in `api/endpoints/queue.ts`.

---

## Task 1: Backend — `log_config_change` audit helper

**Files:**
- Modify: `alchemist_core/audit_log.py` (add method to `AuditLog`, after `log_event` at `:285`)
- Test: `tests/unit/core/test_config_change_audit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/test_config_change_audit.py
from alchemist_core.audit_log import AuditLog


def test_log_config_change_records_old_new_and_iteration():
    log = AuditLog()
    entry = log.log_config_change(
        component="model",
        old={"kernel": "Matern"},
        new={"kernel": "RBF"},
        iteration=12,
    )
    assert entry.entry_type == "config_changed"
    assert entry.parameters["component"] == "model"
    assert entry.parameters["old"] == {"kernel": "Matern"}
    assert entry.parameters["new"] == {"kernel": "RBF"}
    assert entry.parameters["iteration"] == 12
    assert entry.timestamp  # ISO timestamp auto-set
    assert log.get_entries("config_changed") == [entry]


def test_log_config_change_iteration_optional():
    log = AuditLog()
    entry = log.log_config_change(component="acquisition", old={}, new={"strategy": "qEI"})
    assert "iteration" not in entry.parameters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_config_change_audit.py -v`
Expected: FAIL with `AttributeError: 'AuditLog' object has no attribute 'log_config_change'`

- [ ] **Step 3: Write minimal implementation**

Add to `AuditLog` in `alchemist_core/audit_log.py` immediately after the `log_event` method (after line 285):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_config_change_audit.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add alchemist_core/audit_log.py tests/unit/core/test_config_change_audit.py
git commit -m "feat(audit): add log_config_change for mid-campaign provenance"
```

---

## Task 2: Backend — response models for config-change read endpoint

**Files:**
- Modify: `api/models/responses.py` (add after `ObjectiveMetadataResponse` at `:681`)
- Test: covered by Task 4's integration test (models are validated through the endpoint).

- [ ] **Step 1: Add the response models**

Append to `api/models/responses.py` (after the `ObjectiveMetadataResponse` class):

```python
class ConfigChangeEntry(BaseModel):
    timestamp: str
    component: str
    old: Dict[str, Any]
    new: Dict[str, Any]
    iteration: Optional[int] = None


class ConfigChangesResponse(BaseModel):
    changes: List[ConfigChangeEntry]
```

Confirm `Dict`, `List`, `Optional`, `Any` are imported at the top of the file (they are used by existing models; add to the `typing` import line if missing).

- [ ] **Step 2: Verify import compiles**

Run: `~/miniforge3/envs/alchemist-env/bin/python -c "from api.models.responses import ConfigChangesResponse, ConfigChangeEntry; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add api/models/responses.py
git commit -m "feat(api): add ConfigChangesResponse models"
```

---

## Task 3: Backend — log config_changed on model train

**Files:**
- Modify: `api/routers/models.py` (handler `train_model`, `:20`)
- Test: `tests/integration/api/test_config_changes_router.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/api/test_config_changes_router.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


@pytest.fixture
def sid():
    r = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    r.raise_for_status()
    s = r.json()["session_id"]
    client.post(f"/api/v1/sessions/{s}/variables",
                json={"name": "x", "type": "real", "min": 0.0, "max": 10.0})
    # seed a couple experiments so training is possible
    client.post(f"/api/v1/sessions/{s}/experiments",
                json={"inputs": {"x": 1.0}, "output": 2.0})
    client.post(f"/api/v1/sessions/{s}/experiments",
                json={"inputs": {"x": 3.0}, "output": 4.0})
    yield s
    client.delete(f"/api/v1/sessions/{s}")


def test_train_logs_config_changed(sid):
    r = client.post(f"/api/v1/sessions/{sid}/model/train",
                    json={"backend": "sklearn", "kernel": "Matern"})
    r.raise_for_status()
    g = client.get(f"/api/v1/sessions/{sid}/audit/config-changes")
    g.raise_for_status()
    changes = g.json()["changes"]
    model_changes = [c for c in changes if c["component"] == "model"]
    assert len(model_changes) >= 1
    assert model_changes[-1]["new"]["kernel"] == "Matern"
    assert model_changes[-1]["new"]["backend"] == "sklearn"
    assert model_changes[-1]["timestamp"]
```

(Note: this test also exercises Task 4's endpoint. Run it after Task 4; for Task 3, first confirm the audit entry exists via the direct assertion in Step 2.)

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_config_changes_router.py::test_train_logs_config_changed -v`
Expected: FAIL — 404 on `GET .../audit/config-changes` (endpoint added in Task 4) OR empty changes list. Either failure is acceptable at this step; it confirms nothing logs yet.

- [ ] **Step 3: Add config-change logging in `train_model`**

In `api/routers/models.py`, inside `train_model`, immediately AFTER the successful `session.train_model(...)` call (after `models.py:48`), add:

```python
    try:
        session.audit_log.log_config_change(
            component="model",
            old={},
            new={
                "backend": request.backend,
                "kernel": request.kernel,
                "kernel_params": request.kernel_params,
            },
            iteration=len(session.audit_log.get_entries("config_changed")),
        )
    except Exception as e:  # never block a successful train on audit failure
        logger.warning(f"Failed to audit model config change: {e}", exc_info=True)
```

Ensure `logger` is defined at module top (mirror other routers: `import logging; logger = logging.getLogger(__name__)`). Add it if absent.

- [ ] **Step 4: (defer full run to Task 4)** — proceed to Task 4, which adds the read endpoint; the test runs green there.

- [ ] **Step 5: Commit**

```bash
git add api/routers/models.py tests/integration/api/test_config_changes_router.py
git commit -m "feat(api): audit config_changed on model train"
```

---

## Task 4: Backend — GET config-changes endpoint + acquisition logging

**Files:**
- Modify: `api/routers/experiments.py` (add endpoint near the objective-metadata handlers at `:898`)
- Modify: `api/routers/acquisition.py` (handler `suggest_next_experiments`, after the existing `lock_acquisition` call `:80`)
- Test: `tests/integration/api/test_config_changes_router.py` (extend)

- [ ] **Step 1: Write the failing test (acquisition + endpoint shape)**

Append to `tests/integration/api/test_config_changes_router.py`:

```python
def test_suggest_logs_config_changed_and_endpoint_shape(sid):
    client.post(f"/api/v1/sessions/{sid}/model/train",
                json={"backend": "sklearn", "kernel": "Matern"})
    r = client.post(f"/api/v1/sessions/{sid}/acquisition/suggest",
                    json={"strategy": "EI", "goal": "maximize", "n_suggestions": 1})
    r.raise_for_status()
    g = client.get(f"/api/v1/sessions/{sid}/audit/config-changes")
    g.raise_for_status()
    body = g.json()
    assert "changes" in body
    acq = [c for c in body["changes"] if c["component"] == "acquisition"]
    assert len(acq) >= 1
    assert acq[-1]["new"]["strategy"] == "EI"
    # every entry has the required shape
    for c in body["changes"]:
        assert set(["timestamp", "component", "old", "new"]).issubset(c.keys())


def test_config_changes_empty_for_new_session(sid):
    g = client.get(f"/api/v1/sessions/{sid}/audit/config-changes")
    g.raise_for_status()
    assert g.json() == {"changes": []}
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_config_changes_router.py -v`
Expected: FAIL — 404 (endpoint missing) / no acquisition entry.

- [ ] **Step 3a: Add the read endpoint**

In `api/routers/experiments.py`, add near the objective-metadata handlers (around `:908`). Ensure the response models are imported at top: `from ..models.responses import ConfigChangesResponse, ConfigChangeEntry` (add to the existing responses import).

```python
@router.get("/{session_id}/audit/config-changes", response_model=ConfigChangesResponse)
async def get_config_changes(session_id: str, session=Depends(get_session)):
    """Return timestamped mid-campaign optimizer-config changes (provenance)."""
    entries = session.audit_log.get_entries("config_changed")
    changes = [
        ConfigChangeEntry(
            timestamp=e.timestamp,
            component=e.parameters.get("component", ""),
            old=e.parameters.get("old", {}),
            new=e.parameters.get("new", {}),
            iteration=e.parameters.get("iteration"),
        )
        for e in entries
    ]
    return ConfigChangesResponse(changes=changes)
```

- [ ] **Step 3b: Add acquisition config logging**

In `api/routers/acquisition.py`, inside `suggest_next_experiments`, immediately AFTER the existing `session.audit_log.lock_acquisition(...)` call (after `:86`), add:

```python
    try:
        session.audit_log.log_config_change(
            component="acquisition",
            old={},
            new={
                "strategy": request.strategy,
                "goal": request.goal,
                "xi": request.xi,
                "kappa": request.kappa,
            },
            iteration=len(session.audit_log.get_entries("config_changed")),
        )
    except Exception as e:
        logger.warning(f"Failed to audit acquisition config change: {e}", exc_info=True)
```

Ensure `logger` exists at module top of `acquisition.py` (add `import logging; logger = logging.getLogger(__name__)` if absent).

- [ ] **Step 4: Run the full file to verify pass**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_config_changes_router.py -v`
Expected: PASS (all 4 tests: train, suggest+shape, empty, and Task 3's train test).

- [ ] **Step 5: Commit**

```bash
git add api/routers/experiments.py api/routers/acquisition.py tests/integration/api/test_config_changes_router.py
git commit -m "feat(api): GET config-changes endpoint + audit acquisition config"
```

---

## Task 5: Backend — regression guard for objective_label_changed

**Files:**
- Test: `tests/integration/api/test_config_changes_router.py` (extend)

- [ ] **Step 1: Write the test**

```python
def test_objective_label_change_still_logs_separately(sid):
    client.put(f"/api/v1/sessions/{sid}/objective-metadata",
               json={"metadata": {"Output": {"label": "area_x", "unit": "a.u."}}})
    # objective_label_changed is a distinct type; must NOT appear in config-changes
    g = client.get(f"/api/v1/sessions/{sid}/audit/config-changes")
    g.raise_for_status()
    assert all(c["component"] != "objective_label" for c in g.json()["changes"])
```

- [ ] **Step 2: Run to verify it passes immediately** (no code change; asserts separation)

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_config_changes_router.py::test_objective_label_change_still_logs_separately -v`
Expected: PASS

- [ ] **Step 3: Run the full backend suite to confirm no regressions**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/ -q`
Expected: all pass (pre-existing green suite + new tests).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/api/test_config_changes_router.py
git commit -m "test(api): guard objective_label_changed stays distinct from config_changed"
```

---

## Task 6: Frontend — queue + config-changes API endpoints & types

**Files:**
- Create: `alchemist-web/src/api/endpoints/queue.ts`
- Test: `alchemist-web/src/api/endpoints/queue.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// alchemist-web/src/api/endpoints/queue.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest';
import apiClient from '../client';
import { getQueue, getConfigChanges } from './queue';

vi.mock('../client');

describe('queue endpoints', () => {
  beforeEach(() => vi.clearAllMocks());

  it('getQueue calls the queue path and returns data', async () => {
    (apiClient.get as any) = vi.fn().mockResolvedValue({
      data: { items: [], n_pending: 0, n_running: 0, n_done: 0, n_failed: 0 },
    });
    const res = await getQueue('sess1');
    expect(apiClient.get).toHaveBeenCalledWith('/sessions/sess1/experiments/queue');
    expect(res.n_pending).toBe(0);
  });

  it('getConfigChanges calls the config-changes path', async () => {
    (apiClient.get as any) = vi.fn().mockResolvedValue({ data: { changes: [] } });
    const res = await getConfigChanges('sess1');
    expect(apiClient.get).toHaveBeenCalledWith('/sessions/sess1/audit/config-changes');
    expect(res.changes).toEqual([]);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run (in `alchemist-web/`): `npm test -- src/api/endpoints/queue.test.ts`
Expected: FAIL — cannot resolve `./queue`.

- [ ] **Step 3: Create the endpoint module**

```typescript
// alchemist-web/src/api/endpoints/queue.ts
import apiClient from '../client';

export type QueueStatus = 'pending' | 'running' | 'done' | 'failed';

export interface QueueItem {
  id: string;
  inputs: Record<string, unknown>;
  reason: string | null;
  status: QueueStatus;
  output: number | number[] | null;
  noise: number | number[] | null;
  error: string | null;
  dataset_ref: number | null;
  staged_at: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface QueueListResponse {
  items: QueueItem[];
  n_pending: number;
  n_running: number;
  n_done: number;
  n_failed: number;
}

export interface ConfigChangeEntry {
  timestamp: string;
  component: string;
  old: Record<string, unknown>;
  new: Record<string, unknown>;
  iteration: number | null;
}

export interface ConfigChangesResponse {
  changes: ConfigChangeEntry[];
}

export async function getQueue(sessionId: string): Promise<QueueListResponse> {
  const res = await apiClient.get<QueueListResponse>(
    `/sessions/${sessionId}/experiments/queue`
  );
  return res.data;
}

export async function getConfigChanges(sessionId: string): Promise<ConfigChangesResponse> {
  const res = await apiClient.get<ConfigChangesResponse>(
    `/sessions/${sessionId}/audit/config-changes`
  );
  return res.data;
}
```

- [ ] **Step 4: Run to verify it passes**

Run (in `alchemist-web/`): `npm test -- src/api/endpoints/queue.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add alchemist-web/src/api/endpoints/queue.ts alchemist-web/src/api/endpoints/queue.test.ts
git commit -m "feat(web): queue + config-changes API endpoints and types"
```

---

## Task 7: Frontend — react-query test helper

**Files:**
- Create: `alchemist-web/src/test/queryWrapper.tsx`

- [ ] **Step 1: Create the helper (no separate test — it is test infra, exercised by later tasks)**

```tsx
// alchemist-web/src/test/queryWrapper.tsx
import { ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, RenderOptions } from '@testing-library/react';

export function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
}

export function renderWithQuery(ui: ReactNode, options?: RenderOptions) {
  const client = createTestQueryClient();
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
    options
  );
}
```

- [ ] **Step 2: Verify it type-checks by importing in a scratch test**

Run (in `alchemist-web/`): `npx tsc --noEmit`
Expected: no new errors from `queryWrapper.tsx`.

- [ ] **Step 3: Commit**

```bash
git add alchemist-web/src/test/queryWrapper.tsx
git commit -m "test(web): add react-query render helper"
```

---

## Task 8: Frontend — useExperimentQueue / useConfigChanges hooks

**Files:**
- Create: `alchemist-web/src/hooks/api/useQueue.ts`
- Test: `alchemist-web/src/hooks/api/useQueue.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// alchemist-web/src/hooks/api/useQueue.test.tsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as queueApi from '../../api/endpoints/queue';
import { useExperimentQueue } from './useQueue';

function wrapper() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

describe('useExperimentQueue', () => {
  beforeEach(() => vi.restoreAllMocks());

  it('fetches the queue for a session', async () => {
    vi.spyOn(queueApi, 'getQueue').mockResolvedValue({
      items: [], n_pending: 2, n_running: 0, n_done: 0, n_failed: 0,
    });
    const { result } = renderHook(() => useExperimentQueue('sess1'), { wrapper: wrapper() });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.n_pending).toBe(2);
    expect(queueApi.getQueue).toHaveBeenCalledWith('sess1');
  });

  it('is disabled when sessionId is null', () => {
    const { result } = renderHook(() => useExperimentQueue(null), { wrapper: wrapper() });
    expect(result.current.fetchStatus).toBe('idle');
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run (in `alchemist-web/`): `npm test -- src/hooks/api/useQueue.test.tsx`
Expected: FAIL — cannot resolve `./useQueue`.

- [ ] **Step 3: Create the hooks**

```typescript
// alchemist-web/src/hooks/api/useQueue.ts
import { useQuery, UseQueryResult } from '@tanstack/react-query';
import * as queueApi from '../../api/endpoints/queue';
import type { QueueListResponse, ConfigChangesResponse } from '../../api/endpoints/queue';

export function useExperimentQueue(
  sessionId: string | null,
  enabled = true
): UseQueryResult<QueueListResponse> {
  return useQuery({
    queryKey: ['experiments-queue', sessionId],
    queryFn: () => queueApi.getQueue(sessionId!),
    enabled: enabled && !!sessionId,
    refetchOnWindowFocus: false,
  });
}

export function useConfigChanges(
  sessionId: string | null,
  enabled = true
): UseQueryResult<ConfigChangesResponse> {
  return useQuery({
    queryKey: ['config-changes', sessionId],
    queryFn: () => queueApi.getConfigChanges(sessionId!),
    enabled: enabled && !!sessionId,
    refetchOnWindowFocus: false,
  });
}
```

- [ ] **Step 4: Run to verify it passes**

Run (in `alchemist-web/`): `npm test -- src/hooks/api/useQueue.test.tsx`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add alchemist-web/src/hooks/api/useQueue.ts alchemist-web/src/hooks/api/useQueue.test.tsx
git commit -m "feat(web): useExperimentQueue + useConfigChanges hooks"
```

---

## Task 9: Frontend — wire queue events into useSessionEvents

**Files:**
- Modify: `alchemist-web/src/hooks/useSessionEvents.ts` (add branches after `model_trained` at `:123`)
- Test: `alchemist-web/src/hooks/useSessionEvents.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// alchemist-web/src/hooks/useSessionEvents.test.tsx
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useSessionEvents } from './useSessionEvents';

class MockWS {
  static instances: MockWS[] = [];
  onopen: any; onmessage: any; onerror: any; onclose: any;
  close = vi.fn();
  constructor(public url: string) { MockWS.instances.push(this); }
  emit(data: any) { this.onmessage?.({ data: JSON.stringify(data) }); }
}

describe('useSessionEvents queue handling', () => {
  let client: QueryClient;
  beforeEach(() => {
    (globalThis as any).WebSocket = MockWS as any;
    MockWS.instances = [];
    client = new QueryClient();
  });
  afterEach(() => vi.restoreAllMocks());

  function wrapper() {
    return ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  }

  it('invalidates the queue query on queue_item_updated', () => {
    const spy = vi.spyOn(client, 'invalidateQueries');
    renderHook(() => useSessionEvents('sess1'), { wrapper: wrapper() });
    MockWS.instances[0].emit({ event: 'queue_item_updated', item_id: 'a', status: 'running' });
    expect(spy).toHaveBeenCalledWith({ queryKey: ['experiments-queue', 'sess1'] });
  });

  it('invalidates the queue query on queue_updated', () => {
    const spy = vi.spyOn(client, 'invalidateQueries');
    renderHook(() => useSessionEvents('sess1'), { wrapper: wrapper() });
    MockWS.instances[0].emit({ event: 'queue_updated' });
    expect(spy).toHaveBeenCalledWith({ queryKey: ['experiments-queue', 'sess1'] });
  });

  it('resyncs the queue after a reconnect (onclose then reopen invalidates)', () => {
    vi.useFakeTimers();
    const spy = vi.spyOn(client, 'invalidateQueries');
    renderHook(() => useSessionEvents('sess1'), { wrapper: wrapper() });
    // simulate a drop: onclose schedules a 5s reconnect that opens a new socket
    MockWS.instances[0].onclose?.();
    vi.advanceTimersByTime(5000);
    // new socket exists; a queue event on it must still invalidate (resync path)
    const latest = MockWS.instances[MockWS.instances.length - 1];
    latest.onopen?.();
    latest.emit({ event: 'queue_updated' });
    expect(spy).toHaveBeenCalledWith({ queryKey: ['experiments-queue', 'sess1'] });
    vi.useRealTimers();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run (in `alchemist-web/`): `npm test -- src/hooks/useSessionEvents.test.tsx`
Expected: FAIL — `invalidateQueries` not called with the queue key.

- [ ] **Step 3: Add the event branches**

In `alchemist-web/src/hooks/useSessionEvents.ts`, inside the `ws.onmessage` handler, add two `else if` branches immediately after the `model_trained` block (after line 123, before the closing `}` of the `try`):

```typescript
          } else if (data.event === 'queue_item_updated') {
            queryClient.invalidateQueries({ queryKey: ['experiments-queue', sessionId] });

          } else if (data.event === 'queue_updated') {
            queryClient.invalidateQueries({ queryKey: ['experiments-queue', sessionId] });
```

Also broaden `model_trained` to refresh the live plots — replace the two existing invalidations in the `model_trained` branch (`:117-118`) with:

```typescript
            queryClient.invalidateQueries({ queryKey: ['model-info', sessionId] });
            queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
            queryClient.invalidateQueries({ queryKey: ['parity-data', sessionId] });
            queryClient.invalidateQueries({ queryKey: ['metrics-data', sessionId] });
            queryClient.invalidateQueries({ queryKey: ['calibration-curve', sessionId] });
            queryClient.invalidateQueries({ queryKey: ['qq-plot', sessionId] });
            queryClient.invalidateQueries({ queryKey: ['hyperparameters', sessionId] });
```

(Verify these query-key prefixes against `hooks/api/useVisualizations.ts`; adjust the string literals to match the actual keys used there — e.g. `['parity-data', ...]`, `['metrics-data', ...]`. Invalidating by prefix key matches all param variants.)

- [ ] **Step 4: Run to verify it passes**

Run (in `alchemist-web/`): `npm test -- src/hooks/useSessionEvents.test.tsx`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add alchemist-web/src/hooks/useSessionEvents.ts alchemist-web/src/hooks/useSessionEvents.test.tsx
git commit -m "feat(web): invalidate queue + plot queries on work-queue and model events"
```

---

## Task 10: Frontend — QueueTimeline component

**Files:**
- Create: `alchemist-web/src/features/monitoring/QueueTimeline.tsx`
- Test: `alchemist-web/src/features/monitoring/QueueTimeline.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// alchemist-web/src/features/monitoring/QueueTimeline.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithQuery } from '../../test/queryWrapper';
import * as useQueue from '../../hooks/api/useQueue';
import { QueueTimeline } from './QueueTimeline';

function mockQueue(items: any[]) {
  vi.spyOn(useQueue, 'useExperimentQueue').mockReturnValue({
    data: { items, n_pending: 0, n_running: 0, n_done: 0, n_failed: 0 },
    isLoading: false, isSuccess: true,
  } as any);
}

describe('QueueTimeline', () => {
  it('shows an empty/awaiting state when there are no items', () => {
    mockQueue([]);
    renderWithQuery(<QueueTimeline sessionId="s1" />);
    expect(screen.getByText(/awaiting controller/i)).toBeInTheDocument();
  });

  it('renders each item status and a failed item error', () => {
    mockQueue([
      { id: 'a', inputs: { x: 1 }, reason: 'seed', status: 'done', output: 2, error: null,
        noise: null, dataset_ref: 0, staged_at: null, started_at: null, completed_at: null },
      { id: 'b', inputs: { x: 3 }, reason: null, status: 'failed', output: null,
        error: 'sensor timeout', noise: null, dataset_ref: null,
        staged_at: null, started_at: null, completed_at: null },
    ]);
    renderWithQuery(<QueueTimeline sessionId="s1" />);
    expect(screen.getByText(/done/i)).toBeInTheDocument();
    expect(screen.getByText(/failed/i)).toBeInTheDocument();
    expect(screen.getByText(/sensor timeout/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run (in `alchemist-web/`): `npm test -- src/features/monitoring/QueueTimeline.test.tsx`
Expected: FAIL — cannot resolve `./QueueTimeline`.

- [ ] **Step 3: Create the component**

```tsx
// alchemist-web/src/features/monitoring/QueueTimeline.tsx
import { useExperimentQueue } from '../../hooks/api/useQueue';
import type { QueueItem, QueueStatus } from '../../api/endpoints/queue';

const STATUS_LABEL: Record<QueueStatus, string> = {
  pending: 'Pending',
  running: 'Running',
  done: 'Done',
  failed: 'Failed',
};

function formatInputs(inputs: Record<string, unknown>): string {
  return Object.entries(inputs)
    .map(([k, v]) => `${k}=${typeof v === 'number' ? v : String(v)}`)
    .join(', ');
}

export function QueueTimeline({ sessionId }: { sessionId: string }) {
  const { data, isLoading } = useExperimentQueue(sessionId);

  if (isLoading) return <div className="text-sm text-muted-foreground">Loading queue…</div>;

  const items: QueueItem[] = data?.items ?? [];
  if (items.length === 0) {
    return (
      <div className="text-sm text-muted-foreground py-6 text-center">
        No items yet — awaiting controller.
      </div>
    );
  }

  return (
    <ul className="space-y-1">
      {items.map((item) => (
        <li key={item.id} className="flex flex-col gap-0.5 rounded border p-2 text-sm">
          <div className="flex items-center justify-between">
            <span className="font-mono">{formatInputs(item.inputs)}</span>
            <span data-status={item.status}>{STATUS_LABEL[item.status]}</span>
          </div>
          {item.reason && <span className="text-xs text-muted-foreground">{item.reason}</span>}
          {item.status === 'done' && item.output != null && (
            <span className="text-xs">objective: {String(item.output)}</span>
          )}
          {item.status === 'failed' && item.error && (
            <span className="text-xs text-red-600">error: {item.error}</span>
          )}
        </li>
      ))}
    </ul>
  );
}
```

- [ ] **Step 4: Run to verify it passes**

Run (in `alchemist-web/`): `npm test -- src/features/monitoring/QueueTimeline.test.tsx`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add alchemist-web/src/features/monitoring/QueueTimeline.tsx alchemist-web/src/features/monitoring/QueueTimeline.test.tsx
git commit -m "feat(web): QueueTimeline live work-queue view"
```

---

## Task 11: Frontend — ObjectiveTrace component

**Files:**
- Create: `alchemist-web/src/features/monitoring/ObjectiveTrace.tsx`
- Test: `alchemist-web/src/features/monitoring/ObjectiveTrace.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// alchemist-web/src/features/monitoring/ObjectiveTrace.test.tsx
import { describe, it, expect } from 'vitest';
import { computeTrace } from './ObjectiveTrace';
import type { QueueItem } from '../../api/endpoints/queue';

function item(id: string, status: any, output: number | null): QueueItem {
  return { id, inputs: {}, reason: null, status, output, error: null, noise: null,
    dataset_ref: null, staged_at: null, started_at: null,
    completed_at: status === 'done' ? `2026-07-28T00:00:0${id}` : null };
}

describe('computeTrace', () => {
  it('includes only done items, ordered by completion, with cumulative best (maximize)', () => {
    const items = [item('1', 'done', 5), item('2', 'failed', null), item('3', 'done', 8), item('4', 'pending', null)];
    const trace = computeTrace(items, 'maximize');
    expect(trace.map(p => p.value)).toEqual([5, 8]);
    expect(trace.map(p => p.best)).toEqual([5, 8]);
  });

  it('cumulative best for minimize keeps the lowest so far', () => {
    const items = [item('1', 'done', 8), item('2', 'done', 5), item('3', 'done', 9)];
    const trace = computeTrace(items, 'minimize');
    expect(trace.map(p => p.best)).toEqual([8, 5, 5]);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run (in `alchemist-web/`): `npm test -- src/features/monitoring/ObjectiveTrace.test.tsx`
Expected: FAIL — cannot resolve `./ObjectiveTrace` / `computeTrace`.

- [ ] **Step 3: Create the component + exported pure function**

```tsx
// alchemist-web/src/features/monitoring/ObjectiveTrace.tsx
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useExperimentQueue } from '../../hooks/api/useQueue';
import type { QueueItem } from '../../api/endpoints/queue';

export interface TracePoint { index: number; value: number; best: number; }

/** Pure: cumulative-best objective trace over completed items. */
export function computeTrace(items: QueueItem[], goal: 'maximize' | 'minimize'): TracePoint[] {
  const done = items
    .filter((i) => i.status === 'done' && typeof i.output === 'number')
    .sort((a, b) => (a.completed_at ?? '').localeCompare(b.completed_at ?? ''));
  const trace: TracePoint[] = [];
  let best: number | null = null;
  done.forEach((i, idx) => {
    const value = i.output as number;
    if (best === null) best = value;
    else best = goal === 'maximize' ? Math.max(best, value) : Math.min(best, value);
    trace.push({ index: idx + 1, value, best });
  });
  return trace;
}

interface ObjectiveTraceProps {
  sessionId: string;
  goal?: 'maximize' | 'minimize';
  objectiveLabel?: string;
}

export function ObjectiveTrace({ sessionId, goal = 'maximize', objectiveLabel = 'Objective' }: ObjectiveTraceProps) {
  const { data } = useExperimentQueue(sessionId);
  const trace = computeTrace(data?.items ?? [], goal);

  if (trace.length === 0) {
    return <div className="text-sm text-muted-foreground py-6 text-center">No completed experiments yet.</div>;
  }

  return (
    <div>
      <div className="text-xs text-muted-foreground mb-1">{objectiveLabel}</div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={trace}>
          <XAxis dataKey="index" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="value" name="per-experiment" dot />
          <Line type="stepAfter" dataKey="best" name="best so far" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 4: Run to verify it passes**

Run (in `alchemist-web/`): `npm test -- src/features/monitoring/ObjectiveTrace.test.tsx`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add alchemist-web/src/features/monitoring/ObjectiveTrace.tsx alchemist-web/src/features/monitoring/ObjectiveTrace.test.tsx
git commit -m "feat(web): ObjectiveTrace with cumulative-best envelope"
```

---

## Task 12: Frontend — ConfigChangeTimeline component

**Files:**
- Create: `alchemist-web/src/features/monitoring/ConfigChangeTimeline.tsx`
- Test: `alchemist-web/src/features/monitoring/ConfigChangeTimeline.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// alchemist-web/src/features/monitoring/ConfigChangeTimeline.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithQuery } from '../../test/queryWrapper';
import * as useQueue from '../../hooks/api/useQueue';
import { ConfigChangeTimeline } from './ConfigChangeTimeline';

describe('ConfigChangeTimeline', () => {
  it('renders each config change with component and iteration', () => {
    vi.spyOn(useQueue, 'useConfigChanges').mockReturnValue({
      data: { changes: [
        { timestamp: '2026-07-28T00:00:00', component: 'acquisition',
          old: {}, new: { strategy: 'qEI' }, iteration: 12 },
      ] },
      isLoading: false, isSuccess: true,
    } as any);
    renderWithQuery(<ConfigChangeTimeline sessionId="s1" />);
    expect(screen.getByText(/acquisition/i)).toBeInTheDocument();
    expect(screen.getByText(/qEI/)).toBeInTheDocument();
    expect(screen.getByText(/12/)).toBeInTheDocument();
  });

  it('shows an empty state with no changes', () => {
    vi.spyOn(useQueue, 'useConfigChanges').mockReturnValue({
      data: { changes: [] }, isLoading: false, isSuccess: true,
    } as any);
    renderWithQuery(<ConfigChangeTimeline sessionId="s1" />);
    expect(screen.getByText(/no config changes/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run (in `alchemist-web/`): `npm test -- src/features/monitoring/ConfigChangeTimeline.test.tsx`
Expected: FAIL — cannot resolve `./ConfigChangeTimeline`.

- [ ] **Step 3: Create the component**

```tsx
// alchemist-web/src/features/monitoring/ConfigChangeTimeline.tsx
import { useConfigChanges } from '../../hooks/api/useQueue';

export function ConfigChangeTimeline({ sessionId }: { sessionId: string }) {
  const { data, isLoading } = useConfigChanges(sessionId);
  if (isLoading) return <div className="text-sm text-muted-foreground">Loading provenance…</div>;

  const changes = data?.changes ?? [];
  if (changes.length === 0) {
    return <div className="text-sm text-muted-foreground py-4">No config changes recorded.</div>;
  }

  return (
    <ul className="space-y-1">
      {changes.map((c, i) => (
        <li key={`${c.timestamp}-${i}`} className="rounded border p-2 text-sm">
          <div className="flex items-center justify-between">
            <span className="font-medium">{c.component}</span>
            <span className="text-xs text-muted-foreground">
              {c.iteration != null ? `iteration ${c.iteration}` : ''} · {c.timestamp}
            </span>
          </div>
          <div className="text-xs font-mono">
            {JSON.stringify(c.old)} → {JSON.stringify(c.new)}
          </div>
        </li>
      ))}
    </ul>
  );
}
```

- [ ] **Step 4: Run to verify it passes**

Run (in `alchemist-web/`): `npm test -- src/features/monitoring/ConfigChangeTimeline.test.tsx`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add alchemist-web/src/features/monitoring/ConfigChangeTimeline.tsx alchemist-web/src/features/monitoring/ConfigChangeTimeline.test.tsx
git commit -m "feat(web): ConfigChangeTimeline provenance view"
```

---

## Task 13: Frontend — tab components (Config / Live / History)

**Files:**
- Create: `alchemist-web/src/features/monitoring/tabs/ConfigTab.tsx`
- Create: `alchemist-web/src/features/monitoring/tabs/LiveTab.tsx`
- Create: `alchemist-web/src/features/monitoring/tabs/HistoryTab.tsx`
- Test: `alchemist-web/src/features/monitoring/tabs/tabs.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// alchemist-web/src/features/monitoring/tabs/tabs.test.tsx
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
```

- [ ] **Step 2: Run to verify it fails**

Run (in `alchemist-web/`): `npm test -- src/features/monitoring/tabs/tabs.test.tsx`
Expected: FAIL — cannot resolve `./ConfigTab`.

- [ ] **Step 3: Create the three tab components**

```tsx
// alchemist-web/src/features/monitoring/tabs/ConfigTab.tsx
import { VariablesPanel } from '../../../features/variables/VariablesPanel';
import { GPRPanel } from '../../../features/models/GPRPanel';
import { AcquisitionPanel } from '../../../features/acquisition/AcquisitionPanel';
import { InitialDesignPanel } from '../../../features/experiments/InitialDesignPanel';

export function ConfigTab({ sessionId, isRunning }: { sessionId: string; isRunning: boolean }) {
  return (
    <div className="space-y-4">
      {isRunning && (
        <div className="rounded border border-amber-300 bg-amber-50 p-2 text-sm text-amber-900">
          Applies to the next suggestion the controller requests — ALchemist does not initiate cycles.
        </div>
      )}
      <VariablesPanel sessionId={sessionId} />
      <GPRPanel sessionId={sessionId} />
      <AcquisitionPanel sessionId={sessionId} />
      <InitialDesignPanel sessionId={sessionId} />
    </div>
  );
}
```

```tsx
// alchemist-web/src/features/monitoring/tabs/LiveTab.tsx
import { QueueTimeline } from '../QueueTimeline';
import { ObjectiveTrace } from '../ObjectiveTrace';
import { MetricsPlot } from '../../../components/visualizations/MetricsPlot';
import { ParityPlot } from '../../../components/visualizations/ParityPlot';

interface LiveTabProps {
  sessionId: string;
  objectiveLabel: string;
  goal: 'maximize' | 'minimize';
}

export function LiveTab({ sessionId, objectiveLabel, goal }: LiveTabProps) {
  return (
    <div className="space-y-4">
      <section>
        <h3 className="text-sm font-semibold mb-2">Work Queue</h3>
        <QueueTimeline sessionId={sessionId} />
      </section>
      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <MetricsPlot sessionId={sessionId} selectedMetric="R2" cvSplits={5} />
        <ParityPlot sessionId={sessionId} useCalibrated={false} sigmaMultiplier="2" />
      </section>
      <section>
        <h3 className="text-sm font-semibold mb-2">Objective so far</h3>
        <ObjectiveTrace sessionId={sessionId} goal={goal} objectiveLabel={objectiveLabel} />
      </section>
    </div>
  );
}
```

```tsx
// alchemist-web/src/features/monitoring/tabs/HistoryTab.tsx
import { ConfigChangeTimeline } from '../ConfigChangeTimeline';
import { CalibrationCurve } from '../../../components/visualizations/CalibrationCurve';
import { QQPlot } from '../../../components/visualizations/QQPlot';
import { HyperparametersDisplay } from '../../../components/visualizations/HyperparametersDisplay';

export function HistoryTab({ sessionId }: { sessionId: string }) {
  return (
    <div className="space-y-4">
      <section>
        <h3 className="text-sm font-semibold mb-2">Config change provenance</h3>
        <ConfigChangeTimeline sessionId={sessionId} />
      </section>
      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <CalibrationCurve sessionId={sessionId} useCalibrated={false} />
        <QQPlot sessionId={sessionId} useCalibrated={false} />
      </section>
      <HyperparametersDisplay sessionId={sessionId} />
    </div>
  );
}
```

Note: `MetricsPlot`, `ParityPlot`, `CalibrationCurve`, `QQPlot`, `HyperparametersDisplay` are **named exports** — import with braces. `GPRPanel` requires a `VisualizationProvider` in the real app; ensure the app already wraps monitor mode in it (it does at the app root) — if not, wrap the ConfigTab GPRPanel accordingly in Task 14.

- [ ] **Step 4: Run to verify it passes**

Run (in `alchemist-web/`): `npm test -- src/features/monitoring/tabs/tabs.test.tsx`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add alchemist-web/src/features/monitoring/tabs/
git commit -m "feat(web): Config/Live/History monitor tabs"
```

---

## Task 14: Frontend — LiveMonitor shell (tabs + header + ?tab=)

**Files:**
- Create: `alchemist-web/src/features/monitoring/LiveMonitor.tsx`
- Test: `alchemist-web/src/features/monitoring/LiveMonitor.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// alchemist-web/src/features/monitoring/LiveMonitor.test.tsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithQuery } from '../../test/queryWrapper';

vi.mock('./tabs/ConfigTab', () => ({ ConfigTab: () => <div>CONFIG TAB</div> }));
vi.mock('./tabs/LiveTab', () => ({ LiveTab: () => <div>LIVE TAB</div> }));
vi.mock('./tabs/HistoryTab', () => ({ HistoryTab: () => <div>HISTORY TAB</div> }));

import { LiveMonitor } from './LiveMonitor';

describe('LiveMonitor', () => {
  beforeEach(() => { window.history.replaceState({}, '', '/'); });

  it('defaults to the Live tab', () => {
    renderWithQuery(<LiveMonitor sessionId="s1" />);
    expect(screen.getByText('LIVE TAB')).toBeInTheDocument();
  });

  it('honors ?tab=config', () => {
    window.history.replaceState({}, '', '/?tab=config');
    renderWithQuery(<LiveMonitor sessionId="s1" />);
    expect(screen.getByText('CONFIG TAB')).toBeInTheDocument();
  });

  it('switches tabs on click', async () => {
    renderWithQuery(<LiveMonitor sessionId="s1" />);
    await userEvent.click(screen.getByRole('tab', { name: /history/i }));
    expect(screen.getByText('HISTORY TAB')).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run (in `alchemist-web/`): `npm test -- src/features/monitoring/LiveMonitor.test.tsx`
Expected: FAIL — cannot resolve `./LiveMonitor`.

- [ ] **Step 3: Create the shell**

```tsx
// alchemist-web/src/features/monitoring/LiveMonitor.tsx
import { useMemo, useState } from 'react';
import { ConfigTab } from './tabs/ConfigTab';
import { LiveTab } from './tabs/LiveTab';
import { HistoryTab } from './tabs/HistoryTab';
import { useExperimentQueue } from '../../hooks/api/useQueue';
import { useObjectiveMetadata } from '../../hooks/api/useObjectiveMetadata';

type TabKey = 'config' | 'live' | 'history';
const TABS: { key: TabKey; label: string }[] = [
  { key: 'config', label: 'Config' },
  { key: 'live', label: 'Live' },
  { key: 'history', label: 'History' },
];

function initialTab(): TabKey {
  const t = new URLSearchParams(window.location.search).get('tab');
  return t === 'config' || t === 'history' ? t : 'live';
}

export function LiveMonitor({ sessionId }: { sessionId: string }) {
  const [tab, setTab] = useState<TabKey>(initialTab);
  const { data: queue } = useExperimentQueue(sessionId);
  const isRunning = (queue?.n_running ?? 0) > 0;

  // Opaque objective label/unit for display (never interpreted).
  const objMeta = useObjectiveMetadata(sessionId);
  const objectiveLabel = useMemo(() => {
    const map = objMeta.data?.metadata ?? {};
    const first = Object.values(map)[0] as { label?: string; unit?: string } | undefined;
    if (!first?.label) return 'Objective';
    return first.unit ? `${first.label} (${first.unit})` : first.label;
  }, [objMeta.data]);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between border-b px-4 py-2">
        <div className="flex gap-1" role="tablist">
          {TABS.map((t) => (
            <button
              key={t.key}
              role="tab"
              aria-selected={tab === t.key}
              onClick={() => setTab(t.key)}
              className={`px-3 py-1 text-sm rounded ${tab === t.key ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'}`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="text-xs text-muted-foreground">
          objective: <span className="font-medium">{objectiveLabel}</span>
          {isRunning ? ' · running' : ' · idle'}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        {tab === 'config' && <ConfigTab sessionId={sessionId} isRunning={isRunning} />}
        {tab === 'live' && <LiveTab sessionId={sessionId} objectiveLabel={objectiveLabel} goal="maximize" />}
        {tab === 'history' && <HistoryTab sessionId={sessionId} />}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Handle the objective-metadata hook dependency**

If `hooks/api/useObjectiveMetadata.ts` does not already exist, create it (mirrors the queue hook, calling a `getObjectiveMetadata` endpoint):

```typescript
// alchemist-web/src/hooks/api/useObjectiveMetadata.ts
import { useQuery, UseQueryResult } from '@tanstack/react-query';
import apiClient from '../../api/client';

interface ObjectiveMetadataResponse { metadata: Record<string, { label?: string; unit?: string }>; }

export function useObjectiveMetadata(sessionId: string | null): UseQueryResult<ObjectiveMetadataResponse> {
  return useQuery({
    queryKey: ['objective-metadata', sessionId],
    queryFn: async () => (await apiClient.get<ObjectiveMetadataResponse>(
      `/sessions/${sessionId}/objective-metadata`)).data,
    enabled: !!sessionId,
    refetchOnWindowFocus: false,
  });
}
```

First check whether such a hook/endpoint already exists (grep `objective-metadata` under `alchemist-web/src`); reuse it if present rather than duplicating.

- [ ] **Step 5: Run to verify it passes**

Run (in `alchemist-web/`): `npm test -- src/features/monitoring/LiveMonitor.test.tsx`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add alchemist-web/src/features/monitoring/LiveMonitor.tsx alchemist-web/src/hooks/api/useObjectiveMetadata.ts
git commit -m "feat(web): LiveMonitor tab shell with objective header and ?tab="
```

---

## Task 15: Frontend — swap LiveMonitor into App.tsx

**Files:**
- Modify: `alchemist-web/src/App.tsx` (`:20` import, `:383-385` render gate)

- [ ] **Step 1: Replace the import**

At `App.tsx:20`, replace:

```jsx
import { MonitoringDashboard } from './features/monitoring/MonitoringDashboard';
```

with:

```jsx
import { LiveMonitor } from './features/monitoring/LiveMonitor';
```

- [ ] **Step 2: Replace the render gate**

At `App.tsx:383-385`, replace:

```jsx
{isMonitoringMode && sessionId ? (
  <MonitoringDashboard sessionId={sessionId} pollingInterval={90000} />
) : (
```

with:

```jsx
{isMonitoringMode && sessionId ? (
  <LiveMonitor sessionId={sessionId} />
) : (
```

- [ ] **Step 3: Delete the obsolete dashboard**

```bash
git rm alchemist-web/src/features/monitoring/MonitoringDashboard.tsx
```

Then grep for any remaining references and remove/redirect them:

Run (in `alchemist-web/`): `grep -rn "MonitoringDashboard" src` → expect no results.

- [ ] **Step 4: Typecheck + build + full frontend test**

Run (in `alchemist-web/`): `npx tsc --noEmit && npm test`
Expected: no type errors; all tests pass.

- [ ] **Step 5: Commit**

```bash
git add alchemist-web/src/App.tsx
git commit -m "feat(web): replace MonitoringDashboard with LiveMonitor"
```

---

## Task 16: Full-stack verification pass

**Files:** none (verification only)

- [ ] **Step 1: Backend suite green**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 2: Frontend suite green + typecheck + build**

Run (in `alchemist-web/`): `npx tsc --noEmit && npm test && npm run build`
Expected: no type errors; all tests pass; build succeeds.

- [ ] **Step 3: Manual smoke (optional but recommended)**

Start the API (`~/miniforge3/envs/alchemist-env/bin/python -m uvicorn api.main:app --port 8000`) and the web dev server (`npm run dev` in `alchemist-web/`). Open `?mode=monitor`. Create a session, add variables, train, suggest, then stage/complete a queue item via the API and confirm: the Live tab's queue timeline and objective trace update in real time; the History tab shows a `config_changed` entry after a train + suggest; the Config tab shows the "does not initiate cycles" banner while an item is `running`.

- [ ] **Step 4: Confirm no reactor/domain terms leaked**

Run (in `alchemist-web/src/features/monitoring/` and backend changes): `grep -rniE "reactor|spectro|mqtt|\\bband\\b|detector" src/features/monitoring api/routers/models.py api/routers/acquisition.py` → expect no results.

- [ ] **Step 5: Commit any fixups**

```bash
git add -A && git commit -m "chore: live-monitor verification fixups"
```

---

## Self-Review Notes (for the implementer)

- **Query-key alignment (Task 9):** the plot-invalidation keys are best-effort; before finalizing, open `hooks/api/useVisualizations.ts` and match the exact `queryKey` literal prefixes (`parity-data`, `metrics-data`, etc.). Invalidating by prefix (without the trailing param) matches all variants.
- **GPRPanel provider (Task 13):** `GPRPanel` uses `useVisualization()` — verify monitor mode sits inside the app's `VisualizationProvider`. If the monitor branch renders outside it, wrap `LiveMonitor` (or the ConfigTab GPRPanel) in that provider.
- **Objective-metadata hook (Task 14):** reuse an existing hook if one is already present; only create `useObjectiveMetadata.ts` if not.
- **Goal (maximize/minimize):** the ObjectiveTrace/LiveMonitor default to `maximize`. If the session exposes the current goal, thread it through; otherwise this is a safe display default (does not affect optimization).
