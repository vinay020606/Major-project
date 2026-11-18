# CHANGES — November 2026 Core-FL Fixes

Scope: make the "federated learning" in PulmoScan real on the REST / 3-node path,
containerize the stack for a one-command demo, and document the model-update
lifecycle. Auth / CORS / repo-hygiene changes are **not** in this batch.

## Behavioural changes

| Gap | Before | After | Files |
|---|---|---|---|
| 1 | Edge POSTed `{loss, acc, size_mb, weight_hash}` — no tensors crossed the wire. | Edge POSTs the full `state_dict` as a base64-encoded `torch.save` blob (`weights_b64`) alongside the metrics. Server SHA-256-verifies the payload. | `Model/run_hospital_node.py`, `backend/main.py` |
| 2 | `_execute_central_aggregation` printed the FedAvg formula, slept 2 s, then set `new_acc = prev_acc + random.uniform(1.2, 2.5)`. No tensor arithmetic. | `_fedavg()` performs a real sample-weighted average over each key of each client's `state_dict`. Accuracy is from an actual eval on `Model/data/valid_chest_xray/test` if present, else sample-weighted mean of client-reported metrics. | `backend/fl_orchestrator.py` |
| 3 | Orchestrator wrote `torch.save({version, acc, loss, ...})` — a dict of scalars. `model_inference.py` with `weights_only=True` would crash on load. | Writes `{"model_state": aggregated_state_dict, "meta": {...}}`. Inference loader transparently accepts both wrapper and bare state_dict with `weights_only=False`. | `backend/fl_orchestrator.py`, `backend/model_inference.py` |
| 4 | `opacus.PrivacyEngine` was imported but `make_private` was never called — only `clip_grad_norm_` ran. Reported ε was `0.42·round + U(0.01, 0.05)` (fabricated). | Per-round `privacy_engine.make_private(model, optimizer, data_loader, ...)`. Real ε via `get_epsilon(delta)`. Honest fallback when Opacus is unavailable. | `Model/run_hospital_node.py` |
| 5 | FedProx proximal term was `(μ/2)·‖w‖²` — degenerates to L2 weight decay. | Snapshot `w_global` before local training; proximal term is `(μ/2)·Σ‖wᵢ − w_globalᵢ‖²` over named_parameters. | `Model/run_hospital_node.py` |
| 7 | `node_C` was advertised in docs but only existed in the Flower/gRPC path. REST orchestrator pre-seeded A & B only, aggregation triggered at `received_count ≥ 2`. | `node_C` pre-seeded. Quorum configurable via `FL_MIN_CLIENTS` env (default 3). | `backend/fl_orchestrator.py` |

## New files

- `backend/Dockerfile` — Python 3.11 + torch 2.6 CPU wheels + FastAPI + Opacus.
- `Model/Dockerfile` — same base, runs `run_hospital_node.py` per container.
- `Frontend/Dockerfile` — Node 20 + Vite dev server.
- `docker-compose.yml` — one network, one `backend`, one `frontend`, three `hospital-*` nodes. Backend has a `curl /` healthcheck; nodes wait for it before registering.
- `.dockerignore` — keeps images lean (excludes `.git`, `node_modules`, committed `.pth` weights, etc.).
- `MODEL_UPDATES.md` — full FL lifecycle doc with ASCII timing diagram.
- `ARCHITECTURE.md` — one-page tour of the whole stack: services, request paths, data layout, REST surface, care-plan schema, and where to look when things break.

## Care-plan additions (new feature)

- `backend/stages.json` — rewritten per stage with 7 structured sections:
  `immediate_care`, `medication` (regimen/dosing/alternatives/adjuncts/watch_for/resistance_check),
  `isolation`, `nutrition`, `follow_up` (visits/labs/sputum/imaging/contact_tracing/rehab/endpoint),
  `red_flags`, `patient_education`. WHO-aligned content (3HP for LTBI, HRZE 2+4 for active disease, extended continuation for CNS/disseminated).
- `backend/main.py` — new `GET /care-plan/{stage_id}` endpoint and `/predict` now embeds a `care_plan` field for TB-positive stages. Standard disclaimer attached to every plan.
- `Frontend/src/components/StageCarePanel.jsx` — rewritten to render six cards (medication, isolation+immediate care, nutrition, follow-up schedule, red flags, patient education) plus severity + infectious chips and the backend disclaimer. Falls back to local stage data if backend didn't return a plan.
- `Frontend/src/pages/ResultPage.jsx` — threads `care_plan` through from `/predict` payload.

## Why Docker

- The host had Python 3.15.2 installed; PyTorch 2.6 wheels don't exist for 3.15.
  Pinning Python 3.11 in the image sidesteps this and makes "lightweight"
  real — users only need Docker, not a tuned Python+Node+Torch environment.
- Three hospital containers on one bridge network reproduce the 3-laptop
  topology without three laptops. The inter-container network (`pulmoscan`
  bridge) replaces the LAN and matches the docstring "3-Laptop Physical LAN".

## Not in this batch (next up)

- Gap 6 TLS on LAN REST (nodes talk to backend over the Docker bridge today).
- Gap 8 per-round quorum timeout + late-joiner handling.
- Gap 9 round-state persistence across backend restarts.
- Gap 10 wire `ActiveLearningDataset` into `run_hospital_node.py` (currently only the Flower/gRPC path consumes it).
- Gap 12 implement coordinate-wise median on the REST path.
- Gap 13 Dirichlet non-IID partition script.
- Auth, CORS wildcard, CI, repo weight cleanup.

## How to demo

```bash
docker compose up --build
# backend: http://localhost:8000/docs
# frontend: http://localhost:5173
# admin login: admin@pulmoscan.org / AdminPass123!
# nodes auto-register; click "Start FL Round" in the admin dashboard.
```

Watch three nodes train, upload real state_dicts, and watch the orchestrator
log `[AGGREGATION] Real tensor-wise FedAvg complete.` followed by a checkpoint
that `model_inference.ChestXRayInference` can actually load.
