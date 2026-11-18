# PulmoScan Architecture (Post-Fix, November 2026)

A one-page tour of the whole stack: what each service does, how data moves,
what the recent fixes changed, and where to look when something breaks.

For the detailed FL-update flow, see [`MODEL_UPDATES.md`](MODEL_UPDATES.md).
For a before/after diff of every gap fix, see [`CHANGES.md`](CHANGES.md).

## 1. What this project is

A clinical decision-support app for **tuberculosis screening from chest
X-rays**, wrapped in a **federated-learning orchestrator** so multiple hospital
containers can jointly improve the classifier without sharing images.

Two capabilities sit side-by-side:

1. **Inference workflow** — doctor uploads a CXR → OOD gatekeeper → TB
   classifier → Grad-CAM++ heatmap → structured care pathway → clinician
   sign-off → active-learning queue.
2. **Federated training workflow** — admin triggers a round → edge nodes
   download current global weights → train locally → upload `state_dict` →
   central FedAvg → new global checkpoint → next inference picks it up.

## 2. Services (one `docker compose up` brings them all up)

| Service | Image | Role | Port |
|---|---|---|---|
| `backend` | `backend/Dockerfile` | FastAPI + FL orchestrator + SQLite + inference | 8000 |
| `frontend` | `Frontend/Dockerfile` | React/Vite clinical dashboard | 5173 |
| `hospital-a` | `Model/Dockerfile` (`NODE_ID=node_A`) | Edge FL client with its own dataset | — |
| `hospital-b` | `Model/Dockerfile` (`NODE_ID=node_B`) | Edge FL client with its own dataset | — |
| `hospital-c` | `Model/Dockerfile` (`NODE_ID=node_C`) | Edge FL client with its own dataset | — |

All five share one Docker bridge network `pulmoscan`. Backend has a
`curl /` healthcheck; nodes wait for it before registering.

## 3. Request paths

### Inference path (`POST /predict`)

```
Upload page
    │ multipart file (DICOM / JPEG / PNG / WebP ≤ 20 MB)
    ▼
backend.main:predict
    │ _read_image → pydicom PHI strip OR PIL open
    ▼
model_inference.ChestXRayInference
    │ ① validator ResNet-18 — reject non-CXR below 0.35
    │ ② tb ResNet-18 — softmax → {label, confidence}
    │ ③ Grad-CAM++ on layer4 → heatmap_base64 + bboxes
    ▼
confidence → stage bucket (0.60/0.70/0.85/0.95 → 1..4)
    ▼
build_care_plan(stage)  ← new: structured care pathway (see §5)
    ▼
save_prediction + log_audit_event (SQLite)
    ▼
Response: {prediction, confidence, stage, heatmap_base64,
           bounding_boxes, care_plan, ...}
```

### Feedback path (`POST /feedback`)

```
ResultPage → submitFeedback → save_feedback + copy image into
edge_storage/active_learning/ → enqueue_active_learning (pending=1).
```

When `pending_count >= RETRAIN_THRESHOLD` (default 5), `/retrain-check`
flags the admin dashboard; `/retrain` kicks off a federated round.

### FL training path (`POST /api/fl/round/start`)

See [`MODEL_UPDATES.md`](MODEL_UPDATES.md) for the detailed diagram. Short
version — every edge node does:

```
GET  /api/fl/model/current-weights    ← download global state_dict (base64)
load_state_dict(fetched)
snapshot w_global
attach Opacus PrivacyEngine via make_private
train batch_limit mini-batches with loss = CE + (μ/2)·Σ||w − w_global||²
POST /api/fl/round/submit-update {metrics + weights_b64}
```

Backend aggregates when `received_count >= FL_MIN_CLIENTS` (default 3) via
real `_fedavg()` on tensor state_dicts; writes
`Model/models/federated_model.pth` as `{"model_state": sd, "meta": {...}}`.
Next `/predict` load picks it up automatically.

## 4. Data stored on disk

| Path | Content |
|---|---|
| `backend/feedback.db` | SQLite: users, predictions, feedback, audit, active-learning |
| `backend/edge_storage/inferences/{id}.png` | cached inference images for history |
| `backend/edge_storage/dicom/*.dcm` | PHI-scrubbed DICOMs |
| `backend/edge_storage/active_learning/*.png` | doctor-corrected samples for next retrain |
| `Model/models/federated_model.pth` | active TB classifier (wrapped checkpoint) |
| `Model/models/chest_xray_validator.pth` | OOD gatekeeper |
| `Model/checkpoints/federated_model_v{N}.pth` | historical FL rounds |
| `Model/data/hospital_{A,B,C}/` | per-node training set (never leaves its container) |

## 5. Care-plan schema (new)

Every TB-positive prediction returns a `care_plan` object:

```json
{
  "stage": 2,
  "name": "Stage 2 — Mild Active Pulmonary TB",
  "severity": "Moderate",
  "infectious": true,
  "summary": "...",
  "steps": {
    "immediate_care":   [...],
    "medication":       {"regimen", "dosing_reference", "alternatives",
                         "adjuncts", "watch_for", "resistance_check"},
    "isolation":        "...",
    "nutrition":        [...],
    "follow_up":        {"visits", "labs", "sputum", "imaging",
                         "contact_tracing", "rehab", "endpoint"}
  },
  "red_flags":          [...],
  "patient_education":  [...],
  "disclaimer":         "Decision-support only. Not a prescription..."
}
```

Content is seeded from WHO TB guidelines (3HP for LTBI, HRZE 2+4 for active,
extended continuation for CNS/disseminated). It is **decision support, not
prescription**; the disclaimer is attached by the backend and rendered by
the frontend.

`StageCarePanel.jsx` renders six cards: medication, isolation+immediate
care, nutrition, follow-up schedule, red flags, patient education.

## 6. Public REST surface

| Endpoint | Purpose |
|---|---|
| `POST /predict` | CXR inference → prediction + Grad-CAM + care_plan |
| `POST /feedback` | Clinician sign-off / correction |
| `GET  /care-plan/{stage_id}` | Standalone care pathway lookup (1-4) |
| `GET  /stage/{stage_id}` | Raw stage record |
| `GET  /scans` | List predictions by role |
| `POST /auth/{login,register,logout}` | Session management (random UUID tokens — see auth caveat) |
| `POST /retrain` | Trigger subprocess federated run |
| `GET  /retrain-{check,status}` | Retrain state |
| **FL cluster** | |
| `POST /api/fl/round/start` | Begin a round (sets stage=broadcasting → local_training) |
| `POST /api/fl/round/cancel` | Abort active round |
| `GET  /api/fl/live-status` | Current stage, round, nodes, metrics |
| `GET  /api/fl/model/current-weights` | **New** — base64 global state_dict |
| `POST /api/fl/round/submit-update` | Edge → central: metrics + weights_b64 |
| `POST /api/fl/node/{register,heartbeat}` | Node lifecycle |
| `GET  /api/fl/events` | Real-time event ring buffer |
| `GET  /fl-metrics` | Historical FL training metrics (benchmark fallback if none) |

## 7. The five "stages" in the UI vs the four TB-stage integers

The UI's 5-stage pipeline (`DICOM deID → OOD gate → FL training →
classification+Grad-CAM → clinician + active learning`) is the **inference
pipeline stages**. The `stage: 1..4` field on a prediction is the **TB
severity grade** (latent/mild/moderate/severe) used to pick the care plan.
They are unrelated axes — don't conflate them.

## 8. Known deferred items (not in this commit)

Tracked in `CHANGES.md` "Not in this batch":
- Auth enforcement (bearer tokens are issued but never validated server-side)
- CORS wildcard with credentials
- LAN TLS for REST traffic
- Round-state persistence across backend restarts
- Per-round quorum timeout for slow/late nodes
- Active-learning dataset wired into `run_hospital_node.py` (only the Flower/gRPC path consumes it today)
- Coordinate-wise-median aggregation (advertised in docstring, only FedAvg implemented)
- Dirichlet non-IID data partitioner
- CI pipeline and `.pth`/large-binary cleanup from git history

## 9. How to run

```bash
docker compose up --build
# backend:   http://localhost:8000/docs
# frontend:  http://localhost:5173
# default admin: admin@pulmoscan.org / AdminPass123!
```

`docker compose logs -f backend` to watch the orchestrator; each hospital
container's logs show its local training. Click "Start FL Round" in the
admin dashboard — three nodes will train, upload real state_dicts, and the
orchestrator will log `[AGGREGATION] Real tensor-wise FedAvg complete.`

## 10. Where to look when something breaks

| Symptom | Likely file |
|---|---|
| `/predict` returns 503 | `backend/model_inference.py` — checkpoint load |
| Round hangs at `local_training` | `Model/run_hospital_node.py` polling loop — check node container is up and `api_server` env points at `http://backend:8000` |
| Aggregation never fires | `FL_MIN_CLIENTS` env on backend vs how many nodes actually submitted |
| Hash mismatch error on upload | `fl_orchestrator.submit_update` — check network wasn't truncated |
| UI shows mock predictions | backend unreachable — `services/api.js` ENABLE_MOCK_FALLBACK kicked in |
| Care plan doesn't render | `care_plan` missing from `/predict` payload; check `_build_care_plan` and stages.json |
