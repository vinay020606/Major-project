# PulmoScan — Privacy-Preserving Federated Learning for Tuberculosis Screening

PulmoScan is a clinical AI decision-support system for **tuberculosis (TB) detection from chest X-rays**, built on a **multi-hospital federated learning (FL)** platform. Three "hospital" nodes train a shared ResNet-18 classifier *without ever exchanging patient images* — only encrypted, differentially-private model updates leave each node. A central orchestrator aggregates those updates with FedAvg, writes a new global checkpoint, and the clinical web app automatically starts using it on the next inference.

> This README is the single run- and concept-reference. For the detailed FL lifecycle see [`MODEL_UPDATES.md`](MODEL_UPDATES.md); for the architectural tour see [`ARCHITECTURE.md`](ARCHITECTURE.md); for the end-to-end click-through see [`DEMO_GUIDE.md`](DEMO_GUIDE.md).

---

## 0. Project Objectives → Implementation Trace

> **Stated objectives:**
> 1. **AI-Based TB Detection** — CNN system to detect TB from CXRs with **Grad-CAM** explainability.
> 2. **Privacy-Preserving Federated Learning** — multiple hospital nodes collaboratively train the model without sharing raw patient data.
> 3. **Integrated Doctor and Patient Platform** — X-ray analysis, patient history, symptom assessment, monitoring, care information, and feedback.
> 4. **Intelligent Healthcare Support** — assists doctors in TB screening and lets patients monitor symptoms, diagnosis, treatment adherence, and progress.

Every objective is implemented. Direct code pointers:

### Objective 1 — AI-Based TB Detection + Grad-CAM

| Feature | Where in code |
|---|---|
| ResNet-18 TB classifier (2-class head: TB_Positive / TB_Negative) | `backend/model_inference.py::_resnet18`, `ChestXRayInference.predict` |
| Upstream OOD gatekeeper (rejects non-chest / blank / corrupted inputs) | `ChestXRayInference` validator path, threshold `VALIDATOR_THRESHOLD=0.35` |
| Softmax confidence → severity stage (1–4) | `predict()` bucketing against 0.60 / 0.70 / 0.85 / 0.95 |
| **Grad-CAM++** heatmap over `layer4` + lesion **bounding boxes** returned in `/predict` | `model_inference.py::GradCAMPlusPlusExtractor` (line 69), overlay line 152, bbox derivation line 158 |
| Frontend heatmap viewer | `Frontend/src/components/HeatmapViewer.jsx`, consumed in `pages/ResultPage.jsx` |

### Objective 2 — Privacy-Preserving Federated Learning

| Feature | Where in code |
|---|---|
| Central FL orchestrator (round lifecycle, state machine, event bus) | `backend/fl_orchestrator.py` |
| Hospital-node client (download global → train locally → upload weights) | `Model/run_hospital_node.py` |
| **Sample-weighted FedAvg** aggregation (dtype-preserving) | `fl_orchestrator.py::_fedavg`, `_execute_central_aggregation` |
| **Opacus DP-SGD** per-round, real `ε` reported via `get_epsilon(δ=1e-5)` | `run_hospital_node.py` PrivacyEngine attachment |
| **FedProx** proximal term `(μ/2)·Σ‖w − w_global‖²` for non-IID stability | `run_hospital_node.py` loss computation |
| SHA-256 integrity check on every weight upload | `fl_orchestrator.py::submit_update` |
| `torch.load(..., weights_only=True)` everywhere weights are deserialized | FL path, both server and client sides |
| Admin FL controls (start/cancel round, live status, nodes, events) | `/api/fl/*` endpoints in `backend/main.py`, frontend `pages/FLMonitorPage.jsx`, `pages/FederatedMonitorPage.jsx`, `pages/AdminDashboard.jsx` |

### Objective 3 — Integrated Doctor and Patient Platform

| Feature | Where in code |
|---|---|
| **X-ray analysis upload** (DICOM/JPEG/PNG/WebP ≤ 20 MB) | `pages/UploadPage.jsx` + `components/UploadDropzone.jsx` → `POST /predict` |
| **Doctor result review** with Grad-CAM, bboxes, care plan | `pages/ResultPage.jsx` + `components/StageCarePanel.jsx` |
| **Patient dashboard** (latest scan, doctor sign-off status, attending clinician) | `pages/PatientDashboard.jsx` |
| **Patient history** (role-isolated list of past scans) | `pages/HistoryPage.jsx` + `components/HistoryTable.jsx` → `GET /scans` |
| **Doctor feedback** (confirm/override AI, free-text notes → active-learning queue) | `ResultPage.jsx::submitFeedback` → `POST /feedback` → `enqueue_active_learning` |
| **Structured care information** (WHO-aligned regimens, isolation, follow-up, red flags) | `backend/stages.json` + `_build_care_plan` in `backend/main.py` → `components/StageCarePanel.jsx` |
| **Patient symptom self-assessment** (cardinal TB symptoms → risk band Low/Moderate/High) *(new)* | Backend: `database.py::save_symptom_assessment`, `get_symptom_assessments`, `/patient/symptoms` endpoints in `main.py`. Frontend: symptom panel in `pages/PatientDashboard.jsx` |
| **Medication adherence tracker** (per-day regimen check-ins) | `PatientDashboard.jsx` adherence tiles (UI-local; persistence is a known deferred item) |
| **Admin / audit platform** (user management, HIPAA audit log, FL cluster telemetry) | `pages/AdminDashboard.jsx` + `/admin/*`, `/audit/recent` endpoints |

### Objective 4 — Intelligent Healthcare Support

| Feature | Where in code |
|---|---|
| **TB screening for doctors** — inference pipeline + Grad-CAM + severity stage + care plan surfaced in one review page | `pages/ResultPage.jsx` + `/predict` response |
| **Symptom monitoring for patients** — TB cardinal-symptom survey with WHO-aligned screening heuristic and persisted history | `/patient/symptoms` endpoints + `PatientDashboard.jsx` symptom panel |
| **Diagnosis surfacing** — confidence, stage, heatmap, doctor-reviewed label | `ResultPage.jsx` + `PatientDashboard.jsx` Clinical Screening Summary card |
| **Treatment adherence** — medication tiles tied to doctor-confirmed TB-positive diagnoses | `PatientDashboard.jsx` adherence section (conditional on `isPositive && doctorConfirmed`) |
| **Health progress** — chronological scan list, per-record review notes, chronological symptom-assessment history with risk-band trend | `HistoryPage.jsx` + new "Recent assessments" list in symptom panel |
| **Decision-support disclaimer** attached to every care plan (not a prescription) | `backend/main.py::_build_care_plan` disclaimer field |

### Symptom-assessment scoring (reference)

The screening heuristic in `backend/database.py::_score_symptoms` follows the WHO cardinal-symptom approach:

| Signal | Points |
|---|---|
| Cough ≥ 3 weeks | 3 |
| Coughing up blood (hemoptysis) | 3 |
| Each of: persistent fever, night sweats, weight loss, chest pain, fatigue, breathlessness | 1 each |

Risk bands: **0–1 Low · 2–4 Moderate · ≥ 5 High**. The band is persisted with every submission and surfaced in the patient dashboard history list.

### Additional capabilities reinforcing the objectives

- **HIPAA Safe-Harbor de-identification** of DICOMs (strips 18 PHI tags + irreversibly hashes patient IDs) — `backend/dicom_sanitizer.py`.
- **Differential privacy** on every edge node — real `ε` reported from `opacus.PrivacyEngine.get_epsilon(δ=1e-5)` rather than a fabricated constant.
- **Audit trail** of every scan, feedback event, symptom submission, and admin action in `backend/feedback.db` (SQLite) — queryable from the admin dashboard.

---

## 1. Core Usage — What the Project Does

Two workflows run side-by-side:

### 1a. Clinical inference (what a doctor/patient sees)

1. Clinician uploads a CXR (DICOM / JPEG / PNG / WebP, ≤ 20 MB).
2. **Stage 1 — HIPAA Safe-Harbor de-identification** strips 18 PHI tags from DICOM metadata and hashes the patient ID into an irreversible SHA-256 pseudonym (`backend/dicom_sanitizer.py`).
3. **Stage 2 — OOD gatekeeper** ResNet-18 rejects non-chest images (abdominal CTs, dermatology photos, blank/corrupted files) before any diagnostic processing.
4. **Stage 3 — TB classifier** ResNet-18 predicts `TB Positive` / `TB Negative` with softmax confidence.
5. **Stage 4 — Grad-CAM++** overlays a lesion heatmap on `layer4` and returns base64-encoded bounding boxes.
6. **Stage 5 — Care pathway + clinician sign-off.** Confidence → TB severity stage (1 latent → 4 severe) → structured `care_plan` object (WHO-guideline-seeded HRZE / 3HP regimens, isolation rules, follow-up schedule). Clinician accepts/corrects; corrections land in the **active-learning queue** for the next FL round.

### 1b. Federated training (how the model keeps improving)

1. Admin clicks **Start FL Round** in the dashboard → `POST /api/fl/round/start`.
2. Each hospital node (`node_A`, `node_B`, `node_C`) downloads the current global `state_dict` from `GET /api/fl/model/current-weights` (base64-encoded `torch.save` blob, SHA-256 integrity-checked).
3. Each node snapshots `w_global`, attaches an **Opacus `PrivacyEngine`** (`make_private(model, optimizer, dataloader)`) for per-sample gradient clipping + Gaussian noise (DP-SGD), and trains locally with the **FedProx** loss:
   ```
   loss = CE(ŷ, y) + (μ/2) · Σ ‖w − w_global‖²
   ```
   The proximal term pulls each node's update back toward the shared starting point, which keeps training stable when hospital datasets are non-IID.
4. Each node reports `ε` from `privacy_engine.get_epsilon(δ=1e-5)` honestly (no fabricated privacy budget) and posts `{loss, accuracy, epsilon, num_samples, weights_hash, weights_b64}` to `POST /api/fl/round/submit-update`.
5. Once `len(round_updates) ≥ FL_MIN_CLIENTS` (default 3), the orchestrator runs **sample-weighted FedAvg**:
   ```
   weight_i = n_i / Σn_i
   agg[k]  = Σ_i (weight_i · client_i[k].float())
   ```
   evaluates it on a held-out CXR test set (falls back to sample-weighted mean of client metrics if unavailable — never fabricates numbers), and writes `Model/models/federated_model.pth` plus a versioned `Model/checkpoints/federated_model_v{N+1}.pth`.
6. The next `/predict` call lazy-loads the new checkpoint — no restart needed.

**Nothing leaves a hospital node except the weight tensors and scalar metrics.** No images, labels, or patient IDs traverse the network.

---

## 2. Datasets

| Dataset | Location | What it is | Used for |
|---|---|---|---|
| `Model/data/hospital_A` | `TB_Positive/`, `TB_Negative/` | Node A's private CXR shard | Local FL training on node A |
| `Model/data/hospital_B` | `TB_Positive/`, `TB_Negative/` | Node B's private CXR shard | Local FL training on node B |
| `Model/data/hospital_C` *(if present)* | `TB_Positive/`, `TB_Negative/` | Node C's private CXR shard | Local FL training on node C |
| `Model/data/valid_chest_xray/` | `train/`, `val/`, `test/` | Held-out balanced CXR corpus | OOD validator training + central FL eval |
| `Model/data/invalid/` | `skin/`, `blood_tissue/`, `ct_mri_ultrasound/`, `photographs/`, `unseen_ood/` | MedMNIST derivatives (DermaMNIST, PathMNIST, OrganAMNIST, OCTMNIST held out) + photos | Training the chest-X-ray **validator** (OOD gatekeeper) |
| `Model/data/central_evaluation/` | balanced sample | Central aggregator's eval batch | Scoring each aggregated global model |

**Source of truth.** CXR TB images are seeded from the standard open TB-screening corpora (Shenzhen + Montgomery style). OOD imagery comes from MedMNIST v3 (regenerable via `Model/src/.../prepare_validator_data.py`). Nothing in `Model/data/hospital_*/` is uploaded; those directories are mounted into each node's container and never shared.

---

## 3. Model Architecture

- **Base.** `torchvision.models.resnet18` with the final `fc` replaced by a 2-class head (`TB_Positive`, `TB_Negative`).
- **Three identical reference copies** so container boundaries stay clean (any new layer must be applied to all three):
  - `backend/model_inference.py::_resnet18()` — inference time
  - `backend/fl_orchestrator.py::_build_resnet18()` — central eval
  - `Model/src/core/model.py::get_resnet18()` — edge training
- **OOD validator.** A second ResNet-18 (`Model/models/chest_xray_validator.pth`) acts as a binary CXR / non-CXR gate in front of the TB classifier; rejects below `0.35` probability.
- **Checkpoint format** (`Model/models/federated_model.pth`):
  ```python
  {
    "model_state": OrderedDict[str, torch.Tensor],
    "meta": {"version": "v3", "round": 2, "accuracy": 87.4,
             "loss": 0.291, "aggregation": "FedAvg",
             "timestamp": "2026-11-18T22:40:11Z",
             "participants": ["node_A", "node_B", "node_C"]},
  }
  ```
  `model_inference.py::_load_state_dict` transparently accepts both this wrapper and a bare `state_dict`.

---

## 4. How Weights Are Updated (one-paragraph summary)

Every round the orchestrator broadcasts the current global `state_dict`, each node runs `batch_limit` mini-batches locally under Opacus DP-SGD + FedProx proximal regularization, posts its full `state_dict` back as a base64-encoded `torch.save` blob with a SHA-256 integrity tag, and the orchestrator aggregates with sample-weighted FedAvg once ≥ `FL_MIN_CLIENTS` nodes have submitted. The aggregated tensor dictionary is cast back to the original per-tensor dtype (BatchNorm counters stay int64), evaluated on a held-out CXR test set, and written to disk. See [`MODEL_UPDATES.md`](MODEL_UPDATES.md) for the full ASCII flow diagram and network payload sizes (~60 MB per transfer for ResNet-18 fp32).

---

## 5. Quickstart — Run the Whole Stack

> **First-time note.** No model checkpoint is shipped in the repo. `/predict` deliberately returns **HTTP 503 `MODEL_NOT_READY`** until one exists — the previous behavior of silently serving a randomly-initialized ResNet-18 is gone. You seed a real checkpoint either by **running one FL round** (step below) or by dropping a pretrained ResNet-18 `state_dict` at `Model/models/federated_model.pth`. Check readiness anytime with `GET /health`.

### Option A: Docker Compose (recommended demo path)

Brings up backend + frontend + three hospital nodes on one `pulmoscan` bridge network.

```bash
# 1. from the repo root
docker compose up --build

# 2. open the dashboard
#    Frontend: http://localhost:5173
#    Backend : http://localhost:8000/docs
#    Health  : http://localhost:8000/health   ← should say "model_not_ready" at first

# 3. log in as admin
#    admin@pulmoscan.org / AdminPass123!
#    (override with env var PULMOSCAN_ADMIN_PASSWORD BEFORE first startup)

# 4. seed the model
#    In the admin dashboard, click "Start FL Round".
#    Hospital node logs will stream local training, then the backend will log:
#    [AGGREGATION] Real tensor-wise FedAvg complete.
#    /health now reports "ok" and /predict works.

# 5. use the clinical UI
#    - Doctor view : upload a CXR → Grad-CAM heatmap + care plan
#    - Patient view: submit symptom self-assessment, view history, track adherence
```

Shut down with `docker compose down`. Add `-v` to also wipe the SQLite DB + stored inferences + checkpoints.

### Option B: Run each service manually

Requires Python 3.10+ and Node 18+.

**1. Backend (FastAPI + FL orchestrator + inference + SQLite)**

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```
Swagger UI: http://localhost:8000/docs

**2. Frontend (Vite + React)**

```bash
cd Frontend
npm install
npm run dev
```
Dashboard: http://localhost:5173

**3. Hospital FL clients** — in three separate terminals:

```bash
cd Model
python -m pip install -r requirements.txt

# terminal 1
NODE_ID=node_A API_SERVER=http://localhost:8000 python run_hospital_node.py

# terminal 2
NODE_ID=node_B API_SERVER=http://localhost:8000 python run_hospital_node.py

# terminal 3
NODE_ID=node_C API_SERVER=http://localhost:8000 python run_hospital_node.py
```
(On Windows PowerShell: `$env:NODE_ID="node_A"; $env:API_SERVER="http://localhost:8000"; python run_hospital_node.py`.)

Each client registers with the orchestrator, heartbeats every 3 s, and begins polling `/api/fl/live-status` for `stage=local_training`.

**4. End-to-end smoke test**

```bash
python test_end_to_end.py
```
Walks the full five-stage pipeline (DICOM de-ID → OOD gate → inference → feedback → retrain-check).

### Option C: Three physical laptops over LAN

See [`THREE_LAPTOP_FL_GUIDE.md`](THREE_LAPTOP_FL_GUIDE.md) for IP wiring, firewall rules, and TLS scaffolding.

---

## 6. Default Credentials & Environment

| Role | Email | Password | Scope |
|---|---|---|---|
| System Admin | `admin@pulmoscan.org` | `AdminPass123!` (or `$PULMOSCAN_ADMIN_PASSWORD`) | FL controls, HIPAA audit trail |
| Clinicians / Patients | Self-register via `/auth/register` | Set by user | Role-isolated workspaces |

Note: `/auth/register` **rejects** `role=admin` — admin accounts can only be created via the seed path.

### Environment variables worth knowing

| Var | Default | Effect |
|---|---|---|
| `FL_MIN_CLIENTS` | `3` | Quorum before FedAvg fires |
| `ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | CORS allow-list (comma-separated) |
| `PULMOSCAN_ADMIN_PASSWORD` | `AdminPass123!` | Seed password for the default admin account |
| `PULMOSCAN_PATIENT_HASH_SALT` | fallback constant | Salt used by `derive_patient_hash` |
| `NODE_ID` | — (required on edge nodes) | `node_A` / `node_B` / `node_C` |
| `API_SERVER` | — (required on edge nodes) | e.g. `http://backend:8000` in Docker, or the LAN IP of the laptop running the backend |

---

## 7. REST Surface (abbreviated — Swagger has full schema)

| Endpoint | Purpose |
|---|---|
| `POST /predict` | CXR inference → prediction + Grad-CAM + care_plan |
| `POST /feedback` | Clinician sign-off / correction → active-learning queue |
| `GET  /care-plan/{stage_id}` | Standalone care pathway lookup (1–4) |
| `GET  /scans` | List predictions (role-filtered) |
| `GET  /health` | Reports whether both the TB classifier and validator checkpoints are on disk; UI uses this to flag "model not ready" before the first FL round |
| `POST /auth/{login,register,logout}` | Session management |
| `POST /patient/symptoms` | Submit a TB-symptom self-assessment → returns computed risk band |
| `GET  /patient/symptoms?email=...` | Patient's symptom-assessment history (newest first) |
| `POST /retrain` | Trigger subprocess federated run |
| `POST /api/fl/round/start` | Begin an FL round |
| `POST /api/fl/round/cancel` | Abort active round |
| `GET  /api/fl/live-status` | Current stage, round, node metrics |
| `GET  /api/fl/model/current-weights` | base64 global `state_dict` |
| `POST /api/fl/round/submit-update` | Edge → central weight upload |
| `POST /api/fl/node/{register,heartbeat}` | Node lifecycle |
| `GET  /api/fl/events` | Real-time event ring buffer |
| `GET  /fl-metrics` | Historical FL training metrics |

---

## 8. Where Things Live on Disk

| Path | Content |
|---|---|
| `backend/feedback.db` | SQLite — users, predictions, feedback, audit, active-learning |
| `backend/edge_storage/inferences/{id}.png` | Cached inference images for history |
| `backend/edge_storage/dicom/*.dcm` | PHI-scrubbed DICOMs |
| `backend/edge_storage/active_learning/*.png` | Doctor-corrected samples queued for the next retrain |
| `Model/models/federated_model.pth` | Active TB classifier (wrapped checkpoint) |
| `Model/models/chest_xray_validator.pth` | OOD gatekeeper |
| `Model/checkpoints/federated_model_v{N}.pth` | Historical FL rounds |
| `Model/data/hospital_{A,B,C}/` | Per-node private training set |

---

## 9. Troubleshooting

| Symptom | Likely cause / file |
|---|---|
| `/predict` returns 503 `MODEL_NOT_READY` | No TB-classifier checkpoint at `Model/models/federated_model.pth`. Run one FL round (admin → Start FL Round) or drop a pretrained state_dict. Confirm with `GET /health`. |
| `/predict` returns 503 (other) | Checkpoint load failure — `backend/model_inference.py` |
| Round hangs at `local_training` | Node container not up, or `API_SERVER` not pointing at a reachable backend |
| Aggregation never fires | Fewer nodes submitted than `FL_MIN_CLIENTS` |
| `weight hash mismatch` on upload | Network truncation — `fl_orchestrator.submit_update` |
| UI shows mock predictions | Backend unreachable — `services/api.js` `ENABLE_MOCK_FALLBACK` kicked in |
| `care_plan` missing | Check `_build_care_plan` and `backend/stages.json` |
| CORS blocked from a new origin | Add it to `ALLOWED_ORIGINS` env (comma-separated) |

---

## 10. Security Posture — What's Hardened, What Isn't

**Hardened in this branch:**
- `torch.load(..., weights_only=True)` everywhere weights are deserialized from the network (closes a pickle RCE from any FL client).
- CORS no longer uses `allow_origins=["*"]` alongside `allow_credentials=True`; origins come from `ALLOWED_ORIGINS`.
- `/auth/register` rejects `role=admin` so clients cannot self-elevate.
- Audit logs no longer write the literal admin password; seed password sourced from env var.
- DICOM patient-hash salt reads from `PULMOSCAN_PATIENT_HASH_SALT` env with the previous constant as fallback.
- The `SCAN_ANALYZED` audit event derives user role from the DB record instead of substring-matching `"admin"` in the email.

**Correctness fixes in this branch:**
- No silent random fallback. `/predict` returns 503 `MODEL_NOT_READY` (not a random prediction) when checkpoints are absent. `GET /health` reports readiness up front.
- Hospital nodes rebuild a fresh unwrapped ResNet-18 each FL round and load the fetched global `state_dict` into it before Opacus wrapping. Previously round-2 onward silently failed to load (Opacus had prefixed every key with `_module.`) and nodes diverged locally.
- Doctor-corrected `active_learning_queue` entries are now folded into training via `ActiveLearningDataset` and marked complete on successful upload. The REST FL path previously only trained on the static local folder.
- `node_C` now resolves to `Model/data/hospital_C/`. Previously it silently aliased to `hospital_B`, so nodes B and C trained on identical data.

**Known deferred items (not yet fixed — PRs welcome):**
- Bearer tokens are issued but never validated server-side. Any caller can hit `/retrain`, `/admin/*`, `/api/fl/*`, and can read any user's scans by passing `role=doctor` in the query string.
- Password hashing is single-round SHA-256 with a fixed in-source salt. Should be bcrypt or argon2id.
- No LAN TLS on REST — Docker's internal bridge is the only confidentiality boundary today. mTLS scaffolding exists for the Flower/gRPC path (`Model/certificates/`).
- Round state is not persisted across backend restarts.
- No per-round quorum timeout for slow/late nodes.
- Active-learning queue is consumed only by the Flower/gRPC path, not by `run_hospital_node.py`.
- Coordinate-wise-median aggregation is advertised but only FedAvg is implemented on the REST path.

See [`CHANGES.md`](CHANGES.md) for the full history.

---

## 11. Project Structure

```
repo/
├── backend/                 FastAPI + FL orchestrator + inference + SQLite
│   ├── main.py              REST API + CORS + request routing
│   ├── fl_orchestrator.py   Round lifecycle, FedAvg, central eval
│   ├── model_inference.py   ChestXRayInference (OOD + TB + Grad-CAM++)
│   ├── database.py          SQLite schema + seed admin + audit log
│   ├── dicom_sanitizer.py   PHI strip + patient-hash derivation
│   └── requirements.txt
├── Frontend/                React + Vite clinical dashboard
├── Model/                   Edge training + validator
│   ├── run_hospital_node.py Per-hospital FL client (REST path)
│   ├── run_federated.py     Local 3-node simulator (Flower/gRPC)
│   ├── run_inference.py     Offline inference helper
│   ├── src/                 model.py, training loop, Flower server/client
│   ├── data/                hospital_A, hospital_B, valid_chest_xray, invalid, central_evaluation
│   ├── models/              active checkpoints
│   ├── checkpoints/         historical FL rounds
│   └── certificates/        mTLS scaffolding (gRPC path only)
├── docker-compose.yml       One-command stack (5 containers)
├── main.py                  ASGI root shim (`uvicorn main:app` at repo root)
├── test_end_to_end.py       5-stage smoke test
├── ARCHITECTURE.md          Service tour + request paths
├── MODEL_UPDATES.md         FL round lifecycle, deep dive
├── DEMO_GUIDE.md            Click-through runbook
├── THREE_LAPTOP_FL_GUIDE.md Physical LAN setup
├── HEALTHCARE_DESIGN_SYSTEM.md
└── CHANGES.md               Historical gap-fix log
```
