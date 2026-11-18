# PulmoScan — Privacy-Preserving Federated Learning for Tuberculosis Screening

PulmoScan is a clinical AI decision-support system for **tuberculosis (TB) detection from chest X-rays**, built on a **multi-hospital federated learning (FL)** platform. Three "hospital" nodes train a shared ResNet-18 classifier *without ever exchanging patient images* — only encrypted, differentially-private model updates leave each node. A central orchestrator aggregates those updates with FedAvg, writes a new global checkpoint, and the clinical web app automatically starts using it on the next inference.

> This README is the single run- and concept-reference. For the detailed FL lifecycle see [`MODEL_UPDATES.md`](MODEL_UPDATES.md); for the architectural tour see [`ARCHITECTURE.md`](ARCHITECTURE.md); for the end-to-end click-through see [`DEMO_GUIDE.md`](DEMO_GUIDE.md).

---

## 0. Project Objective → Implementation Trace

> **Stated objective:** develop an AI-based system for early tuberculosis detection from chest X-ray images that **(a)** provides accurate disease predictions, **(b)** highlights affected regions for better interpretation, and **(c)** enables collaborative learning across multiple healthcare centers.

Every requirement is implemented. Direct code pointers:

| Requirement | How it's realized | Where in code |
|---|---|---|
| **Early TB detection from CXR** | ResNet-18 (torchvision) with a 2-class head (`TB_Positive`, `TB_Negative`), loaded lazily from the current FL checkpoint. Input accepts DICOM / JPEG / PNG / WebP. An upstream **OOD gatekeeper ResNet-18** rejects non-chest images (abdominal CT, derma, blank/corrupted) *before* the TB model runs, so predictions are only produced on valid CXRs. | `backend/model_inference.py` — `ChestXRayInference`, `_resnet18()`, `predict()` · validator at `Model/models/chest_xray_validator.pth` |
| **Accurate disease predictions** | Softmax confidence returned per prediction; confidence buckets (0.60 / 0.70 / 0.85 / 0.95) map to TB severity stages 1–4, which in turn drive a WHO-guideline-seeded `care_plan` object (HRZE / 3HP regimens, isolation, follow-up). Clinician sign-off is persisted and corrections feed the active-learning queue so accuracy improves over rounds. | `backend/model_inference.py::predict` (softmax + stage bucketing) · `backend/stages.json` + `_build_care_plan` · `backend/database.py::save_feedback` + `enqueue_active_learning` |
| **Highlights affected regions** | **Grad-CAM++** over `layer4` of the TB ResNet-18. The extractor computes a gradient-weighted class activation map, overlays it on the input CXR with a JET colormap, and derives **bounding boxes** around high-activation lesion regions. Both `heatmap_base64` (ready-to-render PNG) and `bounding_boxes` are returned in the `/predict` response and rendered by the frontend result page. | `backend/model_inference.py::GradCAMPlusPlusExtractor` (line 69) · overlay at line 152 · bbox derivation at line 158 · `Frontend/src` result page consumer |
| **Collaborative learning across multiple healthcare centers** | Three "hospital" FL clients (`node_A/B/C`) each hold a private CXR shard (`Model/data/hospital_*`), train locally under **Opacus DP-SGD + FedProx**, and upload only their updated ResNet-18 `state_dict` (SHA-256 integrity-checked) to a central orchestrator. The orchestrator aggregates with **sample-weighted FedAvg** once ≥ `FL_MIN_CLIENTS` nodes have submitted, evaluates the new global model on a held-out central test set, and writes a versioned checkpoint that the inference API picks up on the next request. **No patient images, labels, or identifiers ever leave a hospital node.** | `backend/fl_orchestrator.py` — `_fedavg`, `_execute_central_aggregation`, `submit_update` · `Model/run_hospital_node.py` — download global → FedProx+DP training loop → upload weights · REST endpoints `/api/fl/*` in `backend/main.py` |

Additional capabilities that reinforce the objective but weren't explicitly asked for:

- **HIPAA Safe-Harbor de-identification** of DICOMs (strips 18 PHI tags + irreversibly hashes patient IDs) — `backend/dicom_sanitizer.py`.
- **Differential privacy** on every edge node — real `ε` reported from `opacus.PrivacyEngine.get_epsilon(δ=1e-5)` rather than a fabricated constant.
- **FedProx** proximal term (`loss += (μ/2) · Σ‖w − w_global‖²`) stabilizes training when hospital datasets are non-IID.
- **Audit trail** of every scan, feedback event, and admin action in `backend/feedback.db` (SQLite) — queryable from the admin dashboard.

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

### Option A: Docker Compose (recommended)

Brings up backend + frontend + three hospital nodes on one `pulmoscan` bridge network.

```bash
# from the repo root
docker compose up --build
```

Then:
- Frontend: http://localhost:5173
- Backend / Swagger: http://localhost:8000/docs
- Admin login: `admin@pulmoscan.org` / `AdminPass123!` (override with env var `PULMOSCAN_ADMIN_PASSWORD` **before first startup**)

Click **Start FL Round** in the admin dashboard — logs for all three hospital containers will stream local training, then the backend will log `[AGGREGATION] Real tensor-wise FedAvg complete.`

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
| `POST /auth/{login,register,logout}` | Session management |
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
| `/predict` returns 503 | Checkpoint load failure — `backend/model_inference.py` |
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
