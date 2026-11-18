# How PulmoScan Models Get Updated

End-to-end description of the federated model-update lifecycle **after the
November 2026 fixes**. The pre-fix REST path shipped scalars only — see
`backend/fl_orchestrator.py` git history for the stub version.

## 1. What is "the model"?

A single `ResNet-18` (torchvision) with the final FC replaced by a 2-class
head (`TB Positive`, `TB Negative`). The reference architecture lives in
three identical copies so the container boundaries stay clean:

- `backend/model_inference.py::_resnet18()` — inference time
- `backend/fl_orchestrator.py::_build_resnet18()` — central eval
- `Model/src/core/model.py::get_resnet18()` — edge training

Any new layer or head change must be applied to **all three**.

## 2. Checkpoint format

The orchestrator writes a wrapped checkpoint at
`Model/models/federated_model.pth`:

```python
{
    "model_state": OrderedDict[str, torch.Tensor],   # real state_dict
    "meta": {
        "version":     "v3",
        "round":       2,
        "accuracy":    87.4,
        "loss":        0.291,
        "aggregation": "FedAvg",
        "timestamp":   "2026-11-18T22:40:11Z",
        "participants": ["node_A", "node_B", "node_C"],
    },
}
```

`backend/model_inference.py::_load_state_dict` transparently accepts both
this wrapper and a bare state_dict, so you can hand-drop a HuggingFace or
centrally-trained checkpoint and it still loads.

## 3. Round lifecycle (REST path, what `docker compose up` runs)

```
┌──────────────────────────────── Round N ────────────────────────────────┐
│                                                                         │
│  Admin UI ──POST /api/fl/round/start──▶ Orchestrator                    │
│                                          stage: broadcasting            │
│                                                                         │
│  for each edge node (A, B, C):                                          │
│      Edge ──GET  /api/fl/model/current-weights──▶ Orchestrator          │
│      Edge ◀── {version, weights_b64 (base64 torch.save)} ── Orch        │
│      Edge: model.load_state_dict(decode(weights_b64))                   │
│      Edge: snapshot w_global = {n: p.clone() for n,p in model.named_…}  │
│                                                                         │
│  Orchestrator sets stage: local_training                                │
│                                                                         │
│  Each edge polls /api/fl/live-status every 1.5s; when stage flips it:   │
│      - attaches Opacus PrivacyEngine via make_private(model, opt, dl)   │
│      - runs batch_limit minibatches with:                               │
│          loss = CE(out, y)                                              │
│                 + (mu/2) * Σ ||w_i - w_global_i||²   # FedProx          │
│      - computes real ε via privacy_engine.get_epsilon(delta)            │
│                                                                         │
│  Edge ──POST /api/fl/round/submit-update {                              │
│         node_id, round, loss, acc, epsilon, num_samples, size_mb,       │
│         weights_hash, weights_b64}  ──▶ Orchestrator                    │
│                                                                         │
│  Orchestrator.submit_update:                                            │
│      - recomputes sha256(weights_b64_bytes); rejects on hash mismatch   │
│      - decodes torch.load(buf, weights_only=False) → state_dict         │
│      - stores update in round_updates[node_id]                          │
│      - if len(round_updates) >= FL_MIN_CLIENTS:                         │
│          spawn thread → _execute_central_aggregation                    │
│                                                                         │
│  _execute_central_aggregation:                                          │
│      1. Collect participants with state_dict present.                   │
│      2. weight_i = n_i / Σn_i                                           │
│      3. For each key k in state_dict:                                   │
│             agg[k] = Σ_i (weight_i * client_i[k].float())               │
│             agg[k] = agg[k].to(original_dtype)                          │
│      4. If Model/data/valid_chest_xray/test exists → evaluate agg       │
│         model on up to 25 batches, else sample-weighted mean of         │
│         client-reported acc/loss. (No more random.uniform.)             │
│      5. torch.save({"model_state": agg, "meta": ...})                   │
│         - Model/checkpoints/federated_model_v{N+1}.pth                  │
│         - Model/models/federated_model.pth (active)                     │
│      6. global_version = v{N+1}; emit ROUND_COMPLETED.                  │
│                                                                         │
│  Next /predict call picks up the new active checkpoint automatically    │
│  because ChestXRayInference lazy-loads from the same path.              │
└─────────────────────────────────────────────────────────────────────────┘
```

## 4. Network payload specifics

| Direction | Endpoint | Body | Approx size (ResNet-18, fp32) |
|---|---|---|---|
| Server → Edge | `GET /api/fl/model/current-weights` | `{"version": "v2", "weights_b64": "..."}` | ~60 MB JSON (45 MB binary × 4/3 b64 overhead) |
| Edge → Server | `POST /api/fl/round/submit-update` | metrics + full `state_dict` | ~60 MB |
| Edge → Server | `POST /api/fl/node/heartbeat` (every 3 s) | ~400 B | — |
| Edge → Server | `POST /api/fl/events/emit` | ~600 B / batch | — |

Weight bytes are integrity-checked by SHA-256(truncated to 16 hex). Server
rejects the update on mismatch. **Confidentiality** is provided today by
Docker's internal bridge network only; wire mTLS for production (CA + certs
already scaffolded in `Model/certificates/` — only the Flower/gRPC path uses
them so far).

## 5. Differential privacy

`run_hospital_node.py` attaches `opacus.PrivacyEngine` **per round** via
`make_private(model, optimizer, data_loader, noise_multiplier, max_grad_norm)`.
Opacus wraps the module as `GradSampleModule` and intercepts `optimizer.step`
to perform per-sample clipping + Gaussian noise. Epsilon is read with
`privacy_engine.get_epsilon(delta=1e-5)` at the end of the round and reported
honestly. If Opacus import fails in an environment, the edge falls back to
plain `clip_grad_norm_` and logs that DP is downgraded (no silent lie).

## 6. FedProx

`loss += (μ/2) · Σ ||w − w_global||²` — the sum is over named_parameters,
with `w_global` snapshotted **before any local optimization step**. The
previous `(μ/2) · ||w||²` variant degenerated to L2 weight decay and did
not pull local updates toward the global model.

## 7. Aggregation strategies

Only **sample-weighted FedAvg** is wired on the REST path today
(`_fedavg` in `backend/fl_orchestrator.py`). `trimmed_mean_aggregation` exists
in the Flower/gRPC path (`Model/src/federated/server.py`). The `aggregation`
knob on `/api/fl/round/start` is accepted but currently delegates to FedAvg
for the REST flow — see Gap 12 in `CHANGES.md`.

## 8. Where failures end up

- **Hash mismatch on upload** → `400 {status: "error", message: "weight hash mismatch..."}`; the round never aggregates until quorum.
- **Edge can't download global weights** → logs a warning and trains from current local weights (still contributes a usable update after FedProx pulls toward last-known global).
- **Aggregated eval fails** (no test dir / torch error) → falls back to sample-weighted mean of client acc/loss. Logged, never fabricated.
- **No Opacus in the container** → clearly logged; ε is set to the heuristic `0.42 · round`.
