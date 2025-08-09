## RS-AI

Reptile Sponge Artificial Intelligence (RS-AI) — a modular, lightweight research stack that explores memory-centric neural components, curiosity, introspection, and simple federated-chain simulation.

### Highlights
- Modular "spine" with pluggable components (Hopfield, HRR, SSM, world model)
- File-backed sponge memory with optional compression
- Curiosity, introspection, and policy/why tracing
- Dashboard/metrics (optional) and a simple trainer orchestrator
- Low-memory profile for constrained devices (Chromebook-friendly)
- Verbose logging toggle to understand what’s happening

---

## Quick start
Minimal run (no dashboard, single pass):
```bash
python3 test_run.py
```
Enable verbose logs:
```bash
RS_VERBOSE=1 python3 test_run.py
```
Run on low-memory devices (Chromebook mode):
```bash
RS_LOW_MEM=1 python3 test_run.py
# Combine with verbose
RS_LOW_MEM=1 RS_VERBOSE=1 python3 test_run.py
```

> Output, logs and artifacts are written to `data/` (created automatically).

---

## Installation
This project aims to run with minimal dependencies.

- Minimal deps (quickest):
```bash
python3 -m pip install --user numpy toml
```
- Full deps (optional, for dashboard, etc.):
```bash
python3 -m pip install --user -r requirements.txt
```

Notes (Debian/Ubuntu/Chromebook):
- If pip warns about “externally-managed environment” (PEP 668), either use `--user` as above or create a venv (`python3 -m venv .venv && . .venv/bin/activate`). Some systems require installing `python3-venv` to use `venv`.

---

## Running options
- Single-shot demo driver:
```bash
python3 test_run.py             # one forward pass, store, introspect
```
- Multithreaded trainer (short session):
```bash
python3 train.py                # honors configs/orchestrator.runtime_seconds
```
- Full stack (trainer + federated chain + optional metrics via Docker):
```bash
bash start_all.sh               # starts trainer and chain in background
# logs: data/logs/train_start.log, data/logs/chain_start.log
```

All runs honor these toggles:
- `RS_LOW_MEM=1` — apply the low-memory profile (smaller tensors/batches, disables heavy bits)
- `RS_VERBOSE=1` — more detailed logs (DEBUG)

Examples:
```bash
RS_LOW_MEM=1 RS_VERBOSE=1 python3 train.py
RS_LOW_MEM=1 bash start_all.sh
```

---

## Low-memory mode (Chromebook)
You can toggle the profile via env or config:
- Env: `RS_LOW_MEM=1`
- Config: set `profiles.low_memory.enabled = true` in `configs/rs-config.toml`

What it changes (defaults):
- `filepaths.sponge_size = [7,7,7]` (343 dims vs 19,683)
- `training.batch_size = 2`, `training.max_steps = 100`, `replay_capacity = 200`
- `training.ssm_hidden = 64`, `training.clip_norm = 0.5`
- Disables world model and dashboard by default; compression off
- Shrinks internal queues to fit smaller replay capacity

Override any of these under `[profiles.low_memory]`.

---

## Verbose logging
Set `RS_VERBOSE=1` to enable detailed step-by-step logs across modules. Examples:
```bash
RS_VERBOSE=1 python3 test_run.py
RS_VERBOSE=1 python3 train.py
```
Logs go to stdout and to the reflection logbook at `data/logs/reflections.md`.

---

## Configuration
Main config lives in `configs/rs-config.toml`.
Key sections:
- `[modules]` — maps module names to Python factories
- `[training]` — batch sizes, steps, learning rates, clipping
- `[memory]` + `[compression]` — backend and optional tensor-network compression
- `[dashboard]` — host/port and `enabled`
- `[filepaths]` — logbook, sponge path, checkpoint path, sponge size
- `[profiles.low_memory]` — overrides applied when enabled or when `RS_LOW_MEM=1`

Example (excerpt):
```toml
[profiles.low_memory]
enabled = false
sponge_size = [7, 7, 7]
batch_size = 2
max_steps = 100
replay_capacity = 200
ssm_hidden = 64
clip_norm = 0.5
memory_backend = "hffs"
compression_enabled = false
compression_rank = 4
dashboard_enabled = false
enable_world_model = false
world_lr = 0.0005
explore_prob = 0.02
runtime_seconds = 6
```

---

## Observability
- Reflection logbook: `data/logs/reflections.md`
- If the dashboard is enabled (`[dashboard].enabled = true`):
  - HTTP server on `http://127.0.0.1:8080`
- Optional Prometheus/Grafana (via Docker):
  - `docker-compose up -d`
  - Grafana: `http://localhost:3000` (admin/admin)
  - Prometheus: `http://localhost:9090`

---

## Project structure
- `test_run.py` — single-shot demo driver
- `train.py` — multithreaded trainer orchestrator
- `spine/` — core modules (neural spine, curiosity, introspection, world model)
- `memory/` — memory backends and utilities
- `tools/` — trainer, dashboard, metrics, policies, logging setup, profiles
- `chain/` — simple federated/chain simulation (optional)
- `configs/` — TOML configuration
- `data/` — logs, sponge, checkpoints (created at runtime)

---

## Troubleshooting
- ModuleNotFoundError: numpy
  - Install minimal deps: `python3 -m pip install --user numpy toml`
- Pip PEP 668 error (externally-managed environment)
  - Use `--user` installs or create a venv; on Debian/Ubuntu, `sudo apt install python3-venv`
- High memory use / slow runs / NaNs or infs in loss
  - Run with `RS_LOW_MEM=1`
  - Reduce `training.ssm_lr` and/or increase `training.clip_norm`
  - Reduce `filepaths.sponge_size` and `training.batch_size`
- Dashboard port busy
  - Set `[dashboard].enabled = false` or change `[dashboard].port`

---

## License and Security
- See `NOTICE.md` and `SECURITY.md` for licensing and security policies.

---

## Acknowledgements
Credits to the RS-AI contributors and the broader research community inspiring memory-centric learning, curiosity, and introspection approaches.  
