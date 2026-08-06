# AI Video Surveillance Pipeline

Person detection, multi-object tracking, and zone-based event recognition
(intrusion, loitering) for security camera footage.

> Status: project initialized — FastAPI skeleton, config/logging/settings
> wired up. Detection and tracking are **not** implemented yet.

## Architecture (current)

```
src/
├── main.py                    # ASGI entrypoint: uvicorn src.main:app --reload
└── surveillance/
    ├── api/
    │   ├── app.py              # FastAPI application factory
    │   ├── health.py           # GET /health (unversioned, infra-level)
    │   └── v1/router.py        # versioned router, reserved for future endpoints
    ├── core/
    │   ├── settings.py         # Pydantic BaseSettings, reads .env
    │   ├── config_loader.py    # loads config/config.yaml
    │   ├── constants.py        # project-wide constants
    │   └── logging_config.py   # Loguru setup (console + logs/app.log)
    ├── pipelines/               # detection / tracking / events / output — empty, next step
    ├── services/                 # use-case orchestration — empty, next step
    ├── models/                    # domain models + weight loaders — empty, next step
    ├── database/                  # repositories/migrations — empty, unless persistence is added
    └── utils/                     # cross-cutting helpers — empty, next step
```

Endpoints today:
- `GET /` — service info
- `GET /health` — health check

## Setup

Requires Python 3.9+ (this environment has 3.9.6; no newer interpreter was
available, so the project targets 3.9+ rather than 3.10+).

```bash
# Option A — helper script (creates .venv, installs deps, copies .env)
bash scripts/setup_env.sh          # runtime deps only
bash scripts/setup_env.sh --dev    # + test/lint tooling

# Option B — manual
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # add -r requirements-dev.txt for tests/lint
cp .env.example .env
```

## Run

```bash
source .venv/bin/activate
uvicorn src.main:app --reload
```

Then:
```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/health
```

## Configuration

Two layers, intentionally separate:
- **`.env`** (copied from `.env.example`) — environment-specific values:
  device, ports, paths, database URL, log level. Never committed.
- **`config/config.yaml`** — non-secret application/pipeline configuration
  (app metadata, pipeline defaults). Loaded via `core/config_loader.py`.
- **`config/zones/*.json`** — zone polygon definitions for the event
  detection stage (schema to be defined when that stage is built).

`core/settings.py` exposes a cached `get_settings()` singleton backed by
Pydantic, so environment variables are validated once at startup.

## Dependencies

`requirements.txt` covers the full production stack (FastAPI, OpenCV,
Ultralytics/YOLO, tracking libs, SQLAlchemy/Alembic, etc.) even though
detection/tracking code hasn't landed yet — installing it now pulls in
torch/ultralytics/opencv, which is a large download. `requirements-dev.txt`
adds pytest, ruff, and mypy on top.

## Known limitations (current stage)

- No detection, tracking, or event logic yet — this step is API/project
  scaffolding only.
- `api/v1/router.py` is an empty placeholder for future versioned endpoints.
- No persistence wired up yet (SQLAlchemy/Alembic are in requirements but unused).

## Next steps

- Implement the detector (Ultralytics YOLO) and tracker (ByteTrack/DeepSORT) stages.
- Wire zone-based event detection (intrusion, loitering) against `config/zones/`.
- Flesh out `docs/architecture/` with a pipeline diagram.
