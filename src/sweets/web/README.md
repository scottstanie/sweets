# Sweets Web UI

A web-based interface for the sweets InSAR processing workflow.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Browser                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Svelte Frontend (Vite)                                  │    │
│  │  - Map.svelte      : MapLibre GL JS, bbox drawing        │    │
│  │  - ConfigForm.svelte: Job configuration form             │    │
│  │  - JobList.svelte  : Job list with status polling        │    │
│  │  - ResultsViewer.svelte: COG tile overlay (Phase 3)      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ /api/jobs/*  │  │ /api/ws/{id} │  │ /api/tiles/{z}/{x}/{y}│   │
│  │ CRUD + start │  │ Live logs    │  │ COG tile serving      │   │
│  │ /cancel      │  │ Progress     │  │ (Phase 3)             │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  SQLite Database (~/.sweets/sweets.db)                   │    │
│  │  - Jobs table: id, name, config, status, timestamps      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Job Executor (Background Tasks)                         │    │
│  │  - Spawns `sweets run` subprocess                        │    │
│  │  - Streams logs via WebSocket                            │    │
│  │  - Updates job status in DB                              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
src/sweets/web/
├── __init__.py              # Package init, create_app()
├── app.py                   # FastAPI application factory
├── README.md                # This file
│
├── api/                     # API route handlers
│   ├── __init__.py
│   ├── jobs.py              # Job CRUD + start/cancel endpoints
│   ├── websocket.py         # WebSocket for live log streaming
│   └── tiles.py             # COG tile serving (Phase 3)
│
├── models/                  # Database models (SQLModel)
│   ├── __init__.py
│   ├── database.py          # SQLite engine, session management
│   └── job.py               # Job model and schemas
│
├── services/                # Business logic
│   ├── __init__.py
│   ├── executor.py          # Job execution (local/remote)
│   └── cog.py               # COG conversion utilities (Phase 3)
│
└── frontend/                # Svelte + Vite frontend
    ├── package.json
    ├── vite.config.js
    ├── svelte.config.js
    ├── index.html
    └── src/
        ├── main.js          # Svelte mount point
        ├── App.svelte       # Root component
        └── lib/
            ├── Map.svelte         # MapLibre map with AOI drawing
            ├── ConfigForm.svelte  # Job configuration form
            ├── JobList.svelte     # Job list with actions
            └── ResultsViewer.svelte  # Results map overlay (Phase 3)
```

## API Endpoints

### Jobs API (`/api/jobs/`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/jobs/` | List all jobs (supports `?status=` filter) |
| POST | `/api/jobs/` | Create a new job |
| GET | `/api/jobs/{id}` | Get job by ID |
| PATCH | `/api/jobs/{id}` | Update job |
| DELETE | `/api/jobs/{id}` | Delete job (not if running) |
| POST | `/api/jobs/{id}/start` | Start a pending job |
| POST | `/api/jobs/{id}/cancel` | Cancel a running job |

### WebSocket (`/api/ws/`)

| Endpoint | Description |
|----------|-------------|
| `/api/ws/jobs/{id}/logs` | Stream live logs for a running job |

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Returns `{"status": "ok"}` |

## Development

### Backend

```bash
# Install dependencies
pip install -e ".[web]"
# or: pixi install -e web

# Run with auto-reload
sweets server --reload
```

### Frontend

```bash
cd src/sweets/web/frontend

# Install npm dependencies
npm install

# Run dev server (proxies /api to localhost:8000)
npm run dev

# Build for production
npm run build
```

### Production

```bash
# Build frontend first
cd src/sweets/web/frontend && npm run build && cd -

# Run server (serves built frontend from dist/)
sweets server --host 0.0.0.0 --port 8000
```

## Implementation Phases

- [x] **Phase 1**: Foundation - FastAPI + Svelte scaffold, job CRUD, map AOI drawing
- [x] **Phase 2**: Live monitoring - WebSocket log streaming, progress updates
- [ ] **Phase 3**: Results visualization - COG tile serving, map overlays
- [ ] **Phase 4**: Remote execution - SSH/AWS job submission

## Configuration

The web UI stores its database at `~/.sweets/sweets.db` (SQLite).

Job configurations are stored as JSON and match the `sweets.core.Workflow` Pydantic model structure.
