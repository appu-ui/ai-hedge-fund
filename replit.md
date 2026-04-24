# AI Hedge Fund

A proof-of-concept AI-powered hedge fund that uses multiple LLM-driven agents (Warren Buffett, Ben Graham, Cathie Wood, etc.) to make simulated trading decisions. Educational use only — the system does not place real trades.

## Architecture

- **Backend**: FastAPI app at `app/backend/main.py` (Python 3.11), exposing REST endpoints and SSE streams for the hedge-fund agents, backtester, flow CRUD, language-model providers, API key management, and Ollama integration.
  - Persistence: SQLite (`app/backend/hedge_fund.db`) via SQLAlchemy.
  - Routers (mounted at `/`): `health`, `hedge-fund`, `storage`, `flows`, `flow_runs`, `ollama`, `language-models`, `api-keys`.
  - Agent / graph code lives under `src/` (`src/agents/...`, `src/graph/...`, `src/llm/...`, etc.) and is imported by the backend.
- **Frontend**: Vite + React + TypeScript SPA in `app/frontend/` (uses `@xyflow/react`, shadcn-ui, Tailwind).

## Replit setup

- Single workflow (`Start application`) runs both services for development:
  - Backend: `uvicorn` on `127.0.0.1:8000`
  - Frontend: `vite` on `0.0.0.0:5000` with `allowedHosts: true` and a proxy that forwards `/hedge-fund`, `/flows`, `/storage`, `/ollama`, `/language-models`, `/api-keys`, and `/ping` to the backend.
- Frontend services use relative URLs by default (`VITE_API_URL` is empty), so all browser traffic flows through the Vite proxy in dev and through the FastAPI app in production.
- API keys for LLM providers (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`, `OPENROUTER_API_KEY`, `GIGACHAT_API_KEY`, Azure OpenAI vars) and `FINANCIAL_DATASETS_API_KEY` can be configured either through environment variables or the in-app Settings page (stored in SQLite).

## Market data providers

Two providers are wired into `src/tools/api.py`:

1. **financialdatasets.ai** (default when `FINANCIAL_DATASETS_API_KEY` is set). Free for AAPL, GOOGL, MSFT, NVDA, TSLA without a key; paid for everything else.
2. **yfinance** (`src/tools/yfinance_provider.py`) — free, no key, ~15-min delayed Yahoo Finance data. Used automatically when no `FINANCIAL_DATASETS_API_KEY` is configured. Force one or the other with `DATA_PROVIDER=yfinance` or `DATA_PROVIDER=financialdatasets`. Maps Yahoo's income statement / balance sheet / cashflow / insider transactions / news into the project's existing Pydantic models (`Price`, `FinancialMetrics`, `LineItem`, `InsiderTrade`, `CompanyNews`). Some fields Yahoo doesn't expose (e.g. growth rates, certain ratios) are returned as `None`; agents already handle missing values gracefully.

## Running locally

The `Start application` workflow runs `bash scripts/dev.sh`, which starts both processes. Visit the preview on port 5000.

## Deployment

Configured for VM deployment:

- Build: `bash scripts/build.sh` — installs frontend deps and runs `npm run build`, producing `app/frontend/dist`.
- Run: `bash scripts/start.sh` — launches FastAPI on `0.0.0.0:5000`. When `app/frontend/dist` exists, FastAPI serves the built SPA (with client-side routing fallback) from the same port as the API, so the deployed app is a single-process, single-port service.

## Notable files

- `pyproject.toml` — Poetry-managed Python deps; installed editable into the Replit Python environment.
- `app/frontend/vite.config.ts` — Vite dev server (host `0.0.0.0`, port `5000`, allow all hosts) plus the dev proxy to the backend.
- `app/backend/main.py` — FastAPI app, CORS, route registration, and production static-file serving.
- `scripts/dev.sh` — runs backend + Vite dev server side-by-side.
- `scripts/start.sh` / `scripts/build.sh` — production build & run entrypoints used by the deployment.

## Notes

- The original repo expects a separate frontend (5173) and backend (8000) and uses CORS + hard-coded `http://localhost:8000` URLs in a few client files. To work behind Replit's preview proxy, those URLs were converted to relative paths so the Vite proxy (dev) and FastAPI static handler (prod) can route them on the same origin.
- Filename casing: `App.tsx` previously imported `./components/layout`, but the actual file on disk is `Layout.tsx`. The import was updated to match — important on case-sensitive Linux file systems.
