"""
Tiny FastAPI façade sufficient for the Osiris test-suite.
"""

from __future__ import annotations

import datetime as _dt
import json
import csv
import io
import time
from pathlib import Path
from typing import Any, Dict, Union, List, Optional # Added List, Optional

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse # For specific JSON responses
# Response class from Starlette, also available via fastapi.responses
from starlette.responses import Response as StarletteResponse
from pydantic import BaseModel

from llm_sidecar import db as _db
from llm_sidecar import hermes_plugin as _hermes_plugin
from llm_sidecar import loader as _loader
from llm_sidecar.db import _coerce_ts

# AZR-13 & AZR-15: Prometheus client imports (global for /metrics)
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# --------------------------------------------------------------------------- #
#  Loader wrappers
# --------------------------------------------------------------------------- #
from llm_sidecar.loader import LoadedAdapterComponents

def get_phi3_model_and_tokenizer() -> Optional[LoadedAdapterComponents]:
    return _loader.get_phi3_model_and_tokenizer()

def get_hermes_model_and_tokenizer() -> Optional[LoadedAdapterComponents]:
    return _loader.get_hermes_model_and_tokenizer()

# --------------------------------------------------------------------------- #
#  Patchable stubs
# --------------------------------------------------------------------------- #
async def _generate_hermes_text(prompt: str, max_length: int) -> str:
    return prompt[::-1][:max_length]

async def _generate_phi3_json(prompt: str, max_length: int) -> Dict[str, Any]:
    return {"output": prompt.upper()[:max_length]}

def text_to_speech(text: str) -> bytes:
    return b""

# --------------------------------------------------------------------------- #
#  FastAPI setup
# --------------------------------------------------------------------------- #
openapi_tags_metadata = [
    {"name": "AZR Planner", "description": "Alpha-Zero-Risk planner"},
    {"name": "AZR Planner Internal", "description": "Internal AZR Planner tools"}
]

app = fastapi.FastAPI(
    title="Osiris-stub", version="test", openapi_tags=openapi_tags_metadata
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_START_TS = time.time()

# --------------------------------------------------------------------------- #
#  Typed payloads
# --------------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    prompt: str; model_id: str | None = "hermes"; max_length: int | None = 256
class ScoreHermesRequest(BaseModel):
    proposal: Dict[str, Any]; context: str | None = None
class SpeakRequest(BaseModel):
    text: str

# --------------------------------------------------------------------------- #
#  Standard Routes
# --------------------------------------------------------------------------- #
@app.post("/generate/")
async def generate(req: GenerateRequest) -> Dict[str, Any]:
    if req.model_id == "phi3": return await _generate_phi3_json(req.prompt, req.max_length or 256)
    return {"output": await _generate_hermes_text(req.prompt, req.max_length or 256)}

@app.post("/score/hermes/")
async def score_hermes(req: ScoreHermesRequest) -> Dict[str, Any]:
    score = _hermes_plugin.score_with_hermes(req.proposal, req.context)
    _db.log_hermes_score(score); return {"score": score}

@app.post("/speak/")
async def speak(req: SpeakRequest) -> StarletteResponse:
    return StarletteResponse(content=text_to_speech(req.text), media_type="audio/wav")

@app.get("/health")
def health(adapter_date: bool = False) -> JSONResponse:
    latest_adapter: str | None = None
    if adapter_date: latest_dir = _loader.get_latest_adapter_dir(); latest_adapter = latest_dir.name if latest_dir else None
    phi3_ok = bool(get_phi3_model_and_tokenizer())
    hermes_ok = bool(get_hermes_model_and_tokenizer())
    return JSONResponse({"uptime": time.time()-_START_TS, "phi_ok": phi3_ok, "hermes_ok": hermes_ok,
                         "mean_hermes_score_last_24h": _db.get_mean_hermes_score_last_24h(),
                         "latest_adapter": latest_adapter})

@app.get("/metrics", response_class=StarletteResponse)
async def metrics() -> StarletteResponse:
    """Exposes Prometheus metrics."""
    return StarletteResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# --------------------------------------------------------------------------- #
#  Helper for feedback tests
# --------------------------------------------------------------------------- #
async def submit_phi3_feedback(item: Any) -> Dict[str, str]:
    schema_ver: str = getattr(item, "schema_version", None) or "1.0"
    coerced_ts_val: Union[int, float] = _coerce_ts(getattr(item, "timestamp", None))
    if coerced_ts_val is None:
        coerced_ts_val = time.time()

    content = getattr(item, "feedback_content", "{}")
    if not isinstance(content, str): content = json.dumps(content)

    # Trying to match Mypy's inferred signature based on its error messages
    row = _db.Phi3FeedbackSchema(transaction_id=item.transaction_id,
                                 feedback_type=item.feedback_type,
                                 feedback_content=content, # Reverted based on Mypy's "did you mean"
                                 schema_version=schema_ver,
                                 ts=int(coerced_ts_val)).model_dump() # Reverted to 'ts' and casting to int
    _db.append_feedback(row); return {"status": "OK", "stored_schema_version": schema_ver}

# --------------------------------------------------------------------------- #
#  AZR Planner Public API V1
# --------------------------------------------------------------------------- #
from azr_planner.schemas import PlanningContext as AZRPlanningContext, \
                                TradeProposal as AZRTradeProposal, \
                                DailyPNLReport as AZRDailyPNLReport
from azr_planner.engine import generate_plan as azr_generate_plan_engine
from fastapi import Query

router_azr_v1 = fastapi.APIRouter(prefix="/azr_api/v1", tags=["AZR Planner"])

@router_azr_v1.post("/propose_trade", response_model=AZRTradeProposal, summary="Propose a trade based on market context")
async def propose_trade_v1(ctx: AZRPlanningContext) -> AZRTradeProposal:
    return azr_generate_plan_engine(ctx)

@router_azr_v1.get("/pnl/daily", response_model=List[AZRDailyPNLReport], summary="Fetch last N daily P&L reports")
async def get_daily_pnl_reports(last_n: int = Query(7, ge=1, le=100)) -> List[AZRDailyPNLReport]:
    # Placeholder data for AZR-14
    dummy_reports: List[AZRDailyPNLReport] = []
    if last_n > 0:
        base_date = _dt.date.today()
        for i in range(min(last_n, 5)): # Max 5 dummy reports
            dummy_reports.append(AZRDailyPNLReport(
                date=base_date - _dt.timedelta(days=i), realized_pnl=100.0+i, unrealized_pnl=50.0-i,
                net_position_value=10000.0+(100*i), cash=50000.0-(50*i), total_equity=60000.0+(50*i),
                gross_exposure=15000.0+(100*i), net_exposure=5000.0+(50*i),
                cumulative_max_equity=60000.0+(100*i), current_drawdown=0.01*i,
                equity_curve_points=[60000.0+(j*10) for j in range(10)]))
    return dummy_reports
app.include_router(router_azr_v1)

# --------------------------------------------------------------------------- #
#  AZR Planner Internal Endpoints (OSIRIS_TEST=1)
# --------------------------------------------------------------------------- #
import os
if os.getenv("OSIRIS_TEST") == "1":
    from azr_planner.risk_gate import accept as risk_gate_accept, RiskGateConfig
    # AZR-15 Replay Imports
    from fastapi import UploadFile, File, HTTPException as FastAPIHTTPException # Ensure FastAPIHTTPException is imported
    import tempfile
    import shutil
    from azr_planner.replay.loader import load_bars as replay_load_bars
    from azr_planner.replay.runner import run_replay as replay_run_replay
    from azr_planner.replay.schemas import ReplayReport as AZRReplayReport
    from azr_planner.engine import generate_plan as azr_replay_default_planner
    from azr_planner.risk_gate import accept as azr_replay_default_risk_gate, RiskGateConfig as AZRReplayRiskGateConfig

    # Changed prefix for broader internal AZR tools
    router_azr_internal = fastapi.APIRouter(prefix="/azr_api/internal/azr", tags=["AZR Planner Internal"])

    @router_azr_internal.post("/planner/propose_trade", response_model=AZRTradeProposal,
                             responses={409: {"description": "Trade proposal rejected by Risk Gate"}})
    async def propose_trade_internal(ctx: AZRPlanningContext) -> Union[AZRTradeProposal, JSONResponse]:
        trade_proposal = azr_generate_plan_engine(ctx)
        # Server calls risk_gate.accept with default registry (None) and no specific db_table
        accepted, reason = risk_gate_accept(trade_proposal, db_table=None, cfg=None, registry=None)
        if not accepted:
            return JSONResponse(status_code=fastapi.status.HTTP_409_CONFLICT,
                                content={"detail": "risk_gate_reject", "reason": reason or "unknown_reason"})
        return trade_proposal

    @router_azr_internal.post("/replay", response_model=AZRReplayReport, summary="Run historical replay from uploaded Parquet file")
    async def run_historical_replay(file: UploadFile = File(..., description="Gzipped Parquet file containing bar data.")) -> AZRReplayReport:
        if not file.filename or not (file.filename.endswith(".parquet.gz") or file.filename.endswith(".parquet")):
            raise FastAPIHTTPException(status_code=400, detail="Invalid file type. Please upload a .parquet or .parquet.gz file.")

        tmp_file_name = None # Ensure it's defined for finally block
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).name if file.filename else ".parquet.gz") as tmp_file_path_obj:
                shutil.copyfileobj(file.file, tmp_file_path_obj)
                tmp_file_name = tmp_file_path_obj.name

            bar_stream = replay_load_bars(Path(tmp_file_name))

            report, _ = replay_run_replay(
                bar_stream=bar_stream, initial_equity=100_000.0,
                planner_fn=azr_replay_default_planner,
                risk_gate_fn=azr_replay_default_risk_gate,
                risk_gate_config=AZRReplayRiskGateConfig(),
                instrument_group_label=Path(file.filename).stem.split('_')[0] if file.filename else "server_replay",
                granularity_label="server_replay_bars"
            )
            return report
        except ValueError as ve:
            raise FastAPIHTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            print(f"Error during replay: {e}")
            raise FastAPIHTTPException(status_code=500, detail=f"Internal server error during replay: {type(e).__name__}")
        finally:
            if tmp_file_name and Path(tmp_file_name).exists(): Path(tmp_file_name).unlink()
            if file and hasattr(file, 'file') and hasattr(file.file, 'close'): file.file.close()

    app.include_router(router_azr_internal)

# --------------------------------------------------------------------------- #
#  AZR Planner Live Paper Trading (AZR-16)
# --------------------------------------------------------------------------- #
# These imports are conditional if OSIRIS_TEST is not set,
# but for AZR-16, we assume they are available as it's part of AZR planner dev.
import asyncio
from azr_planner.live.schemas import LiveConfig as AZRLiveConfig, \
                                     LivePosition as AZRLivePosition, \
                                     LivePnl as AZRLivePnl
from azr_planner.live.blotter import Blotter as AZRBlotter
# tasks.py will contain start_live_trading_task and stop_live_trading_task
import azr_planner.live.tasks as azr_live_tasks
import logging # For logging startup/shutdown messages

# Globals for live trading components
# These will be managed by startup/shutdown events
# Type hints for Optional globals
_live_blotter_instance: Optional[AZRBlotter] = None
_live_trading_main_task: Optional[asyncio.Task[None]] = None # Renamed and added type parameter
_live_config_instance: Optional[AZRLiveConfig] = None

@app.on_event("startup")
async def on_app_startup() -> None: # Added return type
    """Handles application startup events, including AZR live trading task."""
    global _live_blotter_instance, _live_trading_main_task, _live_config_instance
    logger = logging.getLogger("uvicorn.error") # Use uvicorn's logger or configure a new one

    if os.getenv("OSIRIS_TEST") == "1": # Only start live trading if in test/dev mode for AZR
        logger.info("AZR Live Trading: OSIRIS_TEST=1 detected, initializing live paper trading task.")
        # Default configuration for live trading
        # In a real scenario, this would come from a config file or env variables.
        _live_config_instance = AZRLiveConfig(
            symbol="MES", # Default to MES, consistent with Instrument.MES.value
            initial_equity=100_000.0,
            max_risk_per_trade_pct=0.01,      # 1% risk per trade
            max_drawdown_pct_account=0.10     # 10% max account drawdown
        )

        # Pass app instance to start_live_trading_task for it to store blotter on app.state
        await azr_live_tasks.start_live_trading_task(app, _live_config_instance)

        # Retrieve instances from app.state if they were stored there by start_live_trading_task
        # This makes them available globally within this module for endpoint access.
        if hasattr(app.state, 'live_blotter') and isinstance(app.state.live_blotter, AZRBlotter):
            _live_blotter_instance = app.state.live_blotter
        if hasattr(app.state, 'live_trading_task') and isinstance(app.state.live_trading_task, asyncio.Task):
            _live_trading_main_task = app.state.live_trading_task

        if not _live_blotter_instance:
            logger.warning("AZR Live Trading: Blotter instance not found after start_live_trading_task.")
    else:
        logger.info("AZR Live Trading: OSIRIS_TEST not set or not '1', live paper trading task will not start.")


@app.on_event("shutdown")
async def on_app_shutdown() -> None: # Added return type
    """Handles application shutdown events, including stopping AZR live trading task."""
    logger = logging.getLogger("uvicorn.error")
    logger.info("AZR Live Trading: Application shutdown sequence started.")
    if os.getenv("OSIRIS_TEST") == "1" and _live_trading_main_task is not None:
        logger.info("AZR Live Trading: Stopping live paper trading task...")
        await azr_live_tasks.stop_live_trading_task()
        logger.info("AZR Live Trading: Live paper trading task shutdown complete.")
    else:
        logger.info("AZR Live Trading: Live paper trading task was not running or OSIRIS_TEST not set.")

# Define new router for AZR Live V1 API
router_azr_live_v1 = fastapi.APIRouter(prefix="/azr_api/v1/live", tags=["AZR Planner Live"])

@router_azr_live_v1.get("/positions", response_model=List[AZRLivePosition], summary="Get current live paper trading positions")
async def get_live_positions() -> List[AZRLivePosition]:
    if _live_blotter_instance is None:
        # Check app.state as a fallback if startup sequence had issues with global assignment timing
        blotter_from_state = getattr(app.state, 'live_blotter', None)
        if not isinstance(blotter_from_state, AZRBlotter):
            raise FastAPIHTTPException(status_code=503, detail="Live trading service not available or not initialized.")
        current_blotter = blotter_from_state
    else:
        current_blotter = _live_blotter_instance

    return await current_blotter.get_current_positions()

@router_azr_live_v1.get("/pnl", response_model=AZRLivePnl, summary="Get current live paper trading P&L")
async def get_live_pnl() -> AZRLivePnl:
    if _live_blotter_instance is None:
        blotter_from_state = getattr(app.state, 'live_blotter', None)
        if not isinstance(blotter_from_state, AZRBlotter):
            raise FastAPIHTTPException(status_code=503, detail="Live trading service not available or not initialized.")
        current_blotter = blotter_from_state
    else:
        current_blotter = _live_blotter_instance

    # get_current_pnl in blotter now calculates MTM based on its internally updated unrealized P&Ls
    return await current_blotter.get_current_pnl()

# Include the new router only if OSIRIS_TEST is set, as live trading is a dev/test feature for now
if os.getenv("OSIRIS_TEST") == "1":
    app.include_router(router_azr_live_v1)

# --------------------------------------------------------------------------- #
#  DGM Kernel Endpoints
# --------------------------------------------------------------------------- #
from dgm_kernel import meta_loop

router_dgm_v1 = fastapi.APIRouter(prefix="/dgm_api/v1", tags=["DGM Kernel"])


@router_dgm_v1.get("/traces/export")
async def export_traces(limit: int = 100, format: str = "jsonl") -> StarletteResponse:
    fmt = format.lower()
    if fmt not in {"jsonl", "csv"}:
        raise fastapi.HTTPException(status_code=400, detail="invalid_format")

    raw: list[str] = []
    for _ in range(limit):
        item = meta_loop.REDIS.lpop(meta_loop.TRACE_QUEUE)
        if item is None:
            break
        raw.append(item)

    if fmt == "jsonl":
        async def jsonl_gen() -> Any:
            for line in raw:
                yield line + "\n"

        headers = {"Content-Disposition": "attachment; filename=traces.jsonl"}
        return fastapi.responses.StreamingResponse(jsonl_gen(), media_type="application/json", headers=headers)

    traces = []
    for txt in raw:
        try:
            traces.append(json.loads(txt))
        except Exception:
            traces.append({})

    fieldnames = sorted({k for t in traces for k in t.keys()})

    async def csv_gen() -> Any:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        yield buf.getvalue()
        buf.seek(0); buf.truncate(0)
        for row in traces:
            writer.writerow(row)
            yield buf.getvalue()
            buf.seek(0); buf.truncate(0)

    headers = {"Content-Disposition": "attachment; filename=traces.csv"}
    return fastapi.responses.StreamingResponse(csv_gen(), media_type="text/csv", headers=headers)


app.include_router(router_dgm_v1)
