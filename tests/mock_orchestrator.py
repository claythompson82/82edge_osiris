import asyncio
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock, MagicMock
from common.otel_init import init_otel
from opentelemetry import trace

# Patch heavy dependencies before import
with (
    patch("osiris_policy.orchestrator.EventBus"),
    patch("osiris_policy.orchestrator.log_run"),
    patch("osiris_policy.orchestrator.init_advice_table"),
    patch("osiris_policy.orchestrator.market_tick_listener", new=AsyncMock()),
    patch(
        "osiris_policy.orchestrator.build_graph",
        return_value=MagicMock(ainvoke=AsyncMock(return_value={})),
    ) as _,
):
    from osiris_policy import orchestrator

# Initialize OTEL and run a span around main_async
init_otel()
tracer = trace.get_tracer(__name__)
args = SimpleNamespace(
    redis_url="redis://localhost:6379/0",
    market_channel="market.ticks",
    ticks_per_proposal=1,
)


async def run():
    with tracer.start_as_current_span("orchestrator.run"):
        await orchestrator.main_async(args)


if __name__ == "__main__":
    asyncio.run(run())
