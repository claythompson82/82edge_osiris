import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, ANY

from osiris_policy import orchestrator as policy_orchestrator


class TestMainAsync(unittest.IsolatedAsyncioTestCase):
    async def test_main_async_runs_listener(self):
        args = SimpleNamespace(
            redis_url="redis://test", market_channel="market.ticks", ticks_per_proposal=5
        )

        with (
            patch("osiris_policy.orchestrator.init_otel") as mock_init_otel,
            patch("osiris_policy.orchestrator.build_graph", return_value=MagicMock()) as mock_build,
            patch("osiris_policy.orchestrator.init_advice_table") as mock_init_table,
            patch("osiris_policy.orchestrator.market_tick_listener", new=AsyncMock()) as mock_listener,
        ):
            await policy_orchestrator.main_async(args)

            mock_init_otel.assert_called_once()
            mock_build.assert_called_once()
            mock_init_table.assert_called_once_with(
                db_path_str=policy_orchestrator.RISK_GATE_CONFIG["lancedb_path"]
            )
            mock_listener.assert_awaited_once()
            _, kwargs = mock_listener.call_args
            self.assertEqual(kwargs["redis_url"], args.redis_url)
            self.assertEqual(kwargs["market_channel"], args.market_channel)
            self.assertEqual(kwargs["ticks_per_proposal"], args.ticks_per_proposal)
            self.assertIs(kwargs["graph_app"], mock_build.return_value)

    async def test_init_advice_failure_stops_execution(self):
        args = SimpleNamespace(
            redis_url="redis://test", market_channel="market.ticks", ticks_per_proposal=1
        )

        with (
            patch("osiris_policy.orchestrator.init_otel"),
            patch("osiris_policy.orchestrator.build_graph", return_value=MagicMock()),
            patch(
                "osiris_policy.orchestrator.init_advice_table", side_effect=Exception("fail")
            ),
            patch("osiris_policy.orchestrator.market_tick_listener", new=AsyncMock()) as mock_listener,
        ):
            await policy_orchestrator.main_async(args)
            mock_listener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
