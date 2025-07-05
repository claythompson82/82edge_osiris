import unittest
import importlib.util
from pathlib import Path
import sys
import types

# Provide minimal stubs for optional heavy dependencies
pa_stub = types.ModuleType("pyarrow")
pa_stub.schema = lambda *_, **__: None
pa_stub.utf8 = lambda: None
pa_stub.int64 = lambda: None
pa_stub.float64 = lambda: None
sys.modules.setdefault("pyarrow", pa_stub)
# Stub out azr_planner modules used by orchestrator
azr_schemas = types.ModuleType("azr_planner.schemas")
azr_schemas.PlanningContext = object
azr_schemas.TradeProposal = object
sys.modules.setdefault("azr_planner.schemas", azr_schemas)

azr_engine = types.ModuleType("azr_planner.engine")
azr_engine.generate_plan = lambda *_args, **_kwargs: {}
sys.modules.setdefault("azr_planner.engine", azr_engine)

azr_utils = types.ModuleType("azr_planner.math_utils")
azr_utils.LR_V2_MIN_POINTS = 1
sys.modules.setdefault("azr_planner.math_utils", azr_utils)
import pydantic
if not hasattr(pydantic, "field_validator"):
    def _fv(*args, **kwargs):
        def dec(fn):
            return fn
        return dec

    setattr(pydantic, "field_validator", _fv)

module_path = Path(__file__).resolve().parents[2] / "osiris" / "intent_router.py"
spec = importlib.util.spec_from_file_location("osiris.intent_router", module_path)
router_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router_module)  # type: ignore[arg-type]

SelectedRoute = router_module.SelectedRoute
decide_route = router_module.decide_route
route_and_respond = router_module.route_and_respond


class TestIntentRouter(unittest.TestCase):
    def test_command_hits_orchestrator(self) -> None:
        route, intent, conf = decide_route("cmd: run diagnostics")
        self.assertEqual(route, SelectedRoute.ORCHESTRATOR)
        resp = route_and_respond("cmd: run diagnostics", {})
        self.assertEqual(resp, "[ORCHESTRATOR] cmd: run diagnostics")

    def test_quick_fact_phi3(self) -> None:
        route, intent, conf = decide_route("quick fact about python")
        self.assertEqual(route, SelectedRoute.PHI3)
        resp = route_and_respond("quick fact about python", {})
        self.assertEqual(resp, "[PHI3] quick fact about python")

    def test_nonsense_fallback(self) -> None:
        route, intent, conf = decide_route("asdfghjkl")
        self.assertEqual(route, SelectedRoute.UNKNOWN)
        resp = route_and_respond("asdfghjkl", {})
        self.assertEqual(resp, "I'm sorry, could you rephrase?")


if __name__ == "__main__":
    unittest.main()
