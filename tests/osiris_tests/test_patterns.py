import importlib.util
import sys
from pathlib import Path


def _load_router() -> type:
    path = Path(__file__).resolve().parents[2] / "src" / "osiris" / "intent_router.py"
    spec = importlib.util.spec_from_file_location("intent_router", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.IntentRouter

IntentRouter = _load_router()


def test_turn_light_on() -> None:
    name, route = IntentRouter.get_intent("turn on the kitchen lights")
    assert (name, route) == ("TURN_LIGHT_ON", "ORCHESTRATOR")


def test_timer_set() -> None:
    name, route = IntentRouter.get_intent("set a timer for 5 minutes")
    assert (name, route) == ("TIMER_SET", "ORCHESTRATOR")


def test_joke() -> None:
    name, route = IntentRouter.get_intent("tell me a joke")
    assert (name, route) == ("JOKE", "ORCHESTRATOR")


def test_general_qa() -> None:
    name, route = IntentRouter.get_intent("who is the president of France?")
    assert (name, route) == ("GENERAL_QA", "ORCHESTRATOR")


def test_quick_fact() -> None:
    name, route = IntentRouter.get_intent("give me a quick fact about space")
    assert (name, route) == ("QUICK_FACT", "ORCHESTRATOR")


def test_unknown() -> None:
    name, route = IntentRouter.get_intent("this does not match anything")
    assert (name, route) == ("UNKNOWN", "UNKNOWN")
