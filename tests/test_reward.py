from llm_sidecar.reward import proofable_reward

def test_zero_reward_on_empty_trade():
    trade = {
        "ticker": "XYZ", "side": "LONG",
        "entry_price": 0, "exit_price": None,
        "entry_ts": "2025-06-03T00:00:00Z", "exit_ts": None,
        "pnl_pct": 0.0, "confidence": 0.5,
    }
    market = {"price_series": [], "ts_series": []}
    assert proofable_reward(trade, market) == 0.0
