# Osiris Cookbook

The Osiris Cookbook collects small, runnable examples that show how to interact with the running services. Each recipe assumes that the `osiris.server` FastAPI sidecar is reachable (default `http://localhost:8000`) and that Redis is running for event bus communication.


## Moving Average Crossover Listener

A classic example strategy is to listen for a fast moving average crossing a slow one. The snippet below buffers market ticks from Redis and, on every crossover event, asks the sidecar to generate a trade proposal.

### Steps
1. Start the sidecar service:
   ```bash
   uvicorn osiris.server:app --reload
   ```
2. Run the orchestrator in event-driven mode so ticks trigger workflows:
   ```bash
   python -m osiris_policy.orchestrator --redis_url redis://localhost:6379/0 --market_channel market.ticks
   ```
3. Publish mock ticks (or connect to real data) using `scripts/ci_tick_publisher.py`:
   ```bash
   python scripts/ci_tick_publisher.py --duration 60 --channel market.ticks
   ```
4. Use the example listener below to detect crossovers and request trade ideas.

### Python snippet
```python
import json
import redis
import httpx

FAST = []
SLOW = []
SIDE_URL = "http://localhost:8000"

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
sub = r.pubsub()
sub.subscribe("market.ticks")

for message in sub.listen():
    if message['type'] != 'message':
        continue
    tick = json.loads(message['data'])
    price = tick['close']
    FAST.append(price)
    SLOW.append(price)
    if len(FAST) > 5:
        FAST.pop(0)
    if len(SLOW) > 20:
        SLOW.pop(0)
    if len(FAST) == 5 and len(SLOW) == 20:
        fast_ma = sum(FAST) / len(FAST)
        slow_ma = sum(SLOW) / len(SLOW)
        if fast_ma > slow_ma:
            prompt = f"Generate a JSON trade proposal to go long when price is {price:.2f}"
            resp = httpx.post(f"{SIDE_URL}/propose_trade_adjustments/", json={"prompt": prompt, "max_length": 300})
            print(resp.json())
```
The response contains the generated proposal and Hermes assessment similar to:
```json
{
  "transaction_id": "abc123",
  "phi3_proposal": {"action": "buy", "size": 1.0},
  "hermes_assessment": "Looks reasonable"
}
```

## Asset Sentiment Monitor

Hermes can be prompted to summarize sentiment for any asset. The following shell commands periodically request a summary and log it to a file.

### Steps
1. Ensure `curl` is installed and the sidecar is running.
2. Use a simple loop to call `/generate/hermes/` every few minutes:
   ```bash
   while true; do
       curl -X POST http://localhost:8000/generate/hermes/ \
            -H 'Content-Type: application/json' \
            -d '{"prompt": "Summarize current sentiment for BTC/USD", "max_length": 120}' \
            >> sentiment.log
       sleep 300
   done
   ```
3. Open `sentiment.log` to review the text summaries.

Expected output lines look like:
```json
{"generated_text":"Market chatter leans slightly bullish ..."}
```

## Exporting Past Proposals

Osiris stores feedback and proposal data in LanceDB. You can export recent records with the helper script in `scripts/` and analyse them using Python.

### Steps
1. Harvest the last seven days of records:
   ```bash
   python scripts/harvest_feedback.py --days-back 7 --out feedback.jsonl
   ```
2. Inspect the JSONL file or load it into pandas:
   ```python
   import pandas as pd
   df = pd.read_json('feedback.jsonl', lines=True)
   print(df.head())
   ```
3. Each entry contains the prompt text and the corrected proposal produced during the workflow.

Example entry:
```json
{"prompt": "Assess this trade", "response": "{\n  \"action\":\"buy\"...}"}
```

These examples are intentionally small to highlight the APIs. Refer to `docs/` and the source code for more advanced usage.
