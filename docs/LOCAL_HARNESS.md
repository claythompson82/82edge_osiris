# Local Harness

This guide walks through running the full Osiris stack on your workstation. It assumes Docker and Python 3.10+ are installed.

![Demo](assets/local_harness_demo.gif)

## 1. Clone the repo

```bash
git clone https://github.com/your-org/osiris.git
cd osiris
pip install -e .
```

## 2. Start infrastructure

Launch the LLM sidecar (and supporting services) via Docker Compose. In a second terminal, start the market data simulator.

```bash
# Start sidecar + dependencies
cd docker
docker compose up -d

# In another shell
python sim/pump_ticks.py --redis_url redis://localhost:6379/0 --speed 1x
```

## 3. Run the orchestrator

```bash
python -m osiris_policy.orchestrator
```

Watch Redis events in yet another terminal:

```bash
redis-cli --raw monitor | grep tick
```

## 4. Inspect results

Stop the orchestrator with `Ctrl+C` then query LanceDB for the run log.

```bash
python scripts/qshell.py runs --limit 5
```

## 5. Cleanup

```bash
docker compose down -v
```

