# 🧠 OSIRIS AI TRADING SYSTEM — PROJECT OVERVIEW

---

## 🔧 SYSTEM SNAPSHOT

**Components:**
- `llm_sidecar/` – Model inference, reward functions, patch logic
- `dgm_kernel/` – Self-improvement loop (Generate → Prove → Apply)
- `azr_planner/` – Curriculum generation via AZR self-play
- `Dropzilla + Surgezilla` – Momentum signal agents (external)
- `verifier_daemon/` – TLA+/Z3 patch safety gate
- `helm/` – Full deployment stack
- `tests/` – All unit and integration tests
- `docker/` – Local runner configs for Tempo, MuseTalk, etc.

---

## 🛣️ ROADMAP

| Milestone                          | Status     | GitHub Issue/PR                 |
| ---------------------------------- | ---------- | ------------------------------- |
| Proofable reward implementation   | ✅ Done     | #91                             |
| Meta-loop scaffold                 | ✅ Done     | #92                             |
| OTEL integration & test           | 🔄 In prog | #OBS-042b                       |
| Tempo observability stack         | ⏳ Next     | #OBS-043 → #OBS-047             |
| Full curriculum RL integration    | ⏳ Planned | (TBD)                           |

---

## 📚 RESEARCH FILES

- `docs/research/`
  - `PAC-Bayesian Sharpe Ratio Uplift Estimator.txt`
  - `A Provably Safe DGM Kernel.txt`
  - `Proof-Guarded Co-evolution.txt`
  - `Provable Curriculum RL Integrating DGM with AZR.txt`
  - `Patch Provability via Reinforcement Learning.txt`

---

## 🤖 WORKING WITH O3 / JULES / GPT

Start any ChatGPT session like this:

```text
Repo: https://github.com/claythompson82/82edge_osiris  
Context: docs/PROJECT_OVERVIEW.md  
Today's goal: <fill this in>
