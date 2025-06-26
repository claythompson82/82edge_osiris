# AZR Planner – Key Constraints (quick reference)

* **Latent-Risk metric**: see §3.2 in _azr_planner_design.pdf_
* **Primary objectives** (priority order)  
  1. Capital preservation  
  2. Volatility smoothing  
  3. Alpha generation
* **Allowed instruments**: CME micro-futures (MES, M2K), US sector ETFs, ETH options
* **Pydantic v2** must be used for any external schema.
* **Pure-math helpers** live in `services/azr_planner/math_utils.py`.

Full design: **docs/azr/azr_planner_design.pdf**
