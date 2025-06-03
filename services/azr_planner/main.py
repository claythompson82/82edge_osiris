from fastapi import FastAPI

app = FastAPI(title="AZR Planner Service")

@app.get("/plan")
async def get_plan():
    """
    Stub endpoint that returns a NOP (No Operation) patch.
    """
    return {"PATCH": "NOP"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# To run directly for local testing: uvicorn services.azr_planner.main:app --reload --port 8001
