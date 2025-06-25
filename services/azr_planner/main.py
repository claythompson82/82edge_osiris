from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from services.azr_planner.logic import generate_plan
from services.azr_planner.schemas import PlannerState, Plan

app = FastAPI()
FastAPIInstrumentor().instrument_app(app)

@app.post("/plan", response_model=Plan)
def create_plan(state: PlannerState):
    """
    Generate and return a Plan based on the provided PlannerState.
    """
    return generate_plan(state)
