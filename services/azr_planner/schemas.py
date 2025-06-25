from pydantic import BaseModel
from typing import Dict, List, Literal

class PlannerState(BaseModel):
    alpha_free: float   # fraction (0–1) or absolute slots
    beta_free: float
    performance: Dict[str, float]  # e.g. {"deduction":0.72,...}

class Task(BaseModel):
    id: str
    kind: Literal["deduction", "induction", "abduction"]
    description: str
    resource: Literal["Alpha", "Beta"]

class Plan(BaseModel):
    tasks: List[Task]
    alpha_count: int
    beta_count: int
    priority: int  # 1–10
