# src/osiris/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union
import datetime

class FeedbackItem(BaseModel):
    transaction_id: str
    feedback_type: str
    feedback_content: Union[str, Dict[str, Any]]
    corrected_proposal: Optional[Dict[str, Any]] = None
    schema_version: str = "1.0"
    timestamp: str

class ScoreRequest(BaseModel):
    proposal: dict
    context: str

class GenerateRequest(BaseModel):
    prompt: str
    model_id: str = "hermes"
    max_length: int = 256

# If you ever want a LanceDB "table schema" for more static typing, add that here!
# For example, here's a schema for harvested feedback (with a nanosecond timestamp):
class FeedbackSchemaForHarvest(BaseModel):
    transaction_id: str
    timestamp: str
    feedback_type: str
    feedback_content: Union[str, Dict[str, Any]]
    corrected_proposal: Optional[str] = None
    schema_version: str
    when: int  # Nanoseconds since epoch

# You can add more as needed for any new endpoints or migrations!
