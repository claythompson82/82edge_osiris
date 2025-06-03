from typing import Optional, Dict
import logging

log = logging.getLogger(__name__)

def proofable_reward(trace: Dict, patch_content: Optional[str] = None) -> float:
    """
    Stub for calculating a proofable reward.
    For DGM-004, this returns a fixed value.
    The patch_content argument is included for future use.
    """
    log.debug(f"proofable_reward called with trace: {trace.get('id', 'unknown_id')}, patch_content (first 50 chars): {patch_content[:50] if patch_content else 'None'}")
    # In a real scenario, this would involve complex logic.
    # For the stub, return a fixed positive value.
    return 0.5
