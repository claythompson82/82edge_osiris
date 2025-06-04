from dataclasses import dataclass

@dataclass
class VerifiedPatch:
    id: str
    diff: str
    score: float
    status: str


def prove_patch(id: str, diff: str) -> VerifiedPatch:
    """
    Creates a VerifiedPatch object, auto-approving it with a score of 1.0.
    """
    return VerifiedPatch(id=id, diff=diff, score=1.0, status="APPROVED")
