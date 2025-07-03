from dgm_kernel.mutation_scheduler import MutationScheduler


def test_scheduler_progression() -> None:
    sched = MutationScheduler()
    rewards = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    strategies = [sched.next_strategy(r) for r in rewards]
    assert strategies[0] == "ASTInsertComment"
    assert strategies[1] == "ASTInsertComment"
    assert strategies[2] == "ASTRenameIdentifier"
    assert strategies[3] == "ASTRenameIdentifier"
    assert strategies[4] == "ASTRenameIdentifier"
    assert strategies[5] == "ASTInsertCommentAndRename"


def test_positive_resets_streak() -> None:
    sched = MutationScheduler()
    assert sched.next_strategy(-1.0) == "ASTInsertComment"
    assert sched.next_strategy(-1.0) == "ASTInsertComment"
    assert sched.next_strategy(1.0) == "ASTInsertComment"
    # after reset, need 3 more negatives to switch
    sched.next_strategy(-1.0)
    sched.next_strategy(-1.0)
    assert sched.next_strategy(-1.0) == "ASTRenameIdentifier"

