def compute_mse(preds, trues):
    """
    Mean Squared Error between two equal-length lists of numbers.
    """
    assert len(preds) == len(trues), "predictions/truths length mismatch"
    n = len(preds)
    return sum((p - t) ** 2 for p, t in zip(preds, trues)) / n
