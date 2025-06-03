import pytest
import asyncio
from unittest.mock import patch, MagicMock
from pathlib import Path # Added for Path object if used by tested code

# Assuming _verify_patch is in dgm_kernel.meta_loop
# Adjust if structure is different or if it's moved to a new module
from dgm_kernel.meta_loop import _verify_patch

@pytest.fixture
def sample_trace():
    return {"id": "trace1", "data": "some_trace_data"}

@pytest.fixture
def sample_patch():
    return {
        "target": "osiris_policy/strategy.py",
        "before": "old_code",
        "after": "new_code_example", # Sample python code for pylint to check
        "rationale": "Test patch"
    }

@pytest.mark.asyncio
@patch('dgm_kernel.meta_loop.proofable_reward') # Mocking the imported function
@patch('dgm_kernel.meta_loop._get_pylint_score') # Mocking helper within the same module
async def test_verify_patch_accepts_good_patch(
    mock_get_pylint_score, mock_proofable_reward, sample_trace, sample_patch
):
    mock_proofable_reward.return_value = 1.0
    mock_get_pylint_score.return_value = 7.5

    accepted, score = await _verify_patch([sample_trace], sample_patch)

    assert accepted is True
    assert score == 7.5
    mock_proofable_reward.assert_called_once_with(sample_trace, sample_patch["after"])
    mock_get_pylint_score.assert_called_once_with(sample_patch["after"])

@pytest.mark.asyncio
@patch('dgm_kernel.meta_loop.proofable_reward')
@patch('dgm_kernel.meta_loop._get_pylint_score')
async def test_verify_patch_rejects_negative_reward(
    mock_get_pylint_score, mock_proofable_reward, sample_trace, sample_patch
):
    mock_proofable_reward.return_value = -0.5
    mock_get_pylint_score.return_value = 8.0

    accepted, score = await _verify_patch([sample_trace], sample_patch)

    assert accepted is False
    assert score == 8.0

@pytest.mark.asyncio
@patch('dgm_kernel.meta_loop.proofable_reward')
@patch('dgm_kernel.meta_loop._get_pylint_score')
async def test_verify_patch_rejects_low_pylint_score(
    mock_get_pylint_score, mock_proofable_reward, sample_trace, sample_patch
):
    mock_proofable_reward.return_value = 0.5
    mock_get_pylint_score.return_value = 4.0

    accepted, score = await _verify_patch([sample_trace], sample_patch)

    assert accepted is False
    assert score == 4.0

@pytest.mark.asyncio
@patch('dgm_kernel.meta_loop.proofable_reward')
@patch('dgm_kernel.meta_loop._get_pylint_score')
async def test_verify_patch_no_traces(
    mock_get_pylint_score, mock_proofable_reward, sample_patch
):
    accepted, score = await _verify_patch([], sample_patch) # Empty list for traces

    assert accepted is False
    assert score == 0.0
    mock_proofable_reward.assert_not_called()
    mock_get_pylint_score.assert_not_called()
