import pytest
from unittest.mock import AsyncMock, patch

from dgm_kernel.meta_loop import _verify_patch


@pytest.fixture
def sample_patch(tmp_path):
    target = tmp_path / "module.py"
    target.write_text("print('hello')\n")
    return {
        "target": str(target),
        "before": "print('hello')\n",
        "after": "print('hello')\n",
    }


@pytest.mark.asyncio
@patch("dgm_kernel.meta_loop._run_unit_tests", new_callable=AsyncMock)
@patch("dgm_kernel.meta_loop._lint_with_ruff", new_callable=AsyncMock)
async def test_verify_patch_accepts_good_patch(mock_ruff, mock_tests, sample_patch):
    mock_ruff.return_value = True
    mock_tests.return_value = True

    accepted = await _verify_patch([], sample_patch)

    assert accepted is True
    mock_ruff.assert_awaited_once_with(sample_patch["after"])
    mock_tests.assert_awaited_once_with(sample_patch["target"], sample_patch["after"])


@pytest.mark.asyncio
async def test_verify_patch_rejects_disallowed_call(sample_patch):
    sample_patch["after"] = "os.system('ls')"

    accepted = await _verify_patch([], sample_patch)

    assert accepted is False


@pytest.mark.asyncio
@patch("dgm_kernel.meta_loop._lint_with_ruff", new_callable=AsyncMock)
async def test_verify_patch_rejects_lint_failure(mock_ruff, sample_patch):
    mock_ruff.return_value = False

    accepted = await _verify_patch([], sample_patch)

    assert accepted is False
    mock_ruff.assert_awaited_once_with(sample_patch["after"])


@pytest.mark.asyncio
@patch("dgm_kernel.meta_loop._run_unit_tests", new_callable=AsyncMock)
@patch("dgm_kernel.meta_loop._lint_with_ruff", new_callable=AsyncMock)
async def test_verify_patch_rejects_test_failure(mock_ruff, mock_tests, sample_patch):
    mock_ruff.return_value = True
    mock_tests.return_value = False

    accepted = await _verify_patch([], sample_patch)

    assert accepted is False
    mock_ruff.assert_awaited_once_with(sample_patch["after"])
    mock_tests.assert_awaited_once_with(sample_patch["target"], sample_patch["after"])

