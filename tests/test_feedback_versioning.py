import datetime
import pytest
from osiris.server import FeedbackItem, submit_phi3_feedback

@pytest.mark.asyncio
async def test_submit_phi3_feedback_stores_version(mocker):
    # Patch the underlying module used by the server
    mocked_append = mocker.patch("llm_sidecar.db.append_feedback")
    feedback_item_default = FeedbackItem(
        transaction_id="tid1",
        feedback_type="correction",
        feedback_content="some content",
        timestamp="ts1",
    )
    await submit_phi3_feedback(feedback_item_default)
    mocked_append.assert_called_once()
    call_args_default = mocked_append.call_args[0][0]
    assert isinstance(call_args_default, dict)
    assert call_args_default.get("schema_version") == "1.0"
    assert call_args_default.get("transaction_id") == "tid1"

    mocked_append.reset_mock()

    feedback_item_custom = FeedbackItem(
        transaction_id="tid2",
        feedback_type="rating",
        feedback_content={"score": 1},
        timestamp="ts2",
        schema_version="custom_v2",
    )
    await submit_phi3_feedback(feedback_item_custom)
    mocked_append.assert_called_once()
    call_args_custom = mocked_append.call_args[0][0]
    assert isinstance(call_args_custom, dict)
    assert call_args_custom.get("schema_version") == "custom_v2"
    assert call_args_custom.get("transaction_id") == "tid2"

