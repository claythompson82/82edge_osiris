import asyncio
import datetime
import json
import os
import sys
import uuid
from unittest.mock import ANY, MagicMock, call, mock_open, patch

import pytest
from fastapi.testclient import TestClient

# Mock modules that may not be in the test environment to allow test collection.
sys.modules.setdefault("sentry_sdk", MagicMock())
sys.modules.setdefault("outlines", MagicMock())

# This entire test suite is currently skipped.
pytest.skip(
    "Feedback mechanism tests skipped due to environment limitations",
    allow_module_level=True,
)

from osiris.server import (
    PHI3_FEEDBACK_DATA_FILE,
    PHI3_FEEDBACK_LOG_FILE,
    app,
    load_recent_feedback,
)


# This test requires the _generate_phi3_json function to be imported for direct testing.
# Assuming it exists in osiris.server.
try:
    from osiris.server import _generate_phi3_json
except ImportError:
    _generate_phi3_json = MagicMock()


class TestFeedbackMechanism:
    """Test suite for the feedback logging and retrieval mechanism."""

    def setup_method(self):
        """Set up the test client and ensure a clean file state before each test."""
        self.client = TestClient(app)
        if os.path.exists(PHI3_FEEDBACK_DATA_FILE):
            os.remove(PHI3_FEEDBACK_DATA_FILE)
        if os.path.exists(PHI3_FEEDBACK_LOG_FILE):
            os.remove(PHI3_FEEDBACK_LOG_FILE)

    def teardown_method(self):
        """Clean up any files created during tests."""
        if os.path.exists(PHI3_FEEDBACK_DATA_FILE):
            os.remove(PHI3_FEEDBACK_DATA_FILE)
        if os.path.exists(PHI3_FEEDBACK_LOG_FILE):
            os.remove(PHI3_FEEDBACK_LOG_FILE)

    @patch("osiris.server._generate_hermes_text")
    @patch("osiris.server._generate_phi3_json")
    @patch("builtins.open", new_callable=mock_open)
    def test_log_propose_trade_adjustments(
        self, mock_file_open, mock_internal_phi3_gen, mock_hermes_gen
    ):
        """Test that a call to the proposal endpoint correctly logs the transaction."""
        mock_phi3_output = {
            "ticker": "TESTPHI3",
            "action": "adjust",
            "confidence": 0.9,
            "rationale": "Phi3 rationale",
        }
        mock_hermes_output = "Hermes assessment text."

        async def _mock_phi3(*args, **kwargs):
            return mock_phi3_output

        async def _mock_hermes(*args, **kwargs):
            return mock_hermes_output

        mock_internal_phi3_gen.side_effect = _mock_phi3
        mock_hermes_gen.side_effect = _mock_hermes

        response = self.client.post(
            "/propose_trade_adjustments/",
            json={"prompt": "Test prompt", "max_length": 50},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["phi3_proposal"] == mock_phi3_output
        assert data["hermes_assessment"] == mock_hermes_output

        mock_file_open.assert_called_once_with(PHI3_FEEDBACK_LOG_FILE, "a")

        write_calls = mock_file_open.return_value.write.call_args_list
        assert len(write_calls) >= 2, "Expected json.dump and newline writes"

        logged_json_str = write_calls[0].args[0]
        logged_json = json.loads(logged_json_str)

        assert uuid.UUID(logged_json["transaction_id"])
        assert datetime.datetime.fromisoformat(logged_json["timestamp"])
        assert logged_json["phi3_proposal"] == mock_phi3_output
        assert logged_json["hermes_assessment"] == mock_hermes_output
        assert write_calls[1].args[0] == "\n"

    @patch("builtins.open", new_callable=mock_open)
    def test_submit_feedback_endpoint(self, mock_file_open):
        """Test that submitting feedback correctly writes to the feedback data file."""
        tx_id = str(uuid.uuid4())
        sample_feedback_payload = {
            "transaction_id": tx_id,
            "feedback_type": "correction",
            "feedback_content": {"old": "bad", "new": "good"},
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "corrected_proposal": {
                "ticker": "TEST",
                "action": "adjust",
                "confidence": 1.0,
                "rationale": "Corrected",
            },
        }

        response = self.client.post("/feedback/phi3/", json=sample_feedback_payload)

        assert response.status_code == 200
        assert response.json()["message"] == "Feedback received successfully"

        mock_file_open.assert_called_once_with(PHI3_FEEDBACK_DATA_FILE, "a")

        write_calls = mock_file_open.return_value.write.call_args_list
        assert len(write_calls) >= 2

        logged_json_str = write_calls[0].args[0]
        logged_json = json.loads(logged_json_str)

        assert logged_json["transaction_id"] == tx_id
        assert logged_json["feedback_type"] == "correction"
        assert (
            logged_json["corrected_proposal"]["rationale"]
            == "Corrected"
        )

    @patch("builtins.print")
    def test_load_recent_feedback_logic(self, mock_print):
        """Test the logic of the load_recent_feedback function under various conditions."""
        mock_data_valid = [
            {"transaction_id": "1", "feedback_type": "rating", "corrected_proposal": None},
            {"transaction_id": "2", "feedback_type": "correction", "corrected_proposal": {"key": "val2"}},
            {"transaction_id": "3", "feedback_type": "correction", "corrected_proposal": {"key": "val3"}},
            {"transaction_id": "4", "feedback_type": "other", "corrected_proposal": {"key": "val4"}},
            {"transaction_id": "5", "feedback_type": "correction", "corrected_proposal": None},
            {"transaction_id": "6", "feedback_type": "correction", "corrected_proposal": {"key": "val6"}},
        ]
        mock_jsonl_string_valid = "\n".join(json.dumps(item) for item in mock_data_valid)

        # Scenario A: Valid feedback, get the last 2 valid items
        with patch("os.path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=mock_jsonl_string_valid)
        ):
            result = load_recent_feedback(max_examples=2)
            assert len(result) == 2
            assert result[0]["corrected_proposal"]["key"] == "val3"
            assert result[1]["corrected_proposal"]["key"] == "val6"

        # Scenario B: Empty file
        with patch("os.path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data="")
        ):
            assert len(load_recent_feedback()) == 0

        # Scenario C: File not found
        with patch("os.path.exists", return_value=False):
            assert len(load_recent_feedback()) == 0
            mock_print.assert_any_call(
                f"Feedback file {PHI3_FEEDBACK_DATA_FILE} not found. No feedback to load."
            )
        mock_print.reset_mock()

        # Scenario D: Corrupted JSON line
        mock_jsonl_corrupted = "this is not json\n" + json.dumps(mock_data_valid[1])
        with patch("os.path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=mock_jsonl_corrupted)
        ):
            result = load_recent_feedback(max_examples=1)
            assert len(result) == 1
            assert result[0]["corrected_proposal"]["key"] == "val2"
            assert any("Error decoding JSON" in call_args[0][0] for call_args in mock_print.call_args_list)
        mock_print.reset_mock()

        # Scenario E: I/O Error on read
        with patch("os.path.exists", return_value=True), patch(
            "builtins.open", mock_open()
        ) as m_open:
            m_open.side_effect = IOError("Disk full")
            assert len(load_recent_feedback()) == 0
            mock_print.assert_any_call(
                f"Error reading feedback file {PHI3_FEEDBACK_DATA_FILE}: Disk full"
            )

    @patch("outlines.generate.json")
    @patch("osiris.server.load_recent_feedback")
    @pytest.mark.asyncio
    async def test_prompt_augmentation_logic(
        self, mock_load_feedback, mock_outlines_gen_json_factory
    ):
        """Test that prompts are correctly augmented with recent feedback."""
        mock_generator_instance = MagicMock()
        async def async_mock_generator(*args, **kwargs):
            return {"ticker": "OUTLINES_TICK", "action": "generated"}

        mock_generator_instance.side_effect = async_mock_generator
        mock_outlines_gen_json_factory.return_value = mock_generator_instance

        mock_phi3_model = MagicMock()
        mock_phi3_tokenizer = MagicMock()
        original_prompt = "This is the original user prompt."

        # Scenario A: With feedback
        mock_feedback_items = [
            {"corrected_proposal": {"ticker": "FB_TICK1", "rationale": "Feedback 1"}},
            {"corrected_proposal": {"ticker": "FB_TICK2", "rationale": "Feedback 2"}},
        ]
        mock_load_feedback.return_value = mock_feedback_items

        await _generate_phi3_json(
            original_prompt, 100, mock_phi3_model, mock_phi3_tokenizer
        )

        mock_load_feedback.assert_called_once_with(max_examples=3)
        effective_prompt = mock_generator_instance.call_args[0][0]
        expected_intro = "Based on past feedback, here are examples of desired JSON outputs:"
        expected_outro = "\n\nNow, considering the above, please process the following request:\n"

        assert effective_prompt.startswith(expected_intro)
        assert json.dumps(mock_feedback_items[0]["corrected_proposal"], indent=2) in effective_prompt
        assert effective_prompt.endswith(f"{expected_outro}{original_prompt}")

        # Scenario B: No feedback
        mock_load_feedback.return_value = []
        mock_load_feedback.reset_mock()
        mock_generator_instance.reset_mock()

        await _generate_phi3_json(
            original_prompt, 100, mock_phi3_model, mock_phi3_tokenizer
        )

        effective_prompt_no_fb = mock_generator_instance.call_args[0][0]
        assert effective_prompt_no_fb == original_prompt