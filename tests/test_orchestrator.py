import asyncio
import json
import unittest
from unittest.mock import patch, AsyncMock, MagicMock, call
import io
import sys
from contextlib import redirect_stdout

# Modules to test
from osiris_policy import orchestrator as policy_orchestrator
from llm_sidecar.db import (
    OrchestratorRunSchema,
)  # For type checking in log_run assertion


# Helper to create mock HTTP responses
def _mock_response(
    status_code=200, json_data=None, text_data=None, raise_for_status=None
):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json = MagicMock(return_value=json_data)
    mock_resp.text = text_data
    if raise_for_status:
        mock_resp.raise_for_status = MagicMock(side_effect=raise_for_status)
    else:
        mock_resp.raise_for_status = MagicMock()
    return mock_resp


# Predefined successful responses for mocking
mock_phi3_success_response_data = {
    "ticker": "TEST",
    "action": "adjust",
    "side": "LONG",
    "new_stop_pct": 0.05,
    "new_target_pct": 0.1,
    "confidence": 0.8,
    "rationale": "Test rationale from phi3",
}

mock_pta_success_response_data = {  # PTA: Propose Trade Adjustments
    "transaction_id": "test-tx-123",
    "phi3_proposal": mock_phi3_success_response_data,  # PTA endpoint returns its own phi3 proposal
    "hermes_assessment": "Hermes assessment looks good.",
}


class TestOrchestratorCLI(unittest.IsolatedAsyncioTestCase):

    @patch("osiris_policy.orchestrator.requests.post")
    @patch("osiris_policy.orchestrator.EventBus")
    @patch("osiris_policy.orchestrator.log_run")  # Patching where it's used
    async def test_cli_successful_run(
        self, mock_log_run, MockEventBus, mock_requests_post
    ):
        # --- Setup Mocks ---
        # Mock EventBus instance and its methods
        mock_event_bus_instance = MockEventBus.return_value
        mock_event_bus_instance.connect = AsyncMock()
        mock_event_bus_instance.publish = AsyncMock()
        mock_event_bus_instance.close = AsyncMock()

        # Mock requests.post responses
        # First call (phi3), Second call (propose_trade_adjustments)
        mock_requests_post.side_effect = [
            _mock_response(status_code=200, json_data=mock_phi3_success_response_data),
            _mock_response(status_code=200, json_data=mock_pta_success_response_data),
        ]

        # --- Execute CLI command (simulated by calling main_async) ---
        test_query = "initiate test policy for XYZ"

        # Capture stdout
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            await policy_orchestrator.main_async(test_query)

        cli_output_str = captured_output.getvalue()

        # --- Assertions ---
        # 1. requests.post calls
        self.assertEqual(mock_requests_post.call_count, 2)
        phi3_call_args = mock_requests_post.call_args_list[0]
        pta_call_args = mock_requests_post.call_args_list[1]
        self.assertTrue(
            "http://localhost:8000/generate?model_id=phi3" in phi3_call_args[0][0]
        )
        self.assertTrue(
            "http://localhost:8000/propose_trade_adjustments" in pta_call_args[0][0]
        )

        # 2. EventBus calls
        mock_event_bus_instance.connect.assert_called_once()

        # Check publish calls (order might vary if asyncio.gather is used, but here it's sequential)
        # Expected payloads for publish:
        # First: phi3_proposal (from PTA response)
        # Second: hermes_assessment (from PTA response)
        # The orchestrator's publish_events_node uses pta_response["phi3_proposal"] for "phi3.proposal.created"
        expected_phi3_event_payload = json.dumps(
            mock_pta_success_response_data["phi3_proposal"]
        )
        expected_assessment_event_payload = json.dumps(
            {
                "proposal_transaction_id": mock_pta_success_response_data[
                    "transaction_id"
                ],
                "assessment_text": mock_pta_success_response_data["hermes_assessment"],
            }
        )

        # Using any_order=True because the exact order of publish calls might be flexible
        # if they were, for example, gathered. In current code, they are sequential.
        mock_event_bus_instance.publish.assert_has_calls(
            [
                call("phi3.proposal.created", expected_phi3_event_payload),
                call("phi3.proposal.assessed", expected_assessment_event_payload),
            ],
            any_order=False,
        )  # Sequential in current implementation
        self.assertEqual(mock_event_bus_instance.publish.call_count, 2)
        mock_event_bus_instance.close.assert_called_once()

        # 3. log_run call
        mock_log_run.assert_called_once()
        args, _ = mock_log_run.call_args
        logged_run_data: OrchestratorRunSchema = args[0]

        self.assertEqual(logged_run_data.input_query, test_query)
        self.assertEqual(logged_run_data.status, "SUCCESS")
        self.assertIsNone(logged_run_data.error_message)
        # The final_output in OrchestratorRunSchema is a dict representation of the CLI output string
        # The CLI output string itself is final_state_dict["final_output"]
        # The orchestrator parses this string back to dict for log_run.
        expected_logged_output_data = {
            "status": "success",
            "data": mock_pta_success_response_data,
        }
        if logged_run_data.final_output:  # final_output is Optional[Dict]
            self.assertEqual(logged_run_data.final_output.get("status"), "success")
            self.assertEqual(
                logged_run_data.final_output.get("data"), mock_pta_success_response_data
            )
        else:
            self.fail("logged_run_data.final_output was None")

        # 4. CLI output (stdout)
        # The CLI output is the 'final_output' field from the state, which is a JSON string.
        # This JSON string contains {"status": "success", "data": pta_response}
        self.assertTrue(
            mock_pta_success_response_data["transaction_id"] in cli_output_str
        )
        self.assertTrue(
            mock_pta_success_response_data["hermes_assessment"] in cli_output_str
        )
        try:
            cli_output_json = json.loads(cli_output_str)
            self.assertEqual(cli_output_json["status"], "success")
            self.assertEqual(cli_output_json["data"], mock_pta_success_response_data)
        except json.JSONDecodeError:
            self.fail(f"CLI output was not valid JSON: {cli_output_str}")

    @patch("osiris_policy.orchestrator.requests.post")
    @patch(
        "osiris_policy.orchestrator.EventBus"
    )  # Mock EventBus to prevent actual calls
    @patch("osiris_policy.orchestrator.log_run")
    async def test_cli_phi3_http_error(
        self, mock_log_run, MockEventBus, mock_requests_post
    ):
        # --- Setup Mocks ---
        mock_event_bus_instance = MockEventBus.return_value  # Prevent EventBus activity
        mock_event_bus_instance.connect = AsyncMock()
        mock_event_bus_instance.publish = AsyncMock()
        mock_event_bus_instance.close = AsyncMock()

        # Simulate HTTP error from Phi-3 call
        http_error = requests.exceptions.HTTPError("Test HTTP Error from Phi-3")
        mock_requests_post.side_effect = [
            _mock_response(status_code=500, raise_for_status=http_error)
        ]

        # --- Execute ---
        test_query = "query that causes phi3 error"
        captured_output = io.StringIO()
        with (
            redirect_stdout(captured_output),
            self.assertLogs(
                logger="osiris_policy.orchestrator", level="ERROR"
            ) as log_watcher,
        ):
            await policy_orchestrator.main_async(test_query)

        cli_output_str = captured_output.getvalue()

        # --- Assertions ---
        # 1. requests.post called once for Phi-3
        self.assertEqual(mock_requests_post.call_count, 1)

        # 2. EventBus.publish should not be called as pipeline fails early
        mock_event_bus_instance.publish.assert_not_called()

        # 3. log_run call
        mock_log_run.assert_called_once()
        args, _ = mock_log_run.call_args
        logged_run_data: OrchestratorRunSchema = args[0]

        self.assertEqual(logged_run_data.input_query, test_query)
        self.assertEqual(logged_run_data.status, "FAILURE")
        self.assertIsNotNone(logged_run_data.error_message)
        self.assertTrue("HTTP request to Phi-3 failed" in logged_run_data.error_message)

        # 4. CLI output should contain error info
        self.assertTrue("error" in cli_output_str.lower())
        self.assertTrue("HTTP request to Phi-3 failed" in cli_output_str)

        # 5. Check logs for error messages
        self.assertTrue(
            any("HTTP request to Phi-3 failed" in msg for msg in log_watcher.output)
        )

    @patch("osiris_policy.orchestrator.requests.post")
    @patch("osiris_policy.orchestrator.EventBus")
    @patch("osiris_policy.orchestrator.log_run")
    async def test_cli_event_publish_error(
        self, mock_log_run, MockEventBus, mock_requests_post
    ):
        # --- Setup Mocks ---
        mock_event_bus_instance = MockEventBus.return_value
        mock_event_bus_instance.connect = AsyncMock()
        mock_event_bus_instance.publish = AsyncMock(
            side_effect=RedisError("Simulated Redis Publish Error")
        )
        mock_event_bus_instance.close = AsyncMock()  # Should still be called

        mock_requests_post.side_effect = [
            _mock_response(status_code=200, json_data=mock_phi3_success_response_data),
            _mock_response(status_code=200, json_data=mock_pta_success_response_data),
        ]

        # --- Execute ---
        test_query = "query causing event publish error"
        captured_output = io.StringIO()
        # Expect an error log from orchestrator.publish_events_node and potentially from main_async
        with (
            redirect_stdout(captured_output),
            self.assertLogs(
                logger="osiris_policy.orchestrator", level="ERROR"
            ) as log_watcher,
        ):
            await policy_orchestrator.main_async(test_query)

        cli_output_str = captured_output.getvalue()

        # --- Assertions ---
        # 1. log_run should be called
        mock_log_run.assert_called_once()
        args, _ = mock_log_run.call_args
        logged_run_data: OrchestratorRunSchema = args[0]

        self.assertEqual(logged_run_data.input_query, test_query)
        # Status is SUCCESS because core pipeline (LLM calls) finished. Event publish error is secondary.
        # The error from event publishing is captured in final_output and error_message.
        self.assertEqual(logged_run_data.status, "SUCCESS")
        self.assertIsNotNone(logged_run_data.error_message)
        self.assertTrue(
            "RedisError during event publishing" in logged_run_data.error_message
        )

        # Check that final_output in DB log contains the event_publishing_error field
        self.assertIsNotNone(logged_run_data.final_output)
        if logged_run_data.final_output:  # for type checker
            self.assertTrue("event_publishing_error" in logged_run_data.final_output)
            self.assertTrue(
                "RedisError during event publishing"
                in logged_run_data.final_output["event_publishing_error"]
            )

        # 2. CLI output should contain original data + event_publishing_error
        self.assertTrue(
            mock_pta_success_response_data["transaction_id"] in cli_output_str
        )
        self.assertTrue("RedisError during event publishing" in cli_output_str)

        # 3. EventBus.close should still be called
        mock_event_bus_instance.close.assert_called_once()

        # 4. Check logs
        self.assertTrue(
            any(
                "RedisError during event publishing" in msg
                for msg in log_watcher.output
            )
        )

    @patch("osiris_policy.orchestrator.requests.post")
    @patch("osiris_policy.orchestrator.EventBus")
    @patch("osiris_policy.orchestrator.log_run")  # Target where it's imported
    async def test_cli_log_run_error(
        self, mock_log_run, MockEventBus, mock_requests_post
    ):
        # --- Setup Mocks ---
        mock_event_bus_instance = MockEventBus.return_value
        mock_event_bus_instance.connect = AsyncMock()
        mock_event_bus_instance.publish = AsyncMock()
        mock_event_bus_instance.close = AsyncMock()

        mock_requests_post.side_effect = [
            _mock_response(status_code=200, json_data=mock_phi3_success_response_data),
            _mock_response(status_code=200, json_data=mock_pta_success_response_data),
        ]

        mock_log_run.side_effect = Exception("Simulated DB error during log_run")

        # --- Execute ---
        test_query = "query causing log_run error"
        captured_output = io.StringIO()
        # Expect an error log from main_async when log_run fails
        with (
            redirect_stdout(captured_output),
            self.assertLogs(
                logger="osiris_policy.orchestrator", level="ERROR"
            ) as log_watcher,
        ):
            await policy_orchestrator.main_async(test_query)

        cli_output_str = captured_output.getvalue()

        # --- Assertions ---
        # 1. log_run was called
        mock_log_run.assert_called_once()

        # 2. CLI output should still contain the main successful data
        # The error in log_run is a background failure, shouldn't stop primary output.
        self.assertTrue(
            mock_pta_success_response_data["transaction_id"] in cli_output_str
        )
        self.assertTrue(
            mock_pta_success_response_data["hermes_assessment"] in cli_output_str
        )
        # The error from log_run itself is not part of the CLI JSON output to stdout.
        self.assertFalse("Simulated DB error during log_run" in cli_output_str)

        # 3. Check logs for the specific log_run failure message
        self.assertTrue(
            any(
                "Failed to log run" in msg and "Simulated DB error" in msg
                for msg in log_watcher.output
            )
        )


if __name__ == "__main__":
    unittest.main()
