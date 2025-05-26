import unittest
import json
import datetime
import uuid
import os
import asyncio
from unittest.mock import patch, mock_open, MagicMock, call

from fastapi.testclient import TestClient

# Adjust the import path according to your project structure
# Assuming server.py is in the parent directory or PYTHONPATH is set up
import sys
# Add the parent directory to sys.path to allow imports from server.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import after sys.path modification
from server import app, FeedbackItem, PHI3_FEEDBACK_LOG_FILE, PHI3_FEEDBACK_DATA_FILE, load_recent_feedback, _generate_phi3_json

# Helper for async mocking if needed
async def mock_async_return_value(value):
    return value

class TestFeedbackMechanism(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        # To ensure a clean state for PHI3_FEEDBACK_DATA_FILE if it's created by tests
        if os.path.exists(PHI3_FEEDBACK_DATA_FILE):
            os.remove(PHI3_FEEDBACK_DATA_FILE)
        if os.path.exists(PHI3_FEEDBACK_LOG_FILE):
            os.remove(PHI3_FEEDBACK_LOG_FILE)


    # Test Case 1: Logging of Phi-3 Proposals and Hermes Assessments
    @patch('server._generate_hermes_text')
    @patch('server._generate_phi3_json') # This mock will be for the one called by propose_trade_adjustments
    @patch('builtins.open', new_callable=mock_open) # Mock the built-in open
    def test_log_propose_trade_adjustments(self, mock_file_open, mock_internal_phi3_gen, mock_hermes_gen):
        mock_phi3_output = {"ticker": "TESTPHI3", "action": "adjust", "confidence": 0.9, "rationale": "Phi3 rationale"}
        mock_hermes_output = "Hermes assessment text."

        async def _mock_phi3(*args, **kwargs):
            return mock_phi3_output
        
        async def _mock_hermes(*args, **kwargs):
            return mock_hermes_output

        mock_internal_phi3_gen.side_effect = _mock_phi3
        mock_hermes_gen.side_effect = _mock_hermes

        response = self.client.post("/propose_trade_adjustments/", json={"prompt": "Test prompt", "max_length": 50})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["phi3_proposal"], mock_phi3_output)
        self.assertEqual(data["hermes_assessment"], mock_hermes_output)

        mock_file_open.assert_called_once_with(PHI3_FEEDBACK_LOG_FILE, "a")
        
        # json.dump(obj, fp) calls fp.write(json.dumps(obj))
        # The first argument to the first call to write should be the JSON string.
        write_calls = mock_file_open.return_value.write.call_args_list
        self.assertTrue(len(write_calls) >= 2, "Expected at least two write calls (json.dump + newline)")
        
        # The first call to write() is by json.dump()
        logged_json_str = write_calls[0].args[0]
        logged_json = json.loads(logged_json_str)
        
        self.assertIn("transaction_id", logged_json)
        self.assertTrue(uuid.UUID(logged_json["transaction_id"])) # Check valid UUID
        self.assertIn("timestamp", logged_json)
        self.assertTrue(datetime.datetime.fromisoformat(logged_json["timestamp"]))
        self.assertEqual(logged_json["phi3_proposal"], mock_phi3_output)
        self.assertEqual(logged_json["hermes_assessment"], mock_hermes_output)
        
        # The second call should be for the newline
        self.assertEqual(write_calls[1].args[0], '\n')


    # Test Case 2: Feedback Submission Endpoint (`/feedback/phi3/`)
    @patch('builtins.open', new_callable=mock_open) # Mock the built-in open
    def test_submit_feedback_endpoint(self, mock_file_open):
        tx_id = str(uuid.uuid4())
        sample_feedback_payload = {
            "transaction_id": tx_id,
            "feedback_type": "correction",
            "feedback_content": {"old_rationale": "bad", "new_rationale": "good"},
            # timestamp will be set by server, so initial value here is just for completeness of payload
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), 
            "corrected_proposal": {"ticker": "TEST", "action": "adjust", "confidence": 1.0, "rationale": "Corrected"}
        }

        response = self.client.post("/feedback/phi3/", json=sample_feedback_payload)

        self.assertEqual(response.status_code, 200)
        resp_json = response.json()
        self.assertEqual(resp_json["transaction_id"], sample_feedback_payload["transaction_id"])
        self.assertEqual(resp_json["message"], "Feedback received successfully")

        mock_file_open.assert_called_once_with(PHI3_FEEDBACK_DATA_FILE, "a")
        
        write_calls = mock_file_open.return_value.write.call_args_list
        self.assertTrue(len(write_calls) >= 2)

        logged_json_str = write_calls[0].args[0]
        logged_json = json.loads(logged_json_str)

        self.assertEqual(logged_json["transaction_id"], sample_feedback_payload["transaction_id"])
        self.assertEqual(logged_json["feedback_type"], sample_feedback_payload["feedback_type"])
        self.assertEqual(logged_json["feedback_content"], sample_feedback_payload["feedback_content"])
        self.assertEqual(logged_json["corrected_proposal"], sample_feedback_payload["corrected_proposal"])
        
        # Server sets its own timestamp
        self.assertTrue(datetime.datetime.fromisoformat(logged_json["timestamp"]))
        # Ensure server timestamp is recent (e.g. within last few seconds of original client one for sanity)
        # This can be flaky, so a simple format check is often enough.
        # self.assertNotEqual(logged_json["timestamp"], sample_feedback_payload["timestamp"]) # This might be true if clocks are very synced
        self.assertTrue(abs(datetime.datetime.fromisoformat(logged_json["timestamp"]) - datetime.datetime.fromisoformat(sample_feedback_payload["timestamp"])) < datetime.timedelta(seconds=5))


    # Test Case 3: load_recent_feedback Function (Direct Test)
    @patch('builtins.print') # Mock print to check error logs
    def test_load_recent_feedback_logic(self, mock_print):
        # Scenario A: Valid Feedback (3 relevant items, ask for 2)
        mock_data_valid = [
            {"transaction_id": "1", "feedback_type": "rating", "feedback_content": "good", "timestamp": "t1", "corrected_proposal": None},
            {"transaction_id": "2", "feedback_type": "correction", "feedback_content": {}, "corrected_proposal": {"key": "val2"}, "timestamp": "t2"},
            {"transaction_id": "3", "feedback_type": "correction", "feedback_content": {}, "corrected_proposal": {"key": "val3"}, "timestamp": "t3"},
            {"transaction_id": "4", "feedback_type": "other", "feedback_content": {}, "corrected_proposal": {"key": "val4"}, "timestamp": "t4"},
            {"transaction_id": "5", "feedback_type": "correction", "feedback_content": {}, "corrected_proposal": None, "timestamp": "t5"}, # Invalid: corrected_proposal is None
            {"transaction_id": "6", "feedback_type": "correction", "feedback_content": {}, "corrected_proposal": {"key": "val6"}, "timestamp": "t6"},
        ]
        mock_jsonl_string_valid = "\n".join(json.dumps(item) for item in mock_data_valid)

        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=mock_jsonl_string_valid)):
                result = load_recent_feedback(max_examples=2)
                self.assertEqual(len(result), 2)
                # Should be the last two valid ones: item 3 and item 6
                self.assertEqual(result[0]["corrected_proposal"]["key"], "val3") 
                self.assertEqual(result[1]["corrected_proposal"]["key"], "val6")
        
        # Scenario B: Empty File or No "correction" Feedback
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="")): # Empty file
                result = load_recent_feedback()
                self.assertEqual(len(result), 0)

        mock_jsonl_string_no_correction = json.dumps({"transaction_id": "7", "feedback_type": "rating", "corrected_proposal": {"key":"val"}, "timestamp": "t7"})
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=mock_jsonl_string_no_correction)):
                result = load_recent_feedback()
                self.assertEqual(len(result), 0)

        # Scenario C: File Not Found
        with patch('os.path.exists', return_value=False):
                 result = load_recent_feedback()
                 self.assertEqual(len(result), 0)
                 mock_print.assert_any_call(f"Feedback file {PHI3_FEEDBACK_DATA_FILE} not found. No feedback to load.")
        mock_print.reset_mock()

        # Scenario D: Corrupted JSON
        mock_jsonl_corrupted = "this is not json\n" + json.dumps(mock_data_valid[1]) # Second line is valid
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=mock_jsonl_corrupted)):
                result = load_recent_feedback(max_examples=1)
                self.assertEqual(len(result), 1) # Should load the valid line
                self.assertEqual(result[0]["corrected_proposal"]["key"], "val2")
                mock_print.assert_any_call(unittest.mock.ANY) # Check if print was called for the error
                self.assertTrue(any("Error decoding JSON" in call_args[0][0] for call_args in mock_print.call_args_list))
        mock_print.reset_mock()

        # Scenario E: IO Error during read
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open()) as m_open:
                m_open.side_effect = IOError("Disk full")
                result = load_recent_feedback()
                self.assertEqual(len(result), 0)
                mock_print.assert_any_call(f"Error reading feedback file {PHI3_FEEDBACK_DATA_FILE}: Disk full")


    # Test Case 4: Prompt Augmentation (testing _generate_phi3_json directly for focused test)
    # This also indirectly tests the augmentation via the /propose_trade_adjustments/ endpoint.
    # To directly test _generate_phi3_json, we need to mock its internal dependencies like outlines
    @patch('outlines.generate.json')
    @patch('server.load_recent_feedback') 
    async def test_prompt_augmentation_logic(self, mock_load_feedback, mock_outlines_gen_json_factory):
        # Mock for outlines.generate.json factory should return a callable (the generator instance)
        mock_generator_instance = MagicMock() # This will be the callable generator
        async def async_mock_generator_instance(*args, **kwargs): # The generator call is async
            return {"ticker": "OUTLINES_TICK", "action": "generated", "confidence": 0.7, "rationale": "Generated by outlines"}
        mock_generator_instance.side_effect = async_mock_generator_instance
        mock_outlines_gen_json_factory.return_value = mock_generator_instance # outlines.generate.json() returns this

        # Mock model and tokenizer (not strictly needed if outlines.generate.json is fully mocked)
        mock_phi3_model = MagicMock()
        mock_phi3_tokenizer = MagicMock()

        original_prompt = "This is the original user prompt."

        # --- Scenario A: With feedback ---
        mock_feedback_items = [
            {"transaction_id": "fb1", "feedback_type": "correction", 
             "corrected_proposal": {"ticker": "FB_TICK1", "action": "adjust", "confidence": 0.8, "rationale": "Feedback 1"}, 
             "timestamp": "ts1"},
            {"transaction_id": "fb2", "feedback_type": "correction", 
             "corrected_proposal": {"ticker": "FB_TICK2", "action": "pass", "confidence": 0.99, "rationale": "Feedback 2"}, 
             "timestamp": "ts2"}
        ]
        mock_load_feedback.return_value = mock_feedback_items
        
        # Call the function we want to test
        await _generate_phi3_json(original_prompt, 100, mock_phi3_model, mock_phi3_tokenizer)

        # Assert that load_recent_feedback was called
        mock_load_feedback.assert_called_once_with(max_examples=3)

        # Check that outlines.generate.json factory was called (with model, schema, tokenizer)
        mock_outlines_gen_json_factory.assert_called_once_with(mock_phi3_model, unittest.mock.ANY, tokenizer=mock_phi3_tokenizer)
        
        # Check that the generator instance was called with the augmented prompt
        mock_generator_instance.assert_called_once()
        call_args_to_generator = mock_generator_instance.call_args[0] # (prompt_str, max_tokens=...)
        effective_prompt = call_args_to_generator[0]
        
        expected_augmentation_intro = "Based on past feedback, here are examples of desired JSON outputs:"
        expected_augmentation_outro = "\n\nNow, considering the above, please process the following request:\n"
        
        self.assertTrue(effective_prompt.startswith(expected_augmentation_intro))
        self.assertIn(json.dumps(mock_feedback_items[0]["corrected_proposal"], indent=2), effective_prompt)
        self.assertIn(json.dumps(mock_feedback_items[1]["corrected_proposal"], indent=2), effective_prompt)
        self.assertTrue(effective_prompt.endswith(f"{expected_augmentation_outro}{original_prompt}"))

        # --- Scenario B: No feedback ---
        mock_load_feedback.return_value = [] # No feedback this time
        mock_load_feedback.reset_mock()
        mock_outlines_gen_json_factory.reset_mock()
        mock_generator_instance.reset_mock()
        
        await _generate_phi3_json(original_prompt, 100, mock_phi3_model, mock_phi3_tokenizer)
        
        mock_load_feedback.assert_called_once_with(max_examples=3)
        mock_outlines_gen_json_factory.assert_called_once() # Still called
        mock_generator_instance.assert_called_once()       # Still called

        call_args_to_generator_no_fb = mock_generator_instance.call_args[0]
        effective_prompt_no_fb = call_args_to_generator_no_fb[0]
        
        self.assertEqual(effective_prompt_no_fb, original_prompt) # Should be original prompt only

    def tearDown(self):
        # Clean up any files created during tests, if any persist beyond mocks
        if os.path.exists(PHI3_FEEDBACK_DATA_FILE):
            os.remove(PHI3_FEEDBACK_DATA_FILE)
        if os.path.exists(PHI3_FEEDBACK_LOG_FILE):
            os.remove(PHI3_FEEDBACK_LOG_FILE)

if __name__ == '__main__':
    # Need to run this within an async context if we have top-level async tests
    # or use a test runner that supports asyncio like `pytest --asyncio-mode=auto`
    # For unittest, if you have async tests not handled by TestClient, structure can be more complex.
    # Here, TestClient handles the async nature of endpoints.
    # For _generate_phi3_json, which is async, we patch it and test its internals,
    # or call it with asyncio.run() if testing it directly outside TestClient.
    # The test_prompt_augmentation_logic is an async def, so unittest TestLoader needs to handle it.
    # Standard unittest doesn't run async test methods directly.
    # A simple way for this specific test case is to wrap its call in asyncio.run()
    
    # Re-structure test_prompt_augmentation_logic to be synchronous but call the async function using asyncio.run
    # Or use a library like `async_case` for unittest with asyncio.
    # For simplicity in this environment, I'll assume standard unittest execution
    # and that the test runner or an adapter would handle `async def test_...` methods.
    # If not, the test_prompt_augmentation_logic would need to be called via asyncio.run().

    # Let's make test_prompt_augmentation_logic callable by standard unittest
    # by wrapping its core logic in asyncio.run()

    # Save the original test method
    original_test_prompt_augmentation_logic = TestFeedbackMechanism.test_prompt_augmentation_logic

    # Create a new sync method that calls the async one
    def sync_test_prompt_augmentation_logic(self, mock_load_feedback, mock_outlines_gen_json_factory):
        asyncio.run(original_test_prompt_augmentation_logic(self, mock_load_feedback, mock_outlines_gen_json_factory))

    TestFeedbackMechanism.test_prompt_augmentation_logic = sync_test_prompt_augmentation_logic
    
    unittest.main()
