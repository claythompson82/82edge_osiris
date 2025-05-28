import asyncio
import json
import logging
from unittest import IsolatedAsyncioTestCase, mock
from datetime import datetime

# Assuming llm_sidecar.loader is importable
from llm_sidecar import loader

# Mock the PeftModel class before it's used by the loader module
# This is important if the loader module itself tries to use PeftModel at import time (though unlikely for from_pretrained)
loader.PeftModel = mock.MagicMock()


class TestAdapterReloader(IsolatedAsyncioTestCase):

    def setUp(self):
        # Reset globals that might be modified by tests or loader module itself
        loader.phi3_adapter_date = None
        loader.phi3_model = None # This will be the adapted model or the base model
        loader.phi3_base_model = mock.Mock(name="BaseModel") # Mock base model for most tests
        loader.g_redis_client = None
        loader.g_pubsub = None
        
        # Cancel any existing task from a previous test run if it wasn't properly cleaned up
        if hasattr(loader, 'adapter_subscription_task') and loader.adapter_subscription_task and not loader.adapter_subscription_task.done():
            loader.adapter_subscription_task.cancel()
            # We need to give a chance for the cancellation to be processed in an async context
            # This is tricky in setUp, usually handled in tearDown or by ensuring tests clean up.
            # For now, we'll rely on tearDown for proper async cleanup.
        loader.adapter_subscription_task = None
        
        # Capture logs
        self.logger = logging.getLogger('llm_sidecar.loader') # Target specific logger
        self.log_capture_handler = mock.MagicMock(spec=logging.Handler)
        self.log_capture_handler.level = logging.INFO
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.log_capture_handler.setFormatter(self.formatter)
        
        # Store original handlers and level
        self.original_handlers = self.logger.handlers[:]
        self.original_level = self.logger.level
        
        self.logger.handlers = [self.log_capture_handler] # Replace handlers
        self.logger.setLevel(logging.INFO)


    async def tearDown(self):
        # Ensure any running task is cancelled
        if loader.adapter_subscription_task and not loader.adapter_subscription_task.done():
            loader.adapter_subscription_task.cancel()
            try:
                await loader.adapter_subscription_task
            except asyncio.CancelledError:
                pass # Expected
            except Exception as e:
                # Log unexpected errors during task cancellation
                print(f"Error during test teardown task cancellation: {e}")


        # Explicitly call shutdown_redis_resources to clean up mock Redis resources
        # This is important because the test might have set g_redis_client or g_pubsub
        await loader.shutdown_redis_resources()

        # Restore original logging configuration
        self.logger.handlers = self.original_handlers
        self.logger.setLevel(self.original_level)
        
        # Reset PeftModel mock for other tests if any
        loader.PeftModel.reset_mock()


    @mock.patch('llm_sidecar.loader.PeftModel.from_pretrained', new_callable=mock.MagicMock) # Use new_callable for PeftModel
    @mock.patch('llm_sidecar.loader.os.path.isfile')
    @mock.patch('llm_sidecar.loader.os.path.isdir')
    @mock.patch('llm_sidecar.loader.aioredis') # Mock the aioredis module used in loader
    async def test_successful_reload(self, mock_aioredis_module, mock_isdir, mock_isfile, mock_peft_load):
        # Setup mock for aioredis.from_url()
        mock_redis_instance = mock.AsyncMock()
        mock_pubsub_instance = mock.AsyncMock()
        mock_aioredis_module.from_url.return_value = mock_redis_instance
        mock_redis_instance.pubsub.return_value = mock_pubsub_instance
        
        # Simulate one message, then cause subsequent calls to wait (simulating no new messages)
        message_payload = {'type': 'message', 'channel': b'adapters.new', 'data': b'{"date": "2023-10-26"}'}
        
        # Use a list to control side_effect outputs
        get_message_effects = [message_payload, asyncio.CancelledError()] # Simulate message then cancellation
        
        async def mock_get_message_side_effect(*args, **kwargs):
            if mock_get_message_side_effect.effects:
                effect = mock_get_message_side_effect.effects.pop(0)
                if isinstance(effect, Exception):
                    raise effect
                return effect
            return None # Should not be reached if task is cancelled
        mock_get_message_side_effect.effects = get_message_effects

        mock_pubsub_instance.get_message.side_effect = mock_get_message_side_effect

        mock_isdir.return_value = True
        mock_isfile.return_value = True
        
        # Ensure PeftModel.from_pretrained is an AsyncMock if it needs to be awaited (it's not usually)
        # or a standard MagicMock if it's a synchronous call. Given it's model loading, it's likely sync.
        mock_peft_load.return_value = mock.Mock(name="AdaptedModel")

        # Ensure phi3_base_model is set for this test, setUp does this, but good to be explicit if needed
        self.assertIsNotNone(loader.phi3_base_model, "Base model should be mocked in setUp")

        # Start the subscription task
        # loader.adapter_subscription_task is global in loader, assign it here
        loader.adapter_subscription_task = asyncio.create_task(loader.subscribe_to_adapter_updates())
        
        try:
            await asyncio.wait_for(loader.adapter_subscription_task, timeout=1.0)
        except asyncio.TimeoutError:
            # This is okay if the task is designed to run forever and got cancelled by get_message side effect
            pass
        except asyncio.CancelledError:
            pass # Expected due to side effect

        mock_peft_load.assert_called_once_with(loader.phi3_base_model, '/app/models/phi3/adapters/20231026')
        self.assertEqual(loader.phi3_adapter_date, "2023-10-26")
        
        # Check log messages
        found_log = False
        for call in self.log_capture_handler.handle.call_args_list:
            record = call.args[0] # The LogRecord is the first argument to handle()
            if isinstance(record, logging.LogRecord) and "[HOT-SWAP] Phi-3 adapter reloaded -> 2023-10-26" in record.getMessage():
                found_log = True
                break
        self.assertTrue(found_log, "Hot-swap log message not found.")

        # Additional cleanup check: Ensure pubsub and redis client were closed (set to None by subscribe_to_adapter_updates)
        self.assertIsNone(loader.g_pubsub, "g_pubsub should be None after task cancellation and cleanup")
        self.assertIsNone(loader.g_redis_client, "g_redis_client should be None after task cancellation and cleanup")

    @mock.patch('llm_sidecar.loader.aioredis') # Mock aioredis to check it's not called
    async def test_base_model_not_loaded(self, mock_aioredis_module):
        loader.phi3_base_model = None # Explicitly set base model to None

        # Call the function directly, not as a task, as it should exit quickly
        await loader.subscribe_to_adapter_updates()

        mock_aioredis_module.from_url.assert_not_called()
        
        found_log = False
        for call in self.log_capture_handler.handle.call_args_list:
            record = call.args[0]
            if isinstance(record, logging.LogRecord) and "Base Phi-3 model not loaded. Cannot start adapter subscription." in record.getMessage():
                found_log = True
                break
        self.assertTrue(found_log, "Log message for base model not loaded not found.")

    @mock.patch('llm_sidecar.loader.PeftModel.from_pretrained', new_callable=mock.MagicMock)
    @mock.patch('llm_sidecar.loader.aioredis')
    async def test_invalid_json_message(self, mock_aioredis_module, mock_peft_load):
        mock_redis_instance = mock.AsyncMock()
        mock_pubsub_instance = mock.AsyncMock()
        mock_aioredis_module.from_url.return_value = mock_redis_instance
        mock_redis_instance.pubsub.return_value = mock_pubsub_instance

        message_payload = {'type': 'message', 'channel': b'adapters.new', 'data': b'{"invalid_json'}
        get_message_effects = [message_payload, asyncio.CancelledError()]
        
        async def mock_get_message_side_effect(*args, **kwargs):
            if mock_get_message_side_effect.effects:
                effect = mock_get_message_side_effect.effects.pop(0)
                if isinstance(effect, Exception):
                    raise effect
                return effect
            return None
        mock_get_message_side_effect.effects = get_message_effects
        mock_pubsub_instance.get_message.side_effect = mock_get_message_side_effect

        loader.adapter_subscription_task = asyncio.create_task(loader.subscribe_to_adapter_updates())
        try:
            await asyncio.wait_for(loader.adapter_subscription_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass 
        except asyncio.CancelledError:
            pass

        mock_peft_load.assert_not_called()
        
        found_log = False
        for call in self.log_capture_handler.handle.call_args_list:
            record = call.args[0]
            if isinstance(record, logging.LogRecord) and "Failed to parse JSON from message" in record.getMessage():
                found_log = True
                break
        self.assertTrue(found_log, "Log message for invalid JSON not found.")

    @mock.patch('llm_sidecar.loader.PeftModel.from_pretrained', new_callable=mock.MagicMock)
    @mock.patch('llm_sidecar.loader.aioredis')
    async def test_message_missing_date_field(self, mock_aioredis_module, mock_peft_load):
        mock_redis_instance = mock.AsyncMock()
        mock_pubsub_instance = mock.AsyncMock()
        mock_aioredis_module.from_url.return_value = mock_redis_instance
        mock_redis_instance.pubsub.return_value = mock_pubsub_instance

        message_payload = {'type': 'message', 'channel': b'adapters.new', 'data': b'{"other_field": "value"}'}
        get_message_effects = [message_payload, asyncio.CancelledError()]
        
        async def mock_get_message_side_effect(*args, **kwargs):
            if mock_get_message_side_effect.effects:
                effect = mock_get_message_side_effect.effects.pop(0)
                if isinstance(effect, Exception):
                    raise effect
                return effect
            return None
        mock_get_message_side_effect.effects = get_message_effects
        mock_pubsub_instance.get_message.side_effect = mock_get_message_side_effect

        loader.adapter_subscription_task = asyncio.create_task(loader.subscribe_to_adapter_updates())
        try:
            await asyncio.wait_for(loader.adapter_subscription_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            pass

        mock_peft_load.assert_not_called()
        
        found_log = False
        for call in self.log_capture_handler.handle.call_args_list:
            record = call.args[0]
            if isinstance(record, logging.LogRecord) and "No 'date' field in message payload." in record.getMessage():
                found_log = True
                break
        self.assertTrue(found_log, "Log message for missing 'date' field not found.")

    @mock.patch('llm_sidecar.loader.PeftModel.from_pretrained', new_callable=mock.MagicMock)
    @mock.patch('llm_sidecar.loader.aioredis')
    async def test_invalid_date_format_in_message(self, mock_aioredis_module, mock_peft_load):
        mock_redis_instance = mock.AsyncMock()
        mock_pubsub_instance = mock.AsyncMock()
        mock_aioredis_module.from_url.return_value = mock_redis_instance
        mock_redis_instance.pubsub.return_value = mock_pubsub_instance

        message_payload = {'type': 'message', 'channel': b'adapters.new', 'data': b'{"date": "2023/10/26"}'} # Invalid format
        get_message_effects = [message_payload, asyncio.CancelledError()]
        
        async def mock_get_message_side_effect(*args, **kwargs):
            if mock_get_message_side_effect.effects:
                effect = mock_get_message_side_effect.effects.pop(0)
                if isinstance(effect, Exception):
                    raise effect
                return effect
            return None
        mock_get_message_side_effect.effects = get_message_effects
        mock_pubsub_instance.get_message.side_effect = mock_get_message_side_effect

        loader.adapter_subscription_task = asyncio.create_task(loader.subscribe_to_adapter_updates())
        try:
            await asyncio.wait_for(loader.adapter_subscription_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            pass

        mock_peft_load.assert_not_called()
        
        found_log = False
        expected_log_message = "Invalid date format '2023/10/26'. Expected YYYY-MM-DD."
        for call in self.log_capture_handler.handle.call_args_list:
            record = call.args[0]
            if isinstance(record, logging.LogRecord) and expected_log_message in record.getMessage():
                found_log = True
                break
        self.assertTrue(found_log, f"Log message for invalid date format not found. Expected: '{expected_log_message}'")

    @mock.patch('llm_sidecar.loader.PeftModel.from_pretrained', new_callable=mock.MagicMock)
    @mock.patch('llm_sidecar.loader.os.path.isfile') # Mock isfile as it's checked after isdir
    @mock.patch('llm_sidecar.loader.os.path.isdir')
    @mock.patch('llm_sidecar.loader.aioredis')
    async def test_adapter_directory_not_found(self, mock_aioredis_module, mock_isdir, mock_isfile, mock_peft_load):
        mock_redis_instance = mock.AsyncMock()
        mock_pubsub_instance = mock.AsyncMock()
        mock_aioredis_module.from_url.return_value = mock_redis_instance
        mock_redis_instance.pubsub.return_value = mock_pubsub_instance

        message_payload = {'type': 'message', 'channel': b'adapters.new', 'data': b'{"date": "2023-10-27"}'}
        get_message_effects = [message_payload, asyncio.CancelledError()]
        
        async def mock_get_message_side_effect(*args, **kwargs):
            if mock_get_message_side_effect.effects:
                effect = mock_get_message_side_effect.effects.pop(0)
                if isinstance(effect, Exception):
                    raise effect
                return effect
            return None
        mock_get_message_side_effect.effects = get_message_effects
        mock_pubsub_instance.get_message.side_effect = mock_get_message_side_effect

        mock_isdir.return_value = False # Simulate directory not found
        mock_isfile.return_value = True # This won't be reached if isdir is false, but set for completeness

        loader.adapter_subscription_task = asyncio.create_task(loader.subscribe_to_adapter_updates())
        try:
            await asyncio.wait_for(loader.adapter_subscription_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            pass

        mock_peft_load.assert_not_called()
        
        found_log = False
        expected_log_message = "Adapter directory /app/models/phi3/adapters/20231027 or adapter_config.json not found. Skipping reload."
        for call in self.log_capture_handler.handle.call_args_list:
            record = call.args[0]
            if isinstance(record, logging.LogRecord) and expected_log_message in record.getMessage():
                found_log = True
                break
        self.assertTrue(found_log, f"Log message for adapter directory not found not found. Expected: '{expected_log_message}'")

    @mock.patch('llm_sidecar.loader.PeftModel.from_pretrained', new_callable=mock.MagicMock)
    @mock.patch('llm_sidecar.loader.os.path.isfile')
    @mock.patch('llm_sidecar.loader.os.path.isdir')
    @mock.patch('llm_sidecar.loader.aioredis')
    async def test_adapter_config_file_not_found(self, mock_aioredis_module, mock_isdir, mock_isfile, mock_peft_load):
        mock_redis_instance = mock.AsyncMock()
        mock_pubsub_instance = mock.AsyncMock()
        mock_aioredis_module.from_url.return_value = mock_redis_instance
        mock_redis_instance.pubsub.return_value = mock_pubsub_instance

        message_payload = {'type': 'message', 'channel': b'adapters.new', 'data': b'{"date": "2023-10-28"}'}
        get_message_effects = [message_payload, asyncio.CancelledError()]
        
        async def mock_get_message_side_effect(*args, **kwargs):
            if mock_get_message_side_effect.effects:
                effect = mock_get_message_side_effect.effects.pop(0)
                if isinstance(effect, Exception):
                    raise effect
                return effect
            return None
        mock_get_message_side_effect.effects = get_message_effects
        mock_pubsub_instance.get_message.side_effect = mock_get_message_side_effect

        mock_isdir.return_value = True # Directory exists
        mock_isfile.return_value = False # Config file does not exist

        loader.adapter_subscription_task = asyncio.create_task(loader.subscribe_to_adapter_updates())
        try:
            await asyncio.wait_for(loader.adapter_subscription_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            pass

        mock_peft_load.assert_not_called()
        
        found_log = False
        expected_log_message = "Adapter directory /app/models/phi3/adapters/20231028 or adapter_config.json not found. Skipping reload."
        for call in self.log_capture_handler.handle.call_args_list:
            record = call.args[0]
            if isinstance(record, logging.LogRecord) and expected_log_message in record.getMessage():
                found_log = True
                break
        self.assertTrue(found_log, f"Log message for adapter config file not found not found. Expected: '{expected_log_message}'")
