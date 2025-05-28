import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
from typing import Optional
from datetime import datetime
from peft import PeftModel
import asyncio
import redis.asyncio as aioredis
import json
import logging
import signal

# Global variables for models and tokenizers
hermes_model = None
hermes_tokenizer = None
phi3_model = None
phi3_tokenizer = None
phi3_adapter_date = None
phi3_base_model = None # New global for the original base model

# Globals for Redis client, pubsub, and the subscription task
g_redis_client = None
g_pubsub = None
adapter_subscription_task = None

# Model and Tokenizer Paths
# The MICRO_LLM_MODEL_PATH should point to the directory containing the ONNX model file (e.g., model.onnx)
# or directly to the .onnx file if ORTModelForCausalLM handles it.
# The fetch_phi3.sh script renames the downloaded ONNX model to phi3.onnx in /app/models/llm_micro/
MICRO_LLM_MODEL_PARENT_DIR = (
    "/app/models/llm_micro"  # Directory where phi3.onnx is located
)
MICRO_LLM_MODEL_PATH = os.getenv(
    "MICRO_LLM_MODEL_PATH", os.path.join(MICRO_LLM_MODEL_PARENT_DIR, "phi3.onnx")
)


PHI3_TOKENIZER_PATH = "microsoft/phi-3-mini-4k-instruct"
HERMES_MODEL_PATH = "/app/hermes-model"  # This is a directory


def load_hermes_model():
    """Loads the Hermes GPTQ model and tokenizer."""
    global hermes_model, hermes_tokenizer
    if hermes_model is not None and hermes_tokenizer is not None:
        print("Hermes model and tokenizer already loaded.")
        return

    print(f"Loading Hermes model from: {HERMES_MODEL_PATH}")
    try:
        hermes_tokenizer = AutoTokenizer.from_pretrained(
            HERMES_MODEL_PATH, use_fast=True
        )
        hermes_model = AutoModelForCausalLM.from_pretrained(
            HERMES_MODEL_PATH,
            device_map="auto",  # Automatically select device (CPU/GPU)
            trust_remote_code=True,  # Required for some custom model architectures
        )
        # No explicit .to(device) needed here due to device_map="auto"
        print("Hermes model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading Hermes model or tokenizer: {e}")
        hermes_model = None
        hermes_tokenizer = None


def load_phi3_model():
    """Loads the Phi-3 ONNX model and tokenizer."""
    global phi3_model, phi3_tokenizer, phi3_base_model, phi3_adapter_date # Added phi3_base_model and phi3_adapter_date
    if phi3_base_model is not None and phi3_tokenizer is not None: # Check base_model for already loaded
        print("Phi-3 base model and tokenizer already loaded.")
        return

    print(f"Loading Phi-3 tokenizer from: {PHI3_TOKENIZER_PATH}")
    print(f"Loading Phi-3 ONNX model from: {MICRO_LLM_MODEL_PATH}")

    # Determine device for ONNX model (provider will handle actual placement on CUDA device)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Phi-3 ONNX model will use provider: {'CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'}"
    )

    try:
        phi3_tokenizer = AutoTokenizer.from_pretrained(
            PHI3_TOKENIZER_PATH, use_fast=True
        )

        model_dir_to_load = MICRO_LLM_MODEL_PARENT_DIR
        if not os.path.exists(MICRO_LLM_MODEL_PATH):
            print(
                f"Error: ONNX model file not found at {MICRO_LLM_MODEL_PATH}. Ensure it has been downloaded and named correctly."
            )
        elif not os.path.isdir(model_dir_to_load):
            print(
                f"Warning: MICRO_LLM_MODEL_PARENT_DIR '{model_dir_to_load}' is not a directory. Attempting to load ONNX model directly from '{MICRO_LLM_MODEL_PATH}'. This might fail if config files are separate."
            )
            model_dir_to_load = MICRO_LLM_MODEL_PATH

        original_model = ORTModelForCausalLM.from_pretrained(
            model_dir_to_load,
            provider=(
                "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
            ),
            use_io_binding=True if device == "cuda" else False,
        )
        
        phi3_base_model = original_model # Store the original base model
        phi3_model = original_model      # Initially, phi3_model is the base model

        adapter_base_path = "/app/models/phi3/adapters/"
        latest_adapter_path = get_latest_adapter_dir(adapter_base_path)

        if latest_adapter_path:
            print(f"Loading PEFT adapter for Phi-3 from: {latest_adapter_path}")
            try:
                phi3_model = PeftModel.from_pretrained(phi3_base_model, latest_adapter_path) # Load adapter onto the base model
                adapter_dir_name = os.path.basename(latest_adapter_path)
                try:
                    date_obj = datetime.strptime(adapter_dir_name, "%Y%m%d")
                    phi3_adapter_date = date_obj.strftime("%Y-%m-%d")
                    print(f"Successfully loaded PEFT adapter {latest_adapter_path} with date {phi3_adapter_date}")
                except ValueError as ve:
                    print(f"Warning: Could not parse date from adapter directory name {adapter_dir_name}: {ve}. Adapter date will not be set.")
                    phi3_adapter_date = None
            except Exception as peft_e:
                print(f"Error loading PEFT adapter from {latest_adapter_path}: {peft_e}")
                print("Proceeding with the base Phi-3 model without adapter.")
                phi3_model = phi3_base_model # Fallback to base model
                phi3_adapter_date = None
        else:
            print("No PEFT adapter found for Phi-3, using base model.")
            phi3_model = phi3_base_model # Ensure phi3_model is set to base if no adapter
            phi3_adapter_date = None

        print("Phi-3 ONNX model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading Phi-3 ONNX model or tokenizer: {e}")
        phi3_base_model = None # Ensure base_model is also None on failure
        phi3_model = None
        phi3_tokenizer = None
        phi3_adapter_date = None


def get_hermes_model_and_tokenizer():
    """Returns the loaded Hermes model and tokenizer."""
    return hermes_model, hermes_tokenizer


def get_phi3_model_and_tokenizer():
    """Returns the loaded Phi-3 model and tokenizer."""
    return phi3_model, phi3_tokenizer


async def subscribe_to_adapter_updates():
    global phi3_model, phi3_adapter_date, phi3_base_model # Access globals
    
    if phi3_base_model is None:
        logging.error("Base Phi-3 model not loaded. Cannot start adapter subscription.")
        return

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    global g_redis_client, g_pubsub # Use global variables
    
    logging.info(f"Connecting to Redis at {redis_host}:{redis_port} for adapter updates.")
    
    # Removed local redis, pubsub = None, None

    while True: # Outer loop for reconnection
        try:
            # Check global client, not local
            if g_redis_client is None or not g_redis_client.is_connected(): # type: ignore
                logging.info(f"Attempting to connect to Redis at {redis_host}:{redis_port}...")
                # Assign to global client
                g_redis_client = await aioredis.from_url(f"redis://{redis_host}:{redis_port}")
                g_pubsub = g_redis_client.pubsub() # Assign to global pubsub
                await g_pubsub.subscribe("adapters.new")
                logging.info("Subscribed to 'adapters.new' Redis channel.")

            while True: # Inner loop for messages
                try:
                    # Use global pubsub
                    message = await g_pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0) # Timeout to allow periodic checks
                    if message and message["type"] == "message":
                        payload_str = message["data"].decode("utf-8")
                        logging.info(f"Received message on 'adapters.new': {payload_str}")
                        try:
                            payload = json.loads(payload_str)
                            new_date_str = payload.get("date") # Expected "YYYY-MM-DD"

                            if not new_date_str:
                                logging.warning("No 'date' field in message payload.")
                                continue

                            # Convert YYYY-MM-DD to YYYYMMDD for directory name
                            try:
                                parsed_date = datetime.strptime(new_date_str, "%Y-%m-%d")
                                adapter_dir_name = parsed_date.strftime("%Y%m%d")
                            except ValueError:
                                logging.error(f"Invalid date format '{new_date_str}'. Expected YYYY-MM-DD.")
                                continue
                                
                            adapter_base_path = "/app/models/phi3/adapters/" # Make this configurable if needed
                            new_adapter_path = os.path.join(adapter_base_path, adapter_dir_name)

                            # Check if the new adapter directory and config exist
                            adapter_config_file = os.path.join(new_adapter_path, "adapter_config.json")
                            if not os.path.isdir(new_adapter_path) or not os.path.isfile(adapter_config_file):
                                logging.error(f"Adapter directory {new_adapter_path} or adapter_config.json not found. Skipping reload.")
                                continue

                            logging.info(f"Attempting to hot-swap to adapter: {new_adapter_path}")
                            
                            if phi3_base_model is None: # Should have been caught earlier, but double check
                                logging.error("phi3_base_model is not available. Cannot hot-swap.")
                                continue

                            # Hot-swap: Load the new adapter onto the *base* model
                            new_adapted_model = PeftModel.from_pretrained(phi3_base_model, new_adapter_path)
                            phi3_model = new_adapted_model # Update global model
                            phi3_adapter_date = new_date_str # Update global date
                            logging.info(f"[HOT-SWAP] Phi-3 adapter reloaded -> {phi3_adapter_date}")

                        except json.JSONDecodeError:
                            logging.error(f"Failed to parse JSON from message: {payload_str}")
                        except Exception as e:
                            logging.error(f"Error processing adapter update message: {e}")
                    await asyncio.sleep(0.1) 
                except aioredis.exceptions.ConnectionError as e:
                    logging.error(f"Redis connection error in message loop: {e}. Breaking to reconnect.")
                    if g_pubsub:
                        try: # Adding try-except for cleanup safety
                            await g_pubsub.unsubscribe("adapters.new") 
                            await g_pubsub.close() # Close pubsub object itself
                        except Exception as ex_ps_close:
                            logging.error(f"Error closing pubsub object: {ex_ps_close}")
                        finally:
                             g_pubsub = None # Set global to None
                    if g_redis_client:
                        try: # Adding try-except for cleanup safety
                            await g_redis_client.close()
                            await g_redis_client.connection_pool.disconnect()
                        except Exception as ex_rd_close:
                            logging.error(f"Error closing redis client: {ex_rd_close}")
                        finally:
                            g_redis_client = None # Set global to None
                    raise # Re-raise to be caught by outer loop for reconnection
                except Exception as e: # This handles other errors in the message loop
                    logging.error(f"An unexpected error occurred in inner subscription loop: {e}")
                    # If it's a CancelledError, we should probably exit the loop
                    if isinstance(e, asyncio.CancelledError):
                        logging.info("Subscription task was cancelled. Exiting message loop.")
                        raise # Propagate cancellation
                    await asyncio.sleep(1) # Wait a bit before trying to process more messages

        except asyncio.CancelledError: # Handle cancellation at the outer loop level
            logging.info("Adapter subscription task cancelled. Shutting down subscription loop.")
            # Perform cleanup here as well, as the loop is exiting
            if g_pubsub:
                try:
                    await g_pubsub.unsubscribe("adapters.new")
                    await g_pubsub.close()
                except Exception as ex_ps_close:
                    logging.error(f"Error closing pubsub object on task cancellation: {ex_ps_close}")
                finally:
                    g_pubsub = None
            if g_redis_client:
                try:
                    await g_redis_client.close()
                    await g_redis_client.connection_pool.disconnect()
                except Exception as ex_rd_close:
                    logging.error(f"Error closing redis client on task cancellation: {ex_rd_close}")
                finally:
                    g_redis_client = None
            raise # Re-raise CancelledError to ensure the task is marked as cancelled

        except aioredis.exceptions.ConnectionError as e:
            logging.error(f"Redis connection failed: {e}. Retrying in 5 seconds...")
            if g_pubsub:
                try:
                    await g_pubsub.unsubscribe("adapters.new") # Should be g_pubsub
                    await g_pubsub.close()
                except Exception as ex_close:
                    logging.error(f"Error during pubsub cleanup: {ex_close}")
                finally:
                    g_pubsub = None # Set global to None
            if g_redis_client:
                try:
                    await g_redis_client.close() # Should be g_redis_client
                    await g_redis_client.connection_pool.disconnect()
                except Exception as ex_close:
                    logging.error(f"Error during redis client cleanup: {ex_close}")
                finally:
                    g_redis_client = None # Set global to None
            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"An unexpected error occurred in outer subscription loop: {e}. Retrying in 10 seconds...")
            await asyncio.sleep(10) # Wait before next cycle's attempt


async def shutdown_redis_resources():
    global g_redis_client, g_pubsub, adapter_subscription_task

    logging.info("Shutting down Redis resources...")
    if adapter_subscription_task and not adapter_subscription_task.done():
        logging.info("Cancelling adapter subscription task...")
        adapter_subscription_task.cancel()
        try:
            await adapter_subscription_task # Allow task to process cancellation
        except asyncio.CancelledError:
            logging.info("Adapter subscription task successfully cancelled.")
        except Exception as e:
            logging.error(f"Error during subscription task cancellation processing: {e}")

    # g_pubsub and g_redis_client are already cleaned up by subscribe_to_adapter_updates
    # when it receives CancelledError. However, we can add checks here for safety
    # or if they were set but the task never ran/exited cleanly without cancelling.

    if g_pubsub:
        logging.info("Ensuring PubSub is closed (should be handled by task cancellation).")
        try:
            # The subscription task should handle this, but as a safeguard:
            if hasattr(g_pubsub, 'is_connected') and g_pubsub.is_connected(): # Check if it has a connection attribute
                 await g_pubsub.unsubscribe("adapters.new")
                 await g_pubsub.close() # Close the pubsub object itself
            logging.info("PubSub closed.")
        except Exception as e:
            logging.error(f"Error during PubSub final cleanup: {e}")
        finally:
            g_pubsub = None
    
    if g_redis_client:
        logging.info("Ensuring Redis client is closed (should be handled by task cancellation).")
        try:
            # The subscription task should handle this, but as a safeguard:
            if g_redis_client.is_connected(): # type: ignore
                await g_redis_client.close()
                await g_redis_client.connection_pool.disconnect()
            logging.info("Redis client closed.")
        except Exception as e:
            logging.error(f"Error closing Redis client during final cleanup: {e}")
        finally:
            g_redis_client = None
    logging.info("Redis resources shutdown complete.")


if __name__ == "__main__":
    # For basic testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("Attempting to load Hermes model...")
    load_hermes_model()
    if hermes_model and hermes_tokenizer:
        print("Hermes loaded.")
    else:
        print("Hermes failed to load.")

    print("\nAttempting to load Phi-3 ONNX model...")
    load_phi3_model() # This will now also set phi3_base_model
    
    if phi3_model and phi3_tokenizer: # Changed from phi3_model to check both
        logging.info("Phi-3 loaded.") # Changed print to logging.info
        if phi3_adapter_date:
            logging.info(f"Phi-3 initial adapter date: {phi3_adapter_date}")

        logging.info("Starting Redis adapter subscription task...")
        loop = asyncio.get_event_loop_policy().get_event_loop() # Get event loop
        
        # Store the task in the global variable
        # This was previously done inside the try block, moving it out
        global adapter_subscription_task 
        adapter_subscription_task = loop.create_task(subscribe_to_adapter_updates())

        # Signal handler setup
        if os.name != 'nt': # Signal handlers are different on Windows
            for sig in (signal.SIGINT, signal.SIGTERM):
                # Updated lambda to ensure it uses the loop from the main thread
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown_redis_resources()))
            logging.info(f"Signal handlers for SIGINT and SIGTERM registered.")
        else:
            logging.info("Signal handlers for SIGINT/SIGTERM not registered on Windows. Use Ctrl+C.")
        
        try:
            logging.info("Application started. Running event loop forever (or until interrupt).")
            loop.run_forever() 
        except KeyboardInterrupt: # This will catch Ctrl+C on all platforms
            logging.info("KeyboardInterrupt received.")
        finally:
            logging.info("Starting application shutdown process...")
            # Check if loop is still running before trying to run_until_complete on it
            if loop.is_running():
                 logging.info("Event loop is running, initiating graceful shutdown of Redis resources.")
                 loop.run_until_complete(shutdown_redis_resources())
            else:
                 # If the loop isn't running (e.g. run_forever was never called or stopped abruptly)
                 # we might still need to clean up. This can be tricky.
                 # A new temporary loop might be needed if the original one is closed.
                 logging.info("Event loop is not running. Attempting shutdown with a new temporary event loop if necessary.")
                 try:
                     asyncio.run(shutdown_redis_resources()) # Try running with asyncio.run for cleanup
                 except RuntimeError as e:
                     logging.error(f"RuntimeError during final shutdown: {e}. Resources might not be fully cleaned.")


            # Ensure the main loop is stopped if it was run_forever
            if loop.is_running():
                loop.stop()
            
            # Close the loop itself
            # It's good practice to close the loop, but be careful if other parts of an app share it.
            # For standalone script, this is generally fine.
            # loop.close() # This might be too aggressive if called from within a running loop context.
            # Consider loop.call_soon_threadsafe(loop.close) or similar if issues arise.
            # For now, let's rely on Python's exit to clean up the loop or ensure it's closed after run_until_complete.

            logging.info("Application shutdown complete.")
    else:
        logging.error("Phi-3 model not loaded, Redis subscription task not started.")


def get_latest_adapter_dir(base_path: str) -> Optional[str]:
    """
    Scans a base path for date-formatted subdirectories (YYYYMMDD)
    containing an 'adapter_config.json' file and returns the path
    to the most recent valid adapter directory.

    Args:
        base_path: The base directory to scan (e.g., /models/phi3/adapters/).

    Returns:
        The full path to the latest valid adapter directory, or None if not found.
    """
    if not os.path.exists(base_path) or not os.path.isdir(base_path):
        print(f"Adapter base path {base_path} does not exist or is not a directory.")
        return None

    latest_dir = None
    latest_date = None

    try:
        for dirname in os.listdir(base_path):
            dirpath = os.path.join(base_path, dirname)
            if not os.path.isdir(dirpath):
                continue

            # Check if dirname matches YYYYMMDD format
            try:
                dir_date = datetime.strptime(dirname, "%Y%m%d").date()
            except ValueError:
                # Not a valid date-formatted directory name
                continue

            # Check for adapter_config.json
            adapter_config_path = os.path.join(dirpath, "adapter_config.json")
            if not os.path.exists(adapter_config_path) or not os.path.isfile(
                adapter_config_path
            ):
                # Missing adapter_config.json
                continue

            # If this is the first valid directory or newer than current latest
            if latest_date is None or dir_date > latest_date:
                latest_date = dir_date
                latest_dir = dirpath

    except OSError as e:
        print(f"Error accessing adapter base path {base_path}: {e}")
        return None

    if latest_dir:
        print(f"Found latest adapter directory: {latest_dir}")
    else:
        print(f"No valid adapter directories found in {base_path}")

    return latest_dir
