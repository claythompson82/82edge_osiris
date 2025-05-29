import argparse
import json
import requests
import logging
import asyncio # Added for new event-driven model
from typing import TypedDict, Dict, Any, Optional, List

# LangGraph
from langgraph.graph import StateGraph, END

# EventBus
from llm_sidecar.event_bus import EventBus, RedisError

# Database for logging runs
from llm_sidecar.db import log_run, OrchestratorRunLog


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables & Constants ---
TICK_BUFFER: List[Dict[str, Any]] = [] 
# TICKS_PER_PROPOSAL will be set by CLI args
MARKET_TICKS_CHANNEL = "market.ticks" # Default, can be overridden by CLI
# REDIS_URL will be set by CLI args

# --- State Definition ---
class WorkflowState(TypedDict):
    query: str # This will now be the stringified tick buffer or a summary
    market_query_result: Optional[str] # Result of processing the query/ticks
    phi3_raw_proposal_request: Optional[Dict[str, Any]]
    phi3_response: Optional[Dict[str, Any]]
    propose_trade_adjustments_response: Optional[Dict[str, Any]]
    final_output: Optional[str]
    error: Optional[str]
    run_id: Optional[str] # To track individual workflow runs triggered by ticks


# --- Node Implementations ---

def query_market_node(state: WorkflowState) -> WorkflowState:
    logger.info(f"Executing QueryMarket node for run_id: {state.get('run_id', 'N/A')}...")
    # The 'query' field in the state now contains the stringified list of buffered ticks.
    # This node will transform it into a "market_query_result".
    # For now, let's just pass it through or add a simple message.
    raw_tick_data_str = state["query"]
    market_query_result = f"Processed market data based on recent ticks: {raw_tick_data_str}"
    logger.info(f"Market query result for run_id {state.get('run_id', 'N/A')}: {market_query_result}")
    return {**state, "market_query_result": market_query_result}


async def generate_proposal_node(state: WorkflowState) -> WorkflowState:
    logger.info(f"Executing GenerateProposal node for run_id: {state.get('run_id', 'N/A')}...")
    if state.get("error"):
        return state

    market_data = state.get("market_query_result", "No market data available.")
    if not market_data: # Should not happen if query_market_node ran
        logger.warning(f"No market data in state for run_id: {state.get('run_id', 'N/A')}. Using default.")
        market_data = "No market data available (warning)."
    
    # Construct prompt for Phi-3
    # Example: "Based on the market context 'Favorable conditions.', generate a trade proposal."
    # This prompt structure is an example; actual prompt engineering would be more complex.
    prompt_text = (
        f"Market Context: {market_data}\n\n"
        "Please generate a detailed JSON trade proposal based on this context. "
        "The proposal should include 'ticker', 'action' (adjust, pass, abort), "
        "'side' (LONG, SHORT), 'new_stop_pct', 'new_target_pct', 'confidence' (0-1), "
        "and 'rationale' (one-sentence justification)."
    )
    
    phi3_payload = {
        "prompt": prompt_text,
        "max_length": 512 # Adjust as needed
    }
    logger.info(f"Sending to /generate?model_id=phi3 for run_id {state.get('run_id', 'N/A')}: {json.dumps(phi3_payload)}")

    try:
        response = requests.post(
            "http://localhost:8000/generate?model_id=phi3", # TODO: Make sidecar URL configurable
            json=phi3_payload,
            timeout=60
        )
        response.raise_for_status()
        phi3_response_json = response.json()
        logger.info(f"Received from /generate?model_id=phi3 for run_id {state.get('run_id', 'N/A')}: {json.dumps(phi3_response_json)}")
        
        if isinstance(phi3_response_json, dict) and "error" not in phi3_response_json:
            return {**state, "phi3_raw_proposal_request": phi3_payload, "phi3_response": phi3_response_json}
        else:
            error_message = f"Phi-3 generation failed for run_id {state.get('run_id', 'N/A')}. Response: {json.dumps(phi3_response_json)}"
            logger.error(error_message)
            return {**state, "error": error_message, "phi3_response": phi3_response_json}

    except requests.exceptions.RequestException as e:
        error_message = f"HTTP request to Phi-3 failed for run_id {state.get('run_id', 'N/A')}: {e}"
        logger.error(error_message)
        return {**state, "error": error_message}
    except json.JSONDecodeError as e:
        error_message = f"Failed to decode JSON response from Phi-3 for run_id {state.get('run_id', 'N/A')}: {e}. Response text: {response.text if response else 'No response'}"
        logger.error(error_message)
        return {**state, "error": error_message}


async def evaluate_proposal_node(state: WorkflowState) -> WorkflowState:
    logger.info(f"Executing EvaluateProposal node for run_id: {state.get('run_id', 'N/A')}...")
    if state.get("error"):
        return state

    phi3_proposal = state.get("phi3_response")
    if not phi3_proposal or not isinstance(phi3_proposal, dict):
        error_message = f"Phi-3 proposal is missing or invalid in state for evaluation for run_id: {state.get('run_id', 'N/A')}."
        logger.error(error_message)
        return {**state, "error": error_message}

    # Construct prompt for /propose_trade_adjustments
    # This endpoint internally calls Phi-3 again and then Hermes for assessment.
    # The prompt for this endpoint should be the "user query" that leads to the proposal.
    # Let's use the original query or market data, as this endpoint regenerates phi3 proposal.
    # The subtask says: "Constructs a prompt using the Phi-3 response for the /propose_trade_adjustments endpoint."
    # This is a bit ambiguous. The /propose_trade_adjustments endpoint takes a general "prompt" (user query)
    # and then internally generates a phi3 proposal AND a hermes assessment.
    # If we pass the phi3_proposal *as the prompt*, it might try to *critique the critique*.
    # Let's assume the intent is to get an assessment on the *original need* or *market data*.
    # However, the endpoint's current implementation in server.py has hermes assess the *phi3_json* it generated.
    # So, the "prompt" to /propose_trade_adjustments should be the one that *generates* the proposal.
    # Let's use the original user query or a derivative.
    # The `phi3_raw_proposal_request` contains the prompt we sent to phi3. Let's use that.
    
    # The 'query' in the state is now the stringified tick data.
    # The /propose_trade_adjustments endpoint expects a "prompt" that is a user query or market context.
    # We should use the `market_query_result` from the previous node, or the original `query` (tick data string).
    # Let's use `state["query"]` which is the stringified ticks, as per current /propose_trade_adjustments logic
    # which internally generates a phi3 proposal based on this prompt.
    prompt_for_pta = state["query"] 

    pta_payload = {
        "prompt": prompt_for_pta, 
        "max_length": 512 
    }
    logger.info(f"Sending to /propose_trade_adjustments for run_id {state.get('run_id', 'N/A')}: {json.dumps(pta_payload)}")

    try:
        response = requests.post(
            "http://localhost:8000/propose_trade_adjustments", # TODO: Make sidecar URL configurable
            json=pta_payload,
            timeout=120 
        )
        response.raise_for_status()
        pta_response_json = response.json()
        logger.info(f"Received from /propose_trade_adjustments for run_id {state.get('run_id', 'N/A')}: {json.dumps(pta_response_json)}")
        
        if isinstance(pta_response_json, dict) and "error" not in pta_response_json:
            return {**state, "propose_trade_adjustments_response": pta_response_json}
        else:
            error_message = f"Call to /propose_trade_adjustments failed for run_id {state.get('run_id', 'N/A')}. Response: {json.dumps(pta_response_json)}"
            logger.error(error_message)
            return {**state, "error": error_message, "propose_trade_adjustments_response": pta_response_json}

    except requests.exceptions.RequestException as e:
        error_message = f"HTTP request to /propose_trade_adjustments failed for run_id {state.get('run_id', 'N/A')}: {e}"
        logger.error(error_message)
        return {**state, "error": error_message}
    except json.JSONDecodeError as e:
        error_message = f"Failed to decode JSON response from /propose_trade_adjustments for run_id {state.get('run_id', 'N/A')}: {e}. Response text: {response.text if response else 'No response'}"
        logger.error(error_message)
        return {**state, "error": error_message}


async def publish_events_node(state: WorkflowState) -> WorkflowState:
    run_id = state.get('run_id', 'N/A')
    logger.info(f"Executing PublishEvents node for run_id: {run_id}...")
    if state.get("error"):
        logger.warning(f"Skipping event publication for run_id {run_id} due to previous error in workflow.")
        final_output = state.get("final_output")
        if not final_output: # Ensure error is captured if no other output was set
            final_output = json.dumps({"status": "error", "run_id": run_id, "details": state["error"]})
        return {**state, "final_output": final_output }

    pta_response = state.get("propose_trade_adjustments_response")
    if not pta_response or "phi3_proposal" not in pta_response or "hermes_assessment" not in pta_response:
        error_message = f"Response from /propose_trade_adjustments is missing or malformed in state for run_id {run_id}. Cannot publish events."
        logger.error(error_message)
        output_data = pta_response if pta_response else state.get("phi3_response", {"error": "No valid proposal or assessment found"})
        output_data["run_id"] = run_id # Add run_id to the output
        return {**state, "error": error_message, "final_output": json.dumps(output_data)}

    phi3_proposal_for_event = pta_response["phi3_proposal"]
    hermes_assessment_for_event = pta_response["hermes_assessment"]
    transaction_id = pta_response.get("transaction_id", f"unknown_tx_for_run_{run_id}") # Ensure tx_id is somewhat unique if missing

    # TODO: Use a shared EventBus instance if possible, or pass redis_url from main args
    event_bus = EventBus(redis_url="redis://localhost:6379/0") 
    try:
        await event_bus.connect()

        if isinstance(phi3_proposal_for_event, dict):
            # Add run_id to the event payload for better traceability
            event_phi3_proposal = {**phi3_proposal_for_event, "orchestrator_run_id": run_id}
            await event_bus.publish("phi3.proposal.created", json.dumps(event_phi3_proposal))
            logger.info(f"Published 'phi3.proposal.created' for transaction {transaction_id} (run_id {run_id})")
        else:
            logger.warning(f"Could not publish 'phi3.proposal.created' for run_id {run_id}: phi3_proposal data is not a dict.")

        assessment_payload = {
            "proposal_transaction_id": transaction_id, 
            "assessment_text": hermes_assessment_for_event,
            "orchestrator_run_id": run_id 
        }
        await event_bus.publish("phi3.proposal.assessed", json.dumps(assessment_payload))
        logger.info(f"Published 'phi3.proposal.assessed' for transaction {transaction_id} (run_id {run_id})")

    except RedisError as e:
        error_message = f"RedisError during event publishing for run_id {run_id}: {e}"
        logger.error(error_message)
        current_error = state.get("error")
        updated_error = f"{current_error}\n{error_message}" if current_error else error_message
        state = {**state, "error": updated_error}
    except Exception as e:
        error_message = f"Unexpected error during event publishing for run_id {run_id}: {e}"
        logger.error(error_message)
        current_error = state.get("error")
        updated_error = f"{current_error}\n{error_message}" if current_error else error_message
        state = {**state, "error": updated_error}
    finally:
        if event_bus.redis_client and await event_bus.is_connected(): # Check before closing
            await event_bus.close()
            logger.info(f"EventBus connection closed for run_id {run_id}.")

    final_output_data = {"status": "success", "run_id": run_id, "data": pta_response}
    
    hermes_assessment_text = hermes_assessment_for_event
    if isinstance(hermes_assessment_text, str) and hermes_assessment_text.strip():
        tts_text = hermes_assessment_text.split('.')[0] + "."
        if len(tts_text) > 150: tts_text = tts_text[:150] + "..."
        
        logger.info(f"Requesting TTS for run_id {run_id}: '{tts_text}'")
        tts_payload = {"text": tts_text, "exaggeration": 0.5}
        
        try:
            # Reconnect event_bus if it was closed after previous event publications
            if not (event_bus.redis_client and await event_bus.is_connected()):
                await event_bus.connect()

            tts_response = requests.post(
                "http://localhost:8000/speak", # TODO: Make sidecar URL configurable
                json=tts_payload,
                timeout=30,
                headers={"Accept": "audio/wav"}
            )
            tts_response.raise_for_status()
            logger.info(f"TTS request successful for run_id {run_id}. Received {len(tts_response.content)} bytes.")
            
            if event_bus.redis_client and await event_bus.is_connected():
                tts_event_payload = {
                    "transaction_id": transaction_id, # This is from pta_response
                    "text_spoken": tts_text,
                    "status": "success",
                    "audio_length_bytes": len(tts_response.content),
                    "orchestrator_run_id": run_id
                }
                await event_bus.publish("hermes.assessment.spoken", json.dumps(tts_event_payload))
                logger.info(f"Published 'hermes.assessment.spoken' for transaction {transaction_id} (run_id {run_id})")
            else:
                logger.warning(f"EventBus not connected, skipping 'hermes.assessment.spoken' event for run_id {run_id}.")

        except requests.exceptions.RequestException as e:
            tts_error_message = f"TTS request failed for run_id {run_id}: {e}"
            logger.error(tts_error_message)
            current_error = state.get("error", "")
            state = {**state, "error": f"{current_error}\n{tts_error_message}".strip()}
            final_output_data["tts_error"] = tts_error_message
        except Exception as e:
            unexpected_tts_error = f"Unexpected error during TTS processing for run_id {run_id}: {e}"
            logger.error(unexpected_tts_error)
            current_error = state.get("error", "")
            state = {**state, "error": f"{current_error}\n{unexpected_tts_error}".strip()}
            final_output_data["tts_unexpected_error"] = unexpected_tts_error
        finally:
            # Close event_bus connection if it was opened specifically for TTS event
            if event_bus.redis_client and await event_bus.is_connected():
                 await event_bus.close()
                 logger.info(f"EventBus connection closed after TTS attempt for run_id {run_id}.")


    if state.get("error"):
        final_output_data["status"] = "partial_success_with_errors"
        if "workflow_errors" not in final_output_data and "tts_error" not in final_output_data and "tts_unexpected_error" not in final_output_data :
             final_output_data["workflow_errors"] = state["error"]
    
    # Log the final output string to the state. This is what gets printed/returned by the graph.
    state_final_output_str = json.dumps(final_output_data, indent=2)
    logger.info(f"Final output for run_id {run_id}: {state_final_output_str}")
    return {**state, "final_output": state_final_output_str}


# --- Graph Assembly ---
def build_graph():
    workflow = StateGraph(WorkflowState)

    workflow.add_node("query_market", query_market_node)
    # LangGraph's add_node can take regular synchronous functions.
    # If an async function is added, the graph invocation `graph.ainvoke` must be used.
    # Let's make all async nodes for consistency with event bus operations.
    # query_market_node is sync, but StateGraph handles mixing if graph.invoke is fine.
    # For simplicity now, let's assume all are potentially async and use ainvoke.
    # Reverting query_market_node to be synchronous as it has no async calls.
    # generate_proposal_node and evaluate_proposal_node use `requests` which is sync.
    # To use `ainvoke` properly, these nodes should be async and use an async http client (e.g. httpx)
    # For now, let's make them synchronous and use `graph.invoke()`.
    # If EventBus publish node remains async, then graph.ainvoke is better.
    # Let's adjust node definitions or graph invocation.

    # For this iteration, let's assume synchronous execution for HTTP nodes for simplicity with `requests`.
    # The `publish_events_node` *must* be async due to `EventBus`.
    # This means the graph *must* be run with `ainvoke`.
    # Synchronous nodes are automatically wrapped by LangGraph to be compatible with `ainvoke`.

    workflow.add_node("generate_proposal", generate_proposal_node) # Will be wrapped by LangGraph for ainvoke
    workflow.add_node("evaluate_proposal", evaluate_proposal_node) # Will be wrapped
    workflow.add_node("publish_events", publish_events_node) # Already async

    # Define edges
    workflow.set_entry_point("query_market")
    workflow.add_edge("query_market", "generate_proposal")
    workflow.add_edge("generate_proposal", "evaluate_proposal")
    workflow.add_edge("evaluate_proposal", "publish_events")
    workflow.add_edge("publish_events", END)
    
    # Conditional edges can be added here if logic dictates (e.g., on error)
    # For now, a linear flow. Errors are propagated in the state.

    return workflow.compile()

# --- Market Tick Listener & Workflow Trigger ---
async def market_tick_listener(
    redis_url: str, 
    market_channel: str, 
    ticks_per_proposal: int, 
    graph_app: StateGraph # Compiled LangGraph app
):
    global TICK_BUFFER # Use the global buffer

    event_bus = EventBus(redis_url=redis_url)
    await event_bus.connect()
    logger.info(f"Connected to Redis for market ticks on channel '{market_channel}'. Waiting for data...")

    async def tick_handler(message_data: str):
        global TICK_BUFFER # Ensure modification of global buffer
        try:
            tick = json.loads(message_data)
            TICK_BUFFER.append(tick)
            logger.info(f"Received tick #{len(TICK_BUFFER)}: {tick.get('timestamp', 'N/A')}, Close: {tick.get('close', 'N/A')}")

            if len(TICK_BUFFER) >= ticks_per_proposal:
                logger.info(f"Buffer limit of {ticks_per_proposal} ticks reached. Triggering workflow.")
                
                # Create a unique run ID for this workflow invocation
                current_run_id = f"run_{int(asyncio.get_running_loop().time())}"
                
                # Prepare query for the graph: stringified version of current buffer
                # Could also be a summary, or just the latest N ticks. For now, whole buffer.
                workflow_query = json.dumps(TICK_BUFFER)
                
                initial_state: WorkflowState = {
                    "query": workflow_query,
                    "market_query_result": None,
                    "phi3_raw_proposal_request": None,
                    "phi3_response": None,
                    "propose_trade_adjustments_response": None,
                    "final_output": None,
                    "error": None,
                    "run_id": current_run_id # Include run_id in the state
                }
                
                logger.info(f"Invoking workflow for run_id: {current_run_id} with {len(TICK_BUFFER)} ticks.")
                # Asynchronously invoke the graph. Does not block further ticks if graph runs long.
                # Be mindful of resource limits if many graph instances run concurrently.
                asyncio.create_task(process_workflow_run(graph_app, initial_state))
                
                TICK_BUFFER = [] # Clear buffer after triggering
                logger.info("Tick buffer cleared.")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse tick data as JSON: {message_data}")
        except Exception as e:
            logger.error(f"Error in tick_handler: {e}")

    await event_bus.subscribe(market_channel, tick_handler)
    
    # Keep the listener running indefinitely (or until an external signal)
    try:
        while True:
            await asyncio.sleep(1) # Keep alive, actual work happens in tick_handler via EventBus
    except KeyboardInterrupt:
        logger.info("Market tick listener stopped by user.")
    except Exception as e:
        logger.error(f"Market tick listener faced an unexpected error: {e}")
    finally:
        logger.info("Closing EventBus connection for market tick listener.")
        await event_bus.close()

async def process_workflow_run(graph_app: StateGraph, initial_state: WorkflowState):
    """
    Processes a single workflow run, including logging its final state.
    """
    run_id = initial_state.get("run_id", "unknown_run")
    logger.info(f"Starting background processing for workflow run_id: {run_id}")
    final_state_dict = await graph_app.ainvoke(initial_state)
    
    status = "FAILURE" if final_state_dict.get("error") else "SUCCESS"
    final_output_str = final_state_dict.get("final_output")
    final_output_dict_for_db: Optional[Dict[str, Any]] = None

    if final_output_str:
        try:
            # The final_output_str is already a JSON string from publish_events_node
            final_output_dict_for_db = json.loads(final_output_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse final_output for DB logging (run_id {run_id}): {final_output_str}")
            final_output_dict_for_db = {"raw_output": final_output_str, "parsing_error": "Could not decode JSON"}
    
    # Ensure run_id from the workflow is part of the log entry if not already in final_output_dict_for_db
    if final_output_dict_for_db and "run_id" not in final_output_dict_for_db:
        final_output_dict_for_db["run_id"] = run_id
    elif not final_output_dict_for_db: # Handle cases where final_output_str might be None or unparsable
         final_output_dict_for_db = {"run_id": run_id, "status_details": "No parsable final output string."}


    # The input_query for the log should be the stringified tick data.
    input_query_for_log = initial_state["query"]

    run_log_entry = OrchestratorRunLog(
        input_query=input_query_for_log, 
        final_output=final_output_dict_for_db, 
        status=status,
        error_message=final_state_dict.get("error"),
        run_id_override=run_id # Explicitly pass the run_id
    )
    try:
        log_run(run_log_entry) # log_run should use the run_id_override
        logger.info(f"Successfully logged workflow run {run_id} to LanceDB.")
    except Exception as e:
        logger.error(f"Failed to log workflow run {run_id} to LanceDB: {e}")

    # Output to console (optional, can be noisy in continuous mode)
    if final_state_dict.get("error"):
        logger.error(f"Workflow run {run_id} completed with errors: {final_state_dict['error']}")
    
    # Printing final_output_str can be very verbose if running continuously.
    # Consider logging to a file or conditional printing based on verbosity settings.
    # For now, let's log it.
    logger.info(f"Console output for run {run_id}: {final_output_str if final_output_str else 'No final_output string.'}")


# --- CLI ---
async def main_async(args):
    # Build the graph application once
    graph_app = build_graph()
    
    logger.info(f"Starting Osiris Policy Orchestrator in event-driven mode.")
    logger.info(f"Listening to Redis channel '{args.market_channel}' on {args.redis_url}")
    logger.info(f"Triggering proposal workflow every {args.ticks_per_proposal} ticks.")

    # Start the market tick listener
    # This function will run indefinitely until stopped (e.g., KeyboardInterrupt)
    await market_tick_listener(
        redis_url=args.redis_url,
        market_channel=args.market_channel,
        ticks_per_proposal=args.ticks_per_proposal,
        graph_app=graph_app
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Osiris Policy Orchestrator - Event-Driven Mode")
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0", 
                        help="Redis URL for market data and event bus.")
    parser.add_argument("--market_channel", type=str, default=MARKET_TICKS_CHANNEL,
                        help="Redis channel to listen for market ticks.")
    parser.add_argument("--ticks_per_proposal", type=int, default=10,
                        help="Number of market ticks to buffer before triggering a new proposal workflow.")
    
    # Potentially add other config args like sidecar URL if needed
    
    args = parser.parse_args()
    asyncio.run(main_async(args))
