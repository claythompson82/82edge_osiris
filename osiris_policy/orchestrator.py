import argparse
import json
import requests
import logging
from typing import TypedDict, Dict, Any, Optional

# LangGraph
from langgraph.graph import StateGraph, END

# EventBus
from llm_sidecar.event_bus import EventBus, RedisError

# Database for logging runs
from llm_sidecar.db import log_run, OrchestratorRunLog


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- State Definition ---
class WorkflowState(TypedDict):
    query: str
    market_query_result: Optional[str]
    phi3_raw_proposal_request: Optional[Dict[str, Any]] # What we send to phi3
    phi3_response: Optional[Dict[str, Any]] # What phi3 returns (parsed JSON)
    # propose_trade_adjustments_request: Optional[Dict[str, Any]] # What we send to propose_trade_adjustments
    # This is implicitly the hermes_prompt in EvaluateProposal
    propose_trade_adjustments_response: Optional[Dict[str, Any]] # Full response from /propose_trade_adjustments
    final_output: Optional[str] # For CLI output
    error: Optional[str] # To capture any errors in the workflow


# --- Node Implementations ---

def query_market_node(state: WorkflowState) -> WorkflowState:
    logger.info("Executing QueryMarket node...")
    query = state["query"]
    # Mock implementation
    market_query_result = f"Market query processed for: '{query}'. Result: Favorable conditions."
    logger.info(f"Market query result: {market_query_result}")
    return {**state, "market_query_result": market_query_result}


async def generate_proposal_node(state: WorkflowState) -> WorkflowState:
    logger.info("Executing GenerateProposal node...")
    if state.get("error"): # If there was an error in a previous step, skip
        return state

    market_data = state.get("market_query_result", "No market data available.")
    
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
    logger.info(f"Sending to /generate?model_id=phi3: {json.dumps(phi3_payload)}")

    try:
        # Using requests.post for synchronous call as langgraph nodes are typically synchronous
        # If async HTTP client is needed, the graph execution itself would need to be async.
        # For now, standard requests will be used. Consider httpx for async if required later.
        response = requests.post(
            "http://localhost:8000/generate?model_id=phi3",
            json=phi3_payload,
            timeout=60  # seconds
        )
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
        phi3_response_json = response.json()
        logger.info(f"Received from /generate?model_id=phi3: {json.dumps(phi3_response_json)}")
        
        # Validate if the response is the expected JSON object, not another error structure
        if isinstance(phi3_response_json, dict) and "error" not in phi3_response_json:
            return {**state, "phi3_raw_proposal_request": phi3_payload, "phi3_response": phi3_response_json}
        else:
            error_message = f"Phi-3 generation failed. Response: {json.dumps(phi3_response_json)}"
            logger.error(error_message)
            return {**state, "error": error_message, "phi3_response": phi3_response_json}

    except requests.exceptions.RequestException as e:
        error_message = f"HTTP request to Phi-3 failed: {e}"
        logger.error(error_message)
        return {**state, "error": error_message}
    except json.JSONDecodeError as e:
        error_message = f"Failed to decode JSON response from Phi-3: {e}. Response text: {response.text if response else 'No response'}"
        logger.error(error_message)
        return {**state, "error": error_message}


async def evaluate_proposal_node(state: WorkflowState) -> WorkflowState:
    logger.info("Executing EvaluateProposal node...")
    if state.get("error"):
        return state

    phi3_proposal = state.get("phi3_response")
    if not phi3_proposal or not isinstance(phi3_proposal, dict):
        error_message = "Phi-3 proposal is missing or invalid in state for evaluation."
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
    
    raw_phi3_prompt_info = state.get("phi3_raw_proposal_request")
    if not raw_phi3_prompt_info or "prompt" not in raw_phi3_prompt_info:
        error_message = "Original prompt for Phi-3 is missing, cannot call /propose_trade_adjustments."
        logger.error(error_message)
        return {**state, "error": error_message}

    # The prompt for /propose_trade_adjustments should be the *initial user query* or something similar
    # that the sidecar can use to generate its own Phi-3 proposal and then assess it.
    # The existing server.py /propose_trade_adjustments takes a "prompt" which is a user query.
    # Let's use the initial query from the state.
    user_query_for_endpoint = state["query"]

    pta_payload = {
        "prompt": user_query_for_endpoint, # This is the "user query" it expects
        "max_length": 512 # This is for the internal Phi-3 call within the endpoint.
    }
    logger.info(f"Sending to /propose_trade_adjustments: {json.dumps(pta_payload)}")

    try:
        response = requests.post(
            "http://localhost:8000/propose_trade_adjustments",
            json=pta_payload,
            timeout=120  # This endpoint does two LLM calls, so longer timeout
        )
        response.raise_for_status()
        pta_response_json = response.json()
        logger.info(f"Received from /propose_trade_adjustments: {json.dumps(pta_response_json)}")
        
        if isinstance(pta_response_json, dict) and "error" not in pta_response_json:
             # The response should contain "phi3_proposal" and "hermes_assessment"
            return {**state, "propose_trade_adjustments_response": pta_response_json}
        else:
            error_message = f"Call to /propose_trade_adjustments failed. Response: {json.dumps(pta_response_json)}"
            logger.error(error_message)
            return {**state, "error": error_message, "propose_trade_adjustments_response": pta_response_json}


    except requests.exceptions.RequestException as e:
        error_message = f"HTTP request to /propose_trade_adjustments failed: {e}"
        logger.error(error_message)
        return {**state, "error": error_message}
    except json.JSONDecodeError as e:
        error_message = f"Failed to decode JSON response from /propose_trade_adjustments: {e}. Response text: {response.text if response else 'No response'}"
        logger.error(error_message)
        return {**state, "error": error_message}


async def publish_events_node(state: WorkflowState) -> WorkflowState:
    logger.info("Executing PublishEvents node...")
    if state.get("error"):
        # Potentially publish an error event or just log and end
        logger.warning("Skipping event publication due to previous error in workflow.")
        # Set final output to error if not already set by a failing node
        final_output = state.get("final_output")
        if not final_output:
            final_output = json.dumps({"status": "error", "details": state["error"]})
        return {**state, "final_output": final_output }


    pta_response = state.get("propose_trade_adjustments_response")
    if not pta_response or "phi3_proposal" not in pta_response or "hermes_assessment" not in pta_response:
        error_message = "Response from /propose_trade_adjustments is missing or malformed in state. Cannot publish events."
        logger.error(error_message)
        # Even if events can't be published, the workflow might have produced a usable result.
        # Let's set final_output from pta_response if available, or phi3_response.
        output_data = pta_response if pta_response else state.get("phi3_response", {"error": "No valid proposal or assessment found"})
        return {**state, "error": error_message, "final_output": json.dumps(output_data)}

    phi3_proposal_for_event = pta_response["phi3_proposal"]
    hermes_assessment_for_event = pta_response["hermes_assessment"]
    # The server.py /propose_trade_adjustments returns a "transaction_id" at the top level of its response
    # This is useful for correlating events.
    transaction_id = pta_response.get("transaction_id", "unknown_transaction")


    event_bus = EventBus(redis_url="redis://localhost:6379/0")
    try:
        await event_bus.connect()

        # Publish phi3.proposal.created
        # The server already publishes this exact event with this payload structure from its own perspective.
        # Orchestrator is re-publishing based on what it received.
        # Payload for phi3.proposal.created is the phi3_json itself.
        if isinstance(phi3_proposal_for_event, dict):
            await event_bus.publish("phi3.proposal.created", json.dumps(phi3_proposal_for_event))
            logger.info(f"Published 'phi3.proposal.created' for transaction {transaction_id}")
        else:
            logger.warning("Could not publish 'phi3.proposal.created': phi3_proposal data is not a dict.")

        # Publish phi3.proposal.assessed
        # The server also publishes this. Orchestrator is re-publishing.
        # Payload should be JSON string containing assessment and relevant IDs.
        assessment_payload = {
            "proposal_transaction_id": transaction_id, # From /propose_trade_adjustments response
            "assessment_text": hermes_assessment_for_event,
            # Add other relevant fields if necessary from pta_response or state
        }
        await event_bus.publish("phi3.proposal.assessed", json.dumps(assessment_payload))
        logger.info(f"Published 'phi3.proposal.assessed' for transaction {transaction_id}")

    except RedisError as e:
        error_message = f"RedisError during event publishing: {e}"
        logger.error(error_message)
        # Update state with this error; previous data is still valuable
        current_error = state.get("error")
        updated_error = f"{current_error}\n{error_message}" if current_error else error_message
        # Decide if this error should overwrite the final_output or just be logged.
        # For now, let's preserve the LLM output and add this as an additional error.
        state = {**state, "error": updated_error}
    except Exception as e: # Catch any other unexpected errors during event publishing
        error_message = f"Unexpected error during event publishing: {e}"
        logger.error(error_message)
        current_error = state.get("error")
        updated_error = f"{current_error}\n{error_message}" if current_error else error_message
        state = {**state, "error": updated_error}
    finally:
        await event_bus.close()
        logger.info("EventBus connection closed.")

    # Set final output for CLI
    # The task asks to print the final JSON output (e.g. hermes_response or structured combination)
    # The `propose_trade_adjustments_response` contains both phi3 and hermes parts.
    final_output_data = {"status": "success", "data": pta_response}
    if state.get("error"): # If an error occurred during event publishing but we have data
        final_output_data["event_publishing_error"] = state["error"]
        
    return {**state, "final_output": json.dumps(final_output_data, indent=2)}


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

# --- CLI ---
async def main_async(query: str):
    app = build_graph()
    initial_state: WorkflowState = {
        "query": query,
        "market_query_result": None,
        "phi3_raw_proposal_request": None,
        "phi3_response": None,
        "propose_trade_adjustments_response": None,
        "final_output": None,
        "error": None,
    }
    
    logger.info(f"Invoking workflow with query: '{query}'")
    # Use ainvoke because publish_events_node is async
    final_state_dict = await app.ainvoke(initial_state) # Renamed to avoid conflict with WorkflowState type hint
    
    # Log the run
    status = "FAILURE" if final_state_dict.get("error") else "SUCCESS"
    final_output_str = final_state_dict.get("final_output")
    final_output_dict_for_db: Optional[Dict[str, Any]] = None

    if final_output_str:
        try:
            final_output_dict_for_db = json.loads(final_output_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse final_output for DB logging: {final_output_str}")
            # Store the raw string or an error placeholder if parsing fails
            final_output_dict_for_db = {"raw_output": final_output_str, "parsing_error": "Could not decode JSON"}

    run_log_entry = OrchestratorRunLog(
        input_query=query,
        final_output=final_output_dict_for_db, # This should be a dict
        status=status,
        error_message=final_state_dict.get("error")
    )
    try:
        log_run(run_log_entry)
        logger.info(f"Successfully logged run {run_log_entry.run_id} to LanceDB.")
    except Exception as e:
        # log_run itself prints an error, but we can log here too if needed for orchestrator context
        logger.error(f"Failed to log run {run_log_entry.run_id} to LanceDB from orchestrator: {e}")

    # Output to console
    if final_state_dict.get("error"):
        logger.error(f"Workflow completed with errors: {final_state_dict['error']}")
    
    if final_output_str:
        print(final_output_str) # Print the original string final_output
    else:
        # Fallback if final_output wasn't set
        logger.warning("Final output not explicitly set in state. Dumping error or partial state.")
        if final_state_dict.get("error"):
            print(json.dumps({"status": "error", "details": final_state_dict["error"]}, indent=2))
        else:
            relevant_output = {
                "query": final_state_dict.get("query"),
                "market_query_result": final_state_dict.get("market_query_result"),
                "phi3_response": final_state_dict.get("phi3_response"),
                "propose_trade_adjustments_response": final_state_dict.get("propose_trade_adjustments_response")
            }
            print(json.dumps(relevant_output, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Osiris Policy Orchestrator CLI")
    parser.add_argument("user_query", type=str, help="The user query to initiate the workflow.")
    args = parser.parse_args()

    # Python 3.7+ for asyncio.run
    import asyncio
    asyncio.run(main_async(args.user_query))
