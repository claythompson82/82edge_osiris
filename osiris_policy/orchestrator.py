import argparse
import json
import requests
import logging
import asyncio  # Added for new event-driven model
import time  # Added for timestamp
import os
from typing import TypedDict, Dict, Any, Optional, List

# LangGraph
from langgraph.graph import StateGraph, END

# EventBus
from llm_sidecar.event_bus import EventBus, RedisError

# Database for logging runs
from llm_sidecar.db import log_run, OrchestratorRunSchema
from advisor.risk_gate import (
    accept as risk_gate_accept,
    log_decision as log_risk_advice,
    AdviceLog,
    init_advice_table,
)
import datetime

from common.otel_init import init_otel
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _setup_tracer() -> "trace.Tracer":
    """Initialise OTLP tracing if OTEL_EXPORTER_ENDPOINT is set."""
    endpoint = os.getenv("OTEL_EXPORTER_ENDPOINT")
    if not endpoint:
        logger.info("OTEL_EXPORTER_ENDPOINT not set; traces disabled.")
        return trace.get_tracer(__name__)
    if not endpoint.startswith("http"):
        endpoint = f"http://{endpoint}"
    provider = TracerProvider()
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    logger.info("OpenTelemetry tracing enabled")
    return trace.get_tracer(__name__)


tracer = _setup_tracer()

# --- Global Variables & Constants ---
TICK_BUFFER: List[Dict[str, Any]] = []
# TICKS_PER_PROPOSAL will be set by CLI args
MARKET_TICKS_CHANNEL = "market.ticks"  # Default, can be overridden by CLI
# REDIS_URL will be set by CLI args

RISK_GATE_CONFIG = {
    "max_position_size_pct": 0.01,  # 1% of NAV
    "max_daily_loss_pct": 0.02,  # 2% of NAV
    "lancedb_path": "/app/lancedb_data",  # Align with llm_sidecar.db DB_ROOT
}


# --- State Definition ---
class WorkflowState(TypedDict):
    query: str  # This will now be the stringified tick buffer or a summary
    market_query_result: Optional[str]  # Result of processing the query/ticks
    phi3_raw_proposal_request: Optional[Dict[str, Any]]
    phi3_response: Optional[Dict[str, Any]]
    propose_trade_adjustments_response: Optional[Dict[str, Any]]
    final_output: Optional[str]
    error: Optional[str]
    run_id: Optional[str]  # To track individual workflow runs triggered by ticks
    current_nav: Optional[float]  # Current Net Asset Value
    daily_pnl: Optional[float]  # Current Daily Profit and Loss
    risk_gate_decision: Optional[Dict[str, Any]]  # Output from risk_gate_accept


# --- Node Implementations ---


def query_market_node(state: WorkflowState) -> WorkflowState:
    logger.info(
        f"Executing QueryMarket node for run_id: {state.get('run_id', 'N/A')}..."
    )
    # The 'query' field in the state now contains the stringified list of buffered ticks.
    # This node will transform it into a "market_query_result".
    # For now, let's just pass it through or add a simple message.
    raw_tick_data_str = state["query"]
    market_query_result = (
        f"Processed market data based on recent ticks: {raw_tick_data_str}"
    )
    logger.info(
        f"Market query result for run_id {state.get('run_id', 'N/A')}: {market_query_result}"
    )
    return {**state, "market_query_result": market_query_result}


async def generate_proposal_node(state: WorkflowState) -> WorkflowState:
    logger.info(
        f"Executing GenerateProposal node for run_id: {state.get('run_id', 'N/A')}..."
    )
    if state.get("error"):
        return state

    market_data = state.get("market_query_result", "No market data available.")
    if not market_data:  # Should not happen if query_market_node ran
        logger.warning(
            f"No market data in state for run_id: {state.get('run_id', 'N/A')}. Using default."
        )
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

    phi3_payload = {"prompt": prompt_text, "max_length": 512}  # Adjust as needed
    logger.info(
        f"Sending to /generate?model_id=phi3 for run_id {state.get('run_id', 'N/A')}: {json.dumps(phi3_payload)}"
    )

    try:
        response = requests.post(
            "http://localhost:8000/generate?model_id=phi3",  # TODO: Make sidecar URL configurable
            json=phi3_payload,
            timeout=60,
        )
        response.raise_for_status()
        phi3_response_json = response.json()
        logger.info(
            f"Received from /generate?model_id=phi3 for run_id {state.get('run_id', 'N/A')}: {json.dumps(phi3_response_json)}"
        )

        if isinstance(phi3_response_json, dict) and "error" not in phi3_response_json:
            return {
                **state,
                "phi3_raw_proposal_request": phi3_payload,
                "phi3_response": phi3_response_json,
            }
        else:
            error_message = f"Phi-3 generation failed for run_id {state.get('run_id', 'N/A')}. Response: {json.dumps(phi3_response_json)}"
            logger.error(error_message)
            return {
                **state,
                "error": error_message,
                "phi3_response": phi3_response_json,
            }

    except requests.exceptions.RequestException as e:
        error_message = (
            f"HTTP request to Phi-3 failed for run_id {state.get('run_id', 'N/A')}: {e}"
        )
        logger.error(error_message)
        return {**state, "error": error_message}
    except json.JSONDecodeError as e:
        error_message = f"Failed to decode JSON response from Phi-3 for run_id {state.get('run_id', 'N/A')}: {e}. Response text: {response.text if response else 'No response'}"
        logger.error(error_message)
        return {**state, "error": error_message}


async def evaluate_proposal_node(state: WorkflowState) -> WorkflowState:
    logger.info(
        f"Executing EvaluateProposal node for run_id: {state.get('run_id', 'N/A')}..."
    )
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

    pta_payload = {"prompt": prompt_for_pta, "max_length": 512}
    logger.info(
        f"Sending to /propose_trade_adjustments for run_id {state.get('run_id', 'N/A')}: {json.dumps(pta_payload)}"
    )

    try:
        response = requests.post(
            "http://localhost:8000/propose_trade_adjustments",  # TODO: Make sidecar URL configurable
            json=pta_payload,
            timeout=120,
        )
        response.raise_for_status()
        pta_response_json = response.json()
        logger.info(
            f"Received from /propose_trade_adjustments for run_id {state.get('run_id', 'N/A')}: {json.dumps(pta_response_json)}"
        )

        if isinstance(pta_response_json, dict) and "error" not in pta_response_json:
            return {**state, "propose_trade_adjustments_response": pta_response_json}
        else:
            error_message = f"Call to /propose_trade_adjustments failed for run_id {state.get('run_id', 'N/A')}. Response: {json.dumps(pta_response_json)}"
            logger.error(error_message)
            return {
                **state,
                "error": error_message,
                "propose_trade_adjustments_response": pta_response_json,
            }

    except requests.exceptions.RequestException as e:
        error_message = f"HTTP request to /propose_trade_adjustments failed for run_id {state.get('run_id', 'N/A')}: {e}"
        logger.error(error_message)
        return {**state, "error": error_message}
    except json.JSONDecodeError as e:
        error_message = f"Failed to decode JSON response from /propose_trade_adjustments for run_id {state.get('run_id', 'N/A')}: {e}. Response text: {response.text if response else 'No response'}"
        logger.error(error_message)
        return {**state, "error": error_message}


async def publish_events_node(state: WorkflowState) -> WorkflowState:
    run_id = state.get("run_id", "N/A")
    logger.info(f"Executing PublishEvents node for run_id: {run_id}...")

    event_bus = EventBus(
        redis_url="redis://localhost:6379/0"
    )  # TODO: Make this configurable

    phi3_proposal_from_state = state.get(
        "phi3_response", {"error": "Original proposal not found in state."}
    )
    if not isinstance(phi3_proposal_from_state, dict):
        logger.warning(
            f"Run_id {run_id}: phi3_response was not a dict: {type(phi3_proposal_from_state)}. Replacing with error dict."
        )
        phi3_proposal_from_state = {
            "error": f"Original proposal was not a valid dictionary, type: {type(phi3_proposal_from_state)}."
        }
        if not state.get("error"):
            state["error"] = phi3_proposal_from_state["error"]

    risk_gate_verdict = state.get("risk_gate_decision")
    if not risk_gate_verdict:
        logger.error(
            f"Run_id {run_id}: risk_gate_decision critically missing in publish_events_node. Defaulting to rejection."
        )
        risk_gate_verdict = {
            "accepted": False,
            "reason": "Risk gate decision critically missing in state at publish node.",
        }
        if not state.get("error"):
            state["error"] = risk_gate_verdict["reason"]

    advice_event_payload = {
        "run_id": run_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "proposal": phi3_proposal_from_state,
        "risk_gate_verdict": risk_gate_verdict,
        "hermes_assessment": None,
    }

    pta_response = state.get("propose_trade_adjustments_response")
    hermes_assessment_text = None
    transaction_id = f"tx_run_{run_id}"  # Default, updated if pta_response is available

    try:
        await event_bus.connect()
        logger.info(f"Event bus connected for run_id {run_id} in publish_events_node.")

        # --- Conditional: Hermes Evaluation and related events (if risk accepted AND no preceding error) ---
        # Check state["error"] *before* deciding to process PTA-related events
        current_workflow_error = state.get("error")
        if risk_gate_verdict.get("accepted") and not current_workflow_error:
            if (
                pta_response
                and isinstance(pta_response, dict)
                and "hermes_assessment" in pta_response
            ):
                logger.info(
                    f"Run_id {run_id}: Proposal was risk-accepted and evaluated by Hermes."
                )
                hermes_assessment_text = pta_response["hermes_assessment"]
                advice_event_payload["hermes_assessment"] = hermes_assessment_text
                transaction_id = pta_response.get("transaction_id", transaction_id)

                phi3_proposal_from_pta = pta_response.get("phi3_proposal")
                if isinstance(phi3_proposal_from_pta, dict):
                    event_phi3_proposal_pta = {
                        **phi3_proposal_from_pta,
                        "orchestrator_run_id": run_id,
                        "transaction_id": transaction_id,
                    }
                    await event_bus.publish(
                        "phi3.proposal.created", json.dumps(event_phi3_proposal_pta)
                    )
                    logger.info(
                        f"Published 'phi3.proposal.created' (from PTA) for tx {transaction_id} (run_id {run_id})"
                    )

                assessment_payload = {
                    "proposal_transaction_id": transaction_id,
                    "assessment_text": hermes_assessment_text,
                    "orchestrator_run_id": run_id,
                }
                await event_bus.publish(
                    "phi3.proposal.assessed", json.dumps(assessment_payload)
                )
                logger.info(
                    f"Published 'phi3.proposal.assessed' for tx {transaction_id} (run_id {run_id})"
                )

            elif (
                not pta_response
            ):  # Risk accepted, no error, but pta_response is missing
                logger.warning(
                    f"Run_id {run_id}: Risk accepted, no workflow error, but pta_response missing. Assessment events skipped."
                )
                state["error"] = (
                    f"{state.get('error', '')} Missing pta_response after risk acceptance.".strip()
                )

        elif risk_gate_verdict.get("accepted") and current_workflow_error:
            logger.warning(
                f"Run_id {run_id}: Risk accepted, but an error occurred ('{current_workflow_error}'). Skipping Hermes/PTA related event publishing."
            )

        # --- Always publish 'advice.generated' event ---
        logger.info(
            f"Publishing 'advice.generated' for run_id {run_id}. Accepted: {risk_gate_verdict.get('accepted')}, Reason: {risk_gate_verdict.get('reason')}"
        )
        await event_bus.publish("advice.generated", json.dumps(advice_event_payload))
        logger.info(f"Successfully published 'advice.generated' for run_id {run_id}.")

        # --- Conditional: TTS for Hermes assessment (if hermes_assessment_text is available) ---
        if (
            hermes_assessment_text
            and isinstance(hermes_assessment_text, str)
            and hermes_assessment_text.strip()
        ):
            tts_text_to_speak = hermes_assessment_text.split(".")[0] + "."
            if len(tts_text_to_speak) > 150:
                tts_text_to_speak = tts_text_to_speak[:150] + "..."

            if not transaction_id:
                logger.error(
                    f"Run_id {run_id}: transaction_id is missing. Skipping TTS acknowledgement and last_speak_ms setting."
                )
            else:
                logger.info(
                    f"Requesting TTS for Hermes assessment, run_id {run_id}, transaction_id {transaction_id}: '{tts_text_to_speak}'"
                )
                tts_payload = {"text": tts_text_to_speak, "exaggeration": 0.5}
                ack_received = False

                try:
                    # 2. Trigger TTS Synthesis via /speak Endpoint
                    tts_http_response = requests.post(
                        "http://localhost:8000/speak", json=tts_payload, timeout=30
                    )  # Removed headers={"Accept": "audio/wav"} as we don't consume audio here
                    tts_http_response.raise_for_status()
                    logger.info(
                        f"TTS request successful for run_id {run_id}, transaction_id {transaction_id}."
                    )

                    # 3. Await Acknowledgement via Redis
                    ack_channel_name = f"tts.acknowledged.{transaction_id}"
                    pubsub = event_bus.redis_client.pubsub()

                    try:
                        await pubsub.subscribe(ack_channel_name)
                        logger.info(
                            f"Run_id {run_id}, transaction_id {transaction_id}: Subscribed to Redis channel '{ack_channel_name}' for TTS acknowledgement."
                        )

                        # Wait for a message
                        while True:  # Loop to get message with timeout
                            message = await pubsub.get_message(
                                ignore_subscribe_messages=True, timeout=30.0
                            )  # 30s timeout
                            if message:
                                logger.info(
                                    f"Run_id {run_id}, transaction_id {transaction_id}: Received TTS acknowledgement: {message['data']}"
                                )
                                ack_received = True
                                break  # Exit loop once message is received
                            else:  # Timeout occurred
                                logger.warning(
                                    f"Run_id {run_id}, transaction_id {transaction_id}: Timeout waiting for TTS acknowledgement on '{ack_channel_name}'."
                                )
                                break  # Exit loop on timeout
                    except Exception as e_redis_sub:
                        logger.error(
                            f"Run_id {run_id}, transaction_id {transaction_id}: Error during Redis subscribe/listen for TTS ack: {e_redis_sub}"
                        )
                        state["error"] = (
                            f"{state.get('error', '')} TTS Redis Ack Error: {e_redis_sub}".strip()
                        )
                    finally:
                        if pubsub:
                            try:
                                await pubsub.unsubscribe(ack_channel_name)
                                logger.info(
                                    f"Run_id {run_id}, transaction_id {transaction_id}: Unsubscribed from '{ack_channel_name}'."
                                )
                                await pubsub.close()  # Close the pubsub connection object
                            except Exception as e_pubsub_close:
                                logger.error(
                                    f"Run_id {run_id}, transaction_id {transaction_id}: Error closing pubsub for TTS ack: {e_pubsub_close}"
                                )

                    # 5. Event Publishing (hermes.assessment.spoken) - only if ack_received
                    if ack_received:
                        tts_event_payload = {
                            "transaction_id": transaction_id,
                            "text_spoken": tts_text_to_speak,
                            "status": "success_acknowledged",
                            "orchestrator_run_id": run_id,
                        }
                        await event_bus.publish(
                            "hermes.assessment.spoken", json.dumps(tts_event_payload)
                        )
                        logger.info(
                            f"Published 'hermes.assessment.spoken' for tx {transaction_id} (run_id {run_id}) after acknowledgement."
                        )

                        # 4. Set last_speak_ms Redis Key - only if ack_received
                        last_speak_key = f"last_speak_ms.{transaction_id}"
                        current_time_ms = int(time.time() * 1000)
                        try:
                            await event_bus.redis_client.set(
                                last_speak_key, current_time_ms, ex=86400
                            )  # 86400 seconds = 24 hours
                            logger.info(
                                f"Run_id {run_id}, transaction_id {transaction_id}: Successfully set Redis key '{last_speak_key}' to {current_time_ms}."
                            )
                        except Exception as e_redis_set:
                            logger.error(
                                f"Run_id {run_id}, transaction_id {transaction_id}: Failed to set Redis key '{last_speak_key}': {e_redis_set}"
                            )
                            state["error"] = (
                                f"{state.get('error', '')} Redis Set Error for last_speak_ms: {e_redis_set}".strip()
                            )
                    else:  # ack_received is False
                        logger.warning(
                            f"Run_id {run_id}, transaction_id {transaction_id}: TTS acknowledgement not received. Skipping 'hermes.assessment.spoken' event and 'last_speak_ms' key setting."
                        )

                except requests.exceptions.RequestException as e_tts_req:
                    tts_error_msg = f"TTS request failed for run_id {run_id}, transaction_id {transaction_id}: {e_tts_req}"
                    logger.error(tts_error_msg)
                    state["error"] = f"{state.get('error', '')} {tts_error_msg}".strip()
                except Exception as e_tts_other:
                    tts_error_msg = f"Unexpected error in TTS processing for run_id {run_id}, transaction_id {transaction_id}"
                    logger.exception(tts_error_msg)
                    state["error"] = (
                        f"{state.get('error', '')} {tts_error_msg}: {e_tts_other}".strip()
                    )
        else:
            if not (
                hermes_assessment_text
                and isinstance(hermes_assessment_text, str)
                and hermes_assessment_text.strip()
            ):
                logger.info(
                    f"Run_id {run_id}: No valid hermes_assessment_text. Skipping TTS processing."
                )

    except RedisError as e_redis:
        error_message = f"RedisError during event publishing for run_id {run_id}"
        logger.exception(error_message)
        state["error"] = f"{state.get('error', '')} {error_message}: {e_redis}".strip()
    except Exception as e_general:
        error_message = f"Unexpected error during event publishing for run_id {run_id}"
        logger.exception(error_message)
        state["error"] = (
            f"{state.get('error', '')} {error_message}: {e_general}".strip()
        )
    finally:
        if (
            event_bus.redis_client and await event_bus.is_connected()
        ):  # Check if client exists before checking connection
            await event_bus.close()
            logger.info(
                f"EventBus connection closed for publish_events_node run_id {run_id}."
            )

    # --- Construct Final Output ---
    final_output_data = {
        "run_id": run_id,
        "status": "success",
        "risk_gate_verdict": risk_gate_verdict,
        "phi3_proposal": phi3_proposal_from_state,
        "hermes_assessment": hermes_assessment_text,
    }

    if not risk_gate_verdict.get("accepted"):
        final_output_data["status"] = "success_risk_rejected"
        final_output_data["details"] = (
            f"Proposal rejected by risk gate: {risk_gate_verdict.get('reason')}"
        )

    current_final_error = state.get("error")
    if current_final_error:
        final_output_data["workflow_errors"] = current_final_error
        if final_output_data["status"] == "success":
            final_output_data["status"] = "partial_success_with_errors"
        elif final_output_data["status"] == "success_risk_rejected":
            final_output_data["status"] = "error_risk_rejected"
        else:
            final_output_data["status"] = "error"

        logger.error(
            f"Workflow for run_id {run_id} completed with errors: {current_final_error}"
        )

    state_final_output_str = json.dumps(final_output_data, indent=2)
    logger.info(f"Final output for run_id {run_id}: {state_final_output_str}")
    return {**state, "final_output": state_final_output_str}


async def risk_management_node(state: WorkflowState) -> WorkflowState:
    run_id = state.get("run_id", "N/A")
    logger.info(f"Executing RiskManagement node for run_id: {run_id}...")

    current_error = state.get("error")
    if current_error:
        logger.warning(
            f"Skipping risk management for run_id {run_id} due to previous error: {current_error}"
        )
        return {
            **state,
            "risk_gate_decision": {
                "accepted": False,
                "reason": f"Skipped due to prior error: {current_error}",
            },
        }

    phi3_response = state.get("phi3_response")
    current_nav = state.get("current_nav")
    daily_pnl = state.get("daily_pnl")

    if not isinstance(phi3_response, dict) or not phi3_response:
        error_message = (
            f"Phi-3 proposal (phi3_response) is missing or invalid for run_id {run_id}."
        )
        logger.error(error_message)
        return {
            **state,
            "error": error_message,
            "risk_gate_decision": {"accepted": False, "reason": error_message},
        }

    if "error" in phi3_response:  # Check if the proposal itself is an error object
        error_message = f"Phi-3 proposal itself contains an error for run_id {run_id}: {phi3_response['error']}"
        logger.error(error_message)
        return {
            **state,
            "error": error_message,
            "risk_gate_decision": {"accepted": False, "reason": error_message},
        }

    if current_nav is None or daily_pnl is None:
        error_message = f"NAV or P&L not provided in state for run_id {run_id}. Cannot perform risk assessment."
        logger.error(error_message)
        return {
            **state,
            "error": error_message,
            "risk_gate_decision": {"accepted": False, "reason": error_message},
        }

    try:
        logger.info(
            f"Calling risk_gate_accept for run_id {run_id} with NAV: {current_nav}, P&L: {daily_pnl}, Proposal: {json.dumps(phi3_response)}"
        )
        decision_result = risk_gate_accept(
            proposal=phi3_response,
            current_nav=current_nav,
            daily_pnl=daily_pnl,
            risk_config=RISK_GATE_CONFIG,
        )
        logger.info(f"Risk gate decision for run_id {run_id}: {decision_result}")

        advice_entry = AdviceLog(
            run_id=run_id,
            proposal=phi3_response,
            accepted=decision_result["accepted"],
            reason=decision_result["reason"],
            nav_before_trade=current_nav,
            daily_pnl_before_trade=daily_pnl,
        )
        log_risk_advice(
            advice_log_entry=advice_entry, db_path_str=RISK_GATE_CONFIG["lancedb_path"]
        )
        logger.info(
            f"Logged risk advice for run_id {run_id} (advice_id: {advice_entry.advice_id})."
        )

        return {**state, "risk_gate_decision": decision_result}

    except Exception as e:
        error_message = f"Error during risk_gate_accept or logging for run_id {run_id}"
        logger.exception(error_message)  # Automatically includes exception info
        return {
            **state,
            "error": f"{error_message}: {e}",
            "risk_gate_decision": {
                "accepted": False,
                "reason": f"{error_message}: {e}",
            },
        }


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

    workflow.add_node(
        "generate_proposal", generate_proposal_node
    )  # Will be wrapped by LangGraph for ainvoke
    workflow.add_node("risk_management", risk_management_node)  # Add new node
    workflow.add_node("evaluate_proposal", evaluate_proposal_node)  # Will be wrapped
    workflow.add_node("publish_events", publish_events_node)  # Already async

    # Define edges
    workflow.set_entry_point("query_market")
    workflow.add_edge("query_market", "generate_proposal")
    # workflow.add_edge("generate_proposal", "evaluate_proposal") # This is replaced
    workflow.add_edge("generate_proposal", "risk_management")

    def should_evaluate_proposal(state: WorkflowState) -> str:
        run_id = state.get("run_id", "N/A")

        current_error = state.get("error")
        if current_error:
            logger.warning(
                f"Conditional edge for run_id {run_id}: Error in state ('{current_error}'), proceeding to publish_events."
            )
            return "publish_events"

        risk_decision = state.get("risk_gate_decision")
        if not risk_decision:
            logger.error(
                f"Conditional edge for run_id {run_id}: risk_gate_decision is missing in state. This is unexpected. Proceeding to publish_events to report."
            )
            # Accumulate error
            state["error"] = (
                f"{state.get('error', '')} Critical: risk_gate_decision missing at conditional edge.".strip()
            )
            return "publish_events"

        if risk_decision.get("accepted"):
            logger.info(
                f"Conditional edge for run_id {run_id}: Risk gate accepted. Proceeding to 'evaluate_proposal'."
            )
            return "evaluate_proposal"
        else:
            logger.info(
                f"Conditional edge for run_id {run_id}: Risk gate rejected (Reason: {risk_decision.get('reason', 'Unknown')}). Skipping 'evaluate_proposal', proceeding to 'publish_events'."
            )
            return "publish_events"

    workflow.add_conditional_edges(
        "risk_management",
        should_evaluate_proposal,
        {"evaluate_proposal": "evaluate_proposal", "publish_events": "publish_events"},
    )
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
    graph_app: StateGraph,  # Compiled LangGraph app
):
    global TICK_BUFFER  # Use the global buffer

    event_bus = EventBus(redis_url=redis_url)
    await event_bus.connect()
    logger.info(
        f"Connected to Redis for market ticks on channel '{market_channel}'. Waiting for data..."
    )

    async def tick_handler(message_data: str):
        global TICK_BUFFER  # Ensure modification of global buffer
        try:
            tick = json.loads(message_data)
            TICK_BUFFER.append(tick)
            logger.info(
                f"Received tick #{len(TICK_BUFFER)}: {tick.get('timestamp', 'N/A')}, Close: {tick.get('close', 'N/A')}"
            )

            if len(TICK_BUFFER) >= ticks_per_proposal:
                logger.info(
                    f"Buffer limit of {ticks_per_proposal} ticks reached. Triggering workflow."
                )

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
                    "run_id": current_run_id,  # Include run_id in the state
                    "current_nav": 100000.0,
                    "daily_pnl": 0.0,
                    "risk_gate_decision": None,
                }

                logger.info(
                    f"Invoking workflow for run_id: {current_run_id} with {len(TICK_BUFFER)} ticks. Using placeholder NAV={initial_state['current_nav']}, PNL={initial_state['daily_pnl']}."
                )
                # Asynchronously invoke the graph. Does not block further ticks if graph runs long.
                # Be mindful of resource limits if many graph instances run concurrently.
                asyncio.create_task(process_workflow_run(graph_app, initial_state))

                TICK_BUFFER = []  # Clear buffer after triggering
                logger.info("Tick buffer cleared.")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse tick data as JSON: {message_data}")
        except Exception as e:
            logger.error(f"Error in tick_handler: {e}")

    await event_bus.subscribe(market_channel, tick_handler)

    # Keep the listener running indefinitely (or until an external signal)
    try:
        while True:
            await asyncio.sleep(
                1
            )  # Keep alive, actual work happens in tick_handler via EventBus
    except KeyboardInterrupt:
        logger.info("Market tick listener stopped by user.")
    except Exception as e:
        logger.error(f"Market tick listener faced an unexpected error: {e}")
    finally:
        logger.info("Closing EventBus connection for market tick listener.")
        await event_bus.close()


async def process_workflow_run(graph_app: StateGraph, initial_state: WorkflowState):
    """Process a single workflow run, including logging its final state."""
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
            logger.error(
                f"Failed to parse final_output for DB logging (run_id {run_id}): {final_output_str}"
            )
            final_output_dict_for_db = {
                "raw_output": final_output_str,
                "parsing_error": "Could not decode JSON",
            }

    # Ensure run_id from the workflow is part of the log entry if not already in final_output_dict_for_db
    if final_output_dict_for_db and "run_id" not in final_output_dict_for_db:
        final_output_dict_for_db["run_id"] = run_id
    elif (
        not final_output_dict_for_db
    ):  # Handle cases where final_output_str might be None or unparsable
        final_output_dict_for_db = {
            "run_id": run_id,
            "status_details": "No parsable final output string.",
        }

    # The input_query for the log should be the stringified tick data.
    input_query_for_log = initial_state["query"]

    run_log_entry = OrchestratorRunSchema(
        input_query=input_query_for_log,
        final_output=final_output_dict_for_db,
        status=status,
        error_message=final_state_dict.get("error"),
        run_id=run_id,
    )
    try:
        log_run(run_log_entry)
        logger.info(f"Successfully logged workflow run {run_id} to LanceDB.")
    except Exception as e:
        logger.error(f"Failed to log workflow run {run_id} to LanceDB: {e}")

    # Output to console (optional, can be noisy in continuous mode)
    if final_state_dict.get("error"):
        logger.error(
            f"Workflow run {run_id} completed with errors: {final_state_dict['error']}"
        )

    # Printing final_output_str can be very verbose if running continuously.
    # Consider logging to a file or conditional printing based on verbosity settings.
    # For now, let's log it.
    logger.info(
        f"Console output for run {run_id}: {final_output_str if final_output_str else 'No final_output string.'}"
    )


# --- CLI ---
async def main_async(args):
    init_otel()  # Initialize OpenTelemetry
    # Build the graph application once
    graph_app = build_graph()

    logger.info(
        f"Initializing LanceDB 'advice' table at {RISK_GATE_CONFIG['lancedb_path']}..."
    )
    try:
        init_advice_table(db_path_str=RISK_GATE_CONFIG["lancedb_path"])
        logger.info("'advice' table initialized successfully.")
    except Exception as e:
        logger.critical(
            f"CRITICAL: Failed to initialize 'advice' table: {e}. The application may not function correctly. Exiting.",
            exc_info=True,
        )
        return

    logger.info(f"Starting Osiris Policy Orchestrator in event-driven mode.")
    logger.info(
        f"Listening to Redis channel '{args.market_channel}' on {args.redis_url}"
    )
    logger.info(f"Triggering proposal workflow every {args.ticks_per_proposal} ticks.")

    # Start the market tick listener
    # This function will run indefinitely until stopped (e.g., KeyboardInterrupt)
    await market_tick_listener(
        redis_url=args.redis_url,
        market_channel=args.market_channel,
        ticks_per_proposal=args.ticks_per_proposal,
        graph_app=graph_app,
    )


def run_orchestrator() -> None:
    """Entry point for running the orchestrator from the CLI."""
    parser = argparse.ArgumentParser(
        description="Osiris Policy Orchestrator - Event-Driven Mode"
    )
    parser.add_argument(
        "--redis_url",
        type=str,
        default="redis://localhost:6379/0",
        help="Redis URL for market data and event bus.",
    )
    parser.add_argument(
        "--market_channel",
        type=str,
        default=MARKET_TICKS_CHANNEL,
        help="Redis channel to listen for market ticks.",
    )
    parser.add_argument(
        "--ticks_per_proposal",
        type=int,
        default=10,
        help="Number of market ticks to buffer before triggering a new proposal workflow.",
    )

    args = parser.parse_args()
    with tracer.start_as_current_span("orchestrator.run"):
        asyncio.run(main_async(args))


if __name__ == "__main__":
    run_orchestrator()
