import os
import torch
from fastapi import FastAPI, HTTPException # Added HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any # Added for FeedbackItem
import json # Ensure json is imported
import datetime
import uuid

from llm_sidecar.loader import (
    load_hermes_model,
    load_phi3_model,
    get_hermes_model_and_tokenizer,
    get_phi3_model_and_tokenizer,
    MICRO_LLM_MODEL_PATH
)

# Import outlines for guided generation
from outlines import generate as outlines_generate
import traceback # For detailed error logging

# Define the FastAPI app
app = FastAPI()

# Log file path
PHI3_FEEDBACK_LOG_FILE = "/app/phi3_feedback_log.jsonl"
PHI3_FEEDBACK_DATA_FILE = "/app/phi3_feedback_data.jsonl" # For storing submitted feedback

# Define the request body model
class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 256

# Define the new request body model for the unified endpoint
class UnifiedPromptRequest(PromptRequest):
    model_id: str = "hermes"

# --- Feedback Model ---
class FeedbackItem(BaseModel):
    transaction_id: str
    feedback_type: str  # E.g., "correction", "rating", "qualitative_comment"
    feedback_content: Any  # Flexible: could be a dict for corrections, str for comments/ratings
    timestamp: str       # ISO format timestamp, likely generated at feedback submission
    corrected_proposal: Optional[Dict[str, Any]] = None # For providing a full corrected version of a proposal

# JSON Schema for guided generation (remains global)
json_schema_str = """
{
  "type": "object",
  "properties": {
    "ticker":        { "type": "string", "description": "Ticker symbol" },
    "action":        { "type": "string", "enum": ["adjust", "pass", "abort"] },
    "side":          { "type": "string", "enum": ["LONG", "SHORT"] },
    "new_stop_pct":  { "type": ["number", "null"] },
    "new_target_pct":{ "type": ["number", "null"] },
    "confidence":    { "type": "number", "minimum": 0, "maximum": 1 },
    "rationale":     { "type": "string", "description": "One-sentence justification" }
  },
  "required": ["ticker", "action", "confidence", "rationale"],
  "additionalProperties": false
}
"""

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Server running on device: {device}")

# Load models at startup
print("Initializing models...")
load_hermes_model()
load_phi3_model()
print("Model initialization complete.")

# --- Helper Function to Load Recent Feedback ---
def load_recent_feedback(max_examples: int = 3) -> list[dict]:
    if not os.path.exists(PHI3_FEEDBACK_DATA_FILE):
        print(f"Feedback file {PHI3_FEEDBACK_DATA_FILE} not found. No feedback to load.")
        return []

    recent_feedback_items = []
    try:
        with open(PHI3_FEEDBACK_DATA_FILE, "r") as f:
            for line in f:
                try:
                    feedback_item = json.loads(line.strip())
                    if (feedback_item.get("feedback_type") == "correction" and
                            isinstance(feedback_item.get("corrected_proposal"), dict)):
                        recent_feedback_items.append(feedback_item)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from feedback file: {e} - Line: {line.strip()}")
                except Exception as e: # Catch other potential errors like missing keys if not validated by Pydantic model previously
                    print(f"Error processing feedback item: {e} - Item: {line.strip()}")
        
        # Return the most recent max_examples
        return recent_feedback_items[-max_examples:]
    except IOError as e:
        print(f"Error reading feedback file {PHI3_FEEDBACK_DATA_FILE}: {e}")
        return []
    except Exception as e: # Catch any other unexpected errors during file processing
        print(f"An unexpected error occurred while loading feedback: {e}\n{traceback.format_exc()}")
        return []

# --- Helper Function for Phi-3 JSON Generation ---
async def _generate_phi3_json(prompt: str, max_length: int, phi3_model, phi3_tokenizer) -> dict:
    if not phi3_model or not phi3_tokenizer:
        return {"error": "Phi-3 ONNX model or tokenizer not available."}
    try:
        if phi3_model is None or phi3_tokenizer is None: # Should be caught by above, but defensive
            raise RuntimeError("Phi-3 model or tokenizer is None after loading check.")

        # Load recent feedback and augment prompt
        feedback_prompt_augmentation = ""
        try:
            recent_feedback = load_recent_feedback(max_examples=3)
            if recent_feedback:
                augmentation_parts = ["Based on past feedback, here are examples of desired JSON outputs:"]
                for i, fb_item in enumerate(recent_feedback):
                    # corrected_proposal is already validated to be a dict in load_recent_feedback
                    try:
                        corrected_json_str = json.dumps(fb_item["corrected_proposal"], indent=2)
                        augmentation_parts.append(f"Example {i+1} (Corrected Output):\n{corrected_json_str}")
                    except TypeError as e: # Should not happen if corrected_proposal is a dict, but defensive
                        print(f"Error serializing corrected_proposal for feedback item: {fb_item.get('transaction_id')}, error: {e}")
                
                if len(augmentation_parts) > 1: # Only add if there are actual examples
                    feedback_prompt_augmentation = "\n\n".join(augmentation_parts) + "\n\nNow, considering the above, please process the following request:\n"
        except Exception as e:
            print(f"Error during feedback loading or prompt augmentation: {e}\n{traceback.format_exc()}")
            # Proceed with original prompt if feedback processing fails

        effective_prompt = feedback_prompt_augmentation + prompt
        
        # For debugging, print the effective prompt
        # print(f"--- Effective Prompt for Phi-3 ---")
        # print(effective_prompt)
        # print(f"--- End of Effective Prompt ---")

        print(f"Generating JSON with Phi-3. Original prompt: '{prompt}'. Schema guided generation.")
        if feedback_prompt_augmentation:
            print(f"Prompt augmented with {len(recent_feedback)} feedback example(s).")

        json_generator = outlines_generate.json(phi3_model, json_schema_str, tokenizer=phi3_tokenizer)
        generated_json_obj = json_generator(effective_prompt, max_tokens=max_length)
        
        print(f"Successfully generated JSON object: {generated_json_obj}")
        return generated_json_obj
    except Exception as e:
        error_message = f"Failed to generate JSON with Phi-3 ONNX using outlines: {str(e)}"
        print(f"Error during Phi-3 JSON generation: {e}\n{traceback.format_exc()}")
        if "ORTModelForCausalLM" in str(e) or "outlines" in str(e).lower():
             error_message = f"Compatibility issue with Outlines and ONNX model or runtime error in Outlines: {str(e)}"
        return {"error": error_message, "details": traceback.format_exc()}

# --- Helper Function for Hermes Text Generation ---
async def _generate_hermes_text(prompt: str, max_length: int, hermes_model, hermes_tokenizer) -> str:
    if not hermes_model or not hermes_tokenizer:
        # This case will be handled by the calling endpoint, but good to have a direct string return for error
        return "Error: Hermes model or tokenizer not available." 
    try:
        inputs = hermes_tokenizer(prompt, return_tensors="pt").to(device)
        output_sequences = hermes_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
        )
        generated_text = hermes_tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print(f"Error during Hermes text generation: {e}\n{traceback.format_exc()}")
        return f"Error generating text with Hermes: {str(e)}"


# --- Updated Endpoints ---
@app.post("/generate/hermes/")
async def generate_text_hermes_endpoint(request: PromptRequest):
    hermes_model, hermes_tokenizer = get_hermes_model_and_tokenizer()
    if not hermes_model or not hermes_tokenizer:
        return {"error": "Hermes model not loaded. Please check server logs."}
    
    generated_text = await _generate_hermes_text(request.prompt, request.max_length, hermes_model, hermes_tokenizer)
    if generated_text.startswith("Error:"): # Check if helper returned an error string
        return {"error": generated_text}
    return {"generated_text": generated_text}

@app.post("/generate/phi3/")
async def generate_text_phi3_json_endpoint(request: PromptRequest):
    phi3_model, phi3_tokenizer = get_phi3_model_and_tokenizer()
    if not phi3_model or not phi3_tokenizer:
        return {"error": "Phi-3 ONNX model not loaded. Please check server logs."}
        
    json_result = await _generate_phi3_json(request.prompt, request.max_length, phi3_model, phi3_tokenizer)
    return json_result # Returns the dict directly (either JSON or error dict)

# --- New Endpoint for Trade Adjustments ---
@app.post("/propose_trade_adjustments/")
async def propose_trade_adjustments(request: PromptRequest):
    phi3_model, phi3_tokenizer = get_phi3_model_and_tokenizer()
    hermes_model, hermes_tokenizer = get_hermes_model_and_tokenizer()

    if not phi3_model or not phi3_tokenizer:
        return {"error": "Phi-3 model/tokenizer not loaded. Cannot generate proposal."}
    if not hermes_model or not hermes_tokenizer:
        return {"error": "Hermes model/tokenizer not loaded. Cannot generate assessment."}

    # Step 1: Call Phi-3 for JSON proposal
    print(f"Calling Phi-3 for initial proposal with prompt: {request.prompt}")
    phi3_json_proposal = await _generate_phi3_json(request.prompt, request.max_length, phi3_model, phi3_tokenizer)

    if isinstance(phi3_json_proposal, dict) and "error" in phi3_json_proposal:
        print(f"Phi-3 failed to generate proposal: {phi3_json_proposal['error']}")
        return {"error": "Phi-3 failed to generate proposal.", "phi3_details": phi3_json_proposal}

    # Step 2: Formulate prompt for Hermes
    try:
        json_str_proposal = json.dumps(phi3_json_proposal, indent=2)
    except Exception as e:
        print(f"Error serializing Phi-3 proposal to JSON string: {e}")
        return {"error": "Failed to serialize Phi-3 proposal.", "details": str(e)}
        
    hermes_prompt = (
        "The following trade adjustment was proposed by an AI assistant. "
        "Please review this proposal and provide your assessment or refinement. "
        "Consider the rationale and confidence score. Be concise and clear.\n\n"
        "Proposed Adjustment:\n"
        f"{json_str_proposal}\n\n"
        "Your assessment (provide a brief analysis and recommendation):"
    )
    print(f"Calling Hermes for assessment with prompt: {hermes_prompt}")

    # Step 3: Call Hermes for assessment
    # Using a potentially longer max_length for Hermes assessment if needed, but reusing for now.
    hermes_assessment_max_length = request.max_length * 2 # Example: allow more tokens for assessment
    hermes_assessment = await _generate_hermes_text(hermes_prompt, hermes_assessment_max_length, hermes_model, hermes_tokenizer)

    if hermes_assessment.startswith("Error:"): # Check if helper returned an error string
         return {"error": "Hermes failed to generate assessment.", 
                 "phi3_proposal": phi3_json_proposal, 
                 "hermes_details": hermes_assessment}

    # Step 4: Log the data
    try:
        transaction_id = str(uuid.uuid4())
        timestamp = datetime.datetime.utcnow().isoformat()
        log_entry = {
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "phi3_proposal": phi3_json_proposal,
            "hermes_assessment": hermes_assessment
        }
        with open(PHI3_FEEDBACK_LOG_FILE, "a") as f:
            json.dump(log_entry, f)
            f.write('\n')
    except IOError as e:
        print(f"Error writing to log file {PHI3_FEEDBACK_LOG_FILE}: {e}")
        # Not returning an error to the client, just logging the issue.

    # Step 5: Return both Phi-3 proposal and Hermes's assessment
    return {"phi3_proposal": phi3_json_proposal, "hermes_assessment": hermes_assessment}


# --- Endpoint for Submitting Feedback ---
@app.post("/feedback/phi3/")
async def submit_phi3_feedback(feedback: FeedbackItem):
    feedback.timestamp = datetime.datetime.utcnow().isoformat()
    feedback_dict = feedback.model_dump()

    try:
        with open(PHI3_FEEDBACK_DATA_FILE, "a") as f:
            json.dump(feedback_dict, f)
            f.write('\n')
        return {"message": "Feedback received successfully", "transaction_id": feedback.transaction_id}
    except IOError as e:
        print(f"Error writing feedback to {PHI3_FEEDBACK_DATA_FILE}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to store feedback. Error: {str(e)}")


# --- New Unified Generate Endpoint ---
@app.post("/generate/")
async def generate_unified(request: UnifiedPromptRequest):
    if request.model_id == "phi3":
        phi3_model, phi3_tokenizer = get_phi3_model_and_tokenizer()
        if not phi3_model or not phi3_tokenizer:
            return {"error": "Phi-3 ONNX model not loaded. Please check server logs."}
        # Call the existing helper function for Phi-3
        json_result = await _generate_phi3_json(request.prompt, request.max_length, phi3_model, phi3_tokenizer)
        return json_result
    elif request.model_id == "hermes":
        hermes_model, hermes_tokenizer = get_hermes_model_and_tokenizer()
        if not hermes_model or not hermes_tokenizer:
            return {"error": "Hermes model not loaded. Please check server logs."}
        # Call the existing helper function for Hermes
        generated_text = await _generate_hermes_text(request.prompt, request.max_length, hermes_model, hermes_tokenizer)
        if generated_text.startswith("Error:"):
            return {"error": generated_text}
        return {"generated_text": generated_text}
    else:
        # Handle invalid model_id
        return {"error": "Invalid model_id specified. Choose 'hermes' or 'phi3'.", "specified_model_id": request.model_id}

@app.get("/health")
async def health_check():
    hermes_model, hermes_tokenizer = get_hermes_model_and_tokenizer()
    phi3_model, phi3_tokenizer = get_phi3_model_and_tokenizer()

    hermes_status = hermes_model is not None and hermes_tokenizer is not None
    phi3_status = phi3_model is not None and phi3_tokenizer is not None
    
    phi3_file_exists = os.path.exists(MICRO_LLM_MODEL_PATH)

    overall_status = "ok"
    if not hermes_status and not phi3_status:
        overall_status = "error"
    elif not hermes_status or not phi3_status:
        overall_status = "partial_error"

    return {
        "status": overall_status,
        "hermes_loaded": hermes_status,
        "phi3_loaded": phi3_status,
        "phi3_model_file_exists": phi3_file_exists,
        "device": device
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Attempting to run server locally, ensuring MICRO_LLM_MODEL_PATH ({MICRO_LLM_MODEL_PATH}) is correct.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
