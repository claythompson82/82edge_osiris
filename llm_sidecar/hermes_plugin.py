import torch
from llm_sidecar.loader import get_hermes_model_and_tokenizer
from typing import Optional, Dict, Any
import re  # For parsing the score

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def score_with_hermes(
    proposal_dict: Dict[str, Any], context: Optional[str] = None
) -> float:
    """
    Scores a given proposal using the Hermes model.

    Args:
        proposal_dict: The proposal to score, as a dictionary.
        context: Optional additional context for scoring.

    Returns:
        A float score between 0.0 and 1.0, or -1.0 if scoring fails.
    """
    model, tokenizer = get_hermes_model_and_tokenizer()

    if not model or not tokenizer:
        print("Hermes model or tokenizer not available. Cannot score.")
        return -1.0  # Indicate scoring failure

    # Construct the prompt
    # Base prompt uses the proposal dictionary
    prompt_parts = [
        "Rate the above trade idea from 0-to-10 where 10 = excellent risk-reward … Reply with just the number.",
        "Trade Idea:",
        str(proposal_dict),  # Convert dict to string for the prompt
    ]

    # Add context if provided
    if context:
        prompt_parts.insert(
            1, f"Context: {context}"
        )  # Insert context before the trade idea

    # The critique instruction from docs/quality_loop.md is:
    # "Rate the above trade idea from 0-to-10 where 10 = excellent risk-reward … Reply with just the number."
    # The model should see the trade idea *then* the instruction.
    # So, the instruction should come after the proposal.
    # Let's re-order:
    # 1. Context (optional)
    # 2. Trade Idea
    # 3. Instruction for rating

    prompt_text = f"Trade Proposal:\n{proposal_dict}"
    if context:
        prompt_text = f"Context:\n{context}\n\n{prompt_text}"

    # Final prompt structure
    final_prompt = (
        f"{prompt_text}\n\n"
        "Rate the above trade idea from 0-to-10 where 10 = excellent risk-reward … Reply with just the number."
    )

    try:
        inputs = tokenizer(
            final_prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(
            DEVICE
        )  # Added truncation

        # Generate response
        # Using similar parameters as _generate_hermes_text in server.py
        # eos_token_id is important for stopping generation after the number.
        output_sequences = model.generate(
            **inputs,
            max_new_tokens=10,  # Score should be short
            do_sample=False,  # We want a deterministic score
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,  # common practice for models without a pad token
        )

        # Decode the response
        # We only want the generated part, not the input prompt
        response_text = tokenizer.decode(
            output_sequences[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()

        # Extract the score (should be just a number)
        # Use regex to find the first number (integer or float) in the response
        match = re.search(r"(\d+(\.\d+)?)", response_text)
        if match:
            raw_score_str = match.group(1)
            raw_score = float(raw_score_str)
            if 0 <= raw_score <= 10:
                normalized_score = raw_score / 10.0
                return normalized_score
            else:
                print(f"Raw score {raw_score} is outside the 0-10 range.")
                return -1.0  # Indicate scoring failure (invalid score range)
        else:
            print(f"Could not parse score from Hermes response: '{response_text}'")
            return -1.0  # Indicate scoring failure (parsing failed)

    except Exception as e:
        print(f"Error during Hermes model inference or score processing: {e}")
        return -1.0  # Indicate scoring failure
