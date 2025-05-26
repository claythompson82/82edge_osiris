# LLM Sidecar with Hermes-3 8B GPTQ

## Project Overview

This project provides a Dockerized sidecar service for running a Large Language Model (LLM), specifically the [Hermes-Trismegistus-III-8B-GPTQ](https://huggingface.co/TheBloke/Hermes-Trismegistus-III-8B-GPTQ) model. It exposes an API for text generation and includes a VRAM watchdog to manage GPU memory usage. Recently, functionality for a Phi-3 based JSON generation with a feedback loop has been added.

## Prerequisites

To run this project, you will need the following installed on your system:
- **Docker:** For running containerized applications.
- **Docker Compose:** For defining and running multi-container Docker applications.
- **NVIDIA Drivers with CUDA Support:** Essential for GPU acceleration of the LLM. Ensure your drivers are up-to-date and compatible with the CUDA version used in the Docker image (see `docker/Dockerfile`).

## Setup and Installation (Hermes-3)

1.  **Environment Configuration:**
    This project uses an environment file to manage settings.
    -   Copy the template file: `cp .env.template .env`
    -   Review the `.env` file. It includes `HERMES_CTX=6144`, which sets the context window size for the Hermes model. Adjust if necessary, though 6144 is a common value for this model variant.

2.  **Running the LLM Sidecar:**
    To build and run the LLM sidecar service, use the following Docker Compose command:
    ```bash
    docker compose -f docker/compose.yaml up -d llm-sidecar
    ```
    This command will:
    -   Build the `llm-sidecar` Docker image if it doesn't exist locally (this includes downloading the Hermes-3 8B GPTQ model and the Phi-3 model).
    -   Start the `llm-sidecar` service in detached mode (`-d`).

    The `llm-sidecar` service runs a FastAPI application that provides an API for interacting with the model. By default, it's accessible at `http://localhost:8000`.

## Expected VRAM Usage

The Hermes-3 8B GPTQ model (8-bit quantization) typically requires around **6-7 GB of VRAM** for operation. The Phi-3 model adds to this, but is smaller. The system is configured with a watchdog that monitors VRAM usage to prevent out-of-memory errors.

## VRAM Watchdog

This project includes a `vram_watchdog.sh` script that monitors the GPU VRAM usage to ensure stability.
-   **Purpose:** The watchdog prevents the LLM service from crashing the system due to excessive VRAM consumption.
-   **Behavior:**
    -   It checks the GPU VRAM usage every 60 seconds.
    -   If VRAM usage exceeds **10.5 GB** for **3 consecutive checks**, the watchdog will automatically restart the `llm-sidecar` service.
-   **Service:** The watchdog runs as a separate service named `vram-watchdog`, defined in the `docker/compose.yaml` file. This service requires access to the Docker socket to manage the `llm-sidecar` container and uses `docker exec` to run `nvidia-smi` within the `llm-sidecar` container for VRAM checks.

## API Endpoints

The `llm-sidecar` service exposes the following API endpoints, implemented in `server.py`:

*   **POST `/generate/`** (Unified Endpoint)
    *   **Description:** Accepts a prompt and an optional `model_id` to generate text using either Hermes or Phi-3. Defaults to Hermes if `model_id` is not specified. Phi-3 generation will use the predefined JSON schema for structured output and may incorporate feedback from the Phi-3 feedback loop.
    *   **Request Body (JSON):**
        ```json
        {
          "prompt": "Your text prompt here",
          "max_length": 256, // Optional, defaults to 256
          "model_id": "hermes" // Optional, "hermes" or "phi3". Defaults to "hermes".
        }
        ```
    *   **Response (JSON for Hermes - success):**
        ```json
        {
          "generated_text": "The model's response..."
        }
        ```
    *   **Response (JSON for Phi-3 - success, example assumes the schema output):**
        ```json
        {
          // Example output based on the schema in server.py
          "ticker": "XYZ",
          "action": "adjust",
          "side": "LONG",
          "new_stop_pct": 5.0,
          "new_target_pct": 10.0,
          "confidence": 0.85,
          "rationale": "Price action suggests upward trend."
        }
        ```
    *   **Response (JSON for error - e.g., invalid model_id):**
        ```json
        {
          "error": "Invalid model_id specified. Choose 'hermes' or 'phi3'.",
          "specified_model_id": "your_invalid_id"
        }
        ```
    *   **Response (JSON for error - e.g., model not loaded):**
        ```json
        {
          "error": "Hermes model not loaded. Please check server logs." 
          // Or: "Phi-3 ONNX model not loaded. Please check server logs."
        }
        ```

    *Note: Model-specific endpoints (`/generate/hermes/` and `/generate/phi3/`) are also available for direct access if you prefer to target a specific model directly.*

*   **POST `/generate/hermes/`**
    *   **Description:** Accepts a prompt and generates text using the Hermes-3 model. This is a dedicated endpoint for the Hermes model.
    *   **Request Body (JSON):**
        ```json
        {
          "prompt": "Your text prompt here",
          "max_length": 256 // Optional, defaults to 256
        }
        ```
    *   **Response (JSON):**
        ```json
        {
          "generated_text": "The model's response..."
        }
        ```
    *   **Response (JSON for error - e.g., model not loaded):**
        ```json
        {
          "error": "Hermes model not loaded. Please check server logs."
        }
        ```

*   **POST `/generate/phi3/`**
    *   **Description:** Accepts a prompt and generates a structured JSON output using the Phi-3 model based on a predefined schema. This endpoint also benefits from the Phi-3 feedback loop.
    *   **Request Body (JSON):**
        ```json
        {
          "prompt": "Your text prompt here",
          "max_length": 256 // Optional, defaults to 256
        }
        ```
    *   **Response (JSON - success, example assumes the schema output):**
        ```json
        {
          // Example output based on the schema in server.py
          "ticker": "XYZ",
          "action": "adjust",
          // ... other fields as per schema
          "rationale": "Rationale for the action."
        }
        ```
    *   **Response (JSON for error - e.g., model not loaded):**
        ```json
        {
          "error": "Phi-3 ONNX model not loaded. Please check server logs."
        }
        ```

*   **POST `/propose_trade_adjustments/`**
    *   **Description:** A specialized endpoint that first uses Phi-3 to generate a structured trade proposal (using the same schema as `/generate/phi3/`) and then uses Hermes to provide a textual assessment of that proposal. Outputs from this endpoint are logged as part of the Phi-3 feedback loop.
    *   **Request Body (JSON):**
        ```json
        {
          "prompt": "Your prompt for generating a trade adjustment proposal",
          "max_length": 256 // Optional, for Phi-3 generation part
        }
        ```
    *   **Response (JSON - success):**
        ```json
        {
          "phi3_proposal": {
            "ticker": "XYZ",
            "action": "adjust",
            // ... other fields from Phi-3 schema
            "rationale": "Phi-3's rationale."
          },
          "hermes_assessment": "Hermes' textual assessment of the proposal."
        }
        ```
    *   **Response (JSON - error, e.g., if a model fails):**
        ```json
        {
          "error": "Phi-3 failed to generate proposal.", // or Hermes related error
          "phi3_details": { /* ... if applicable ... */ } 
        }
        ```

*   **POST `/feedback/phi3/`**
    *   **Description:** Used to submit feedback on Phi-3's proposals, especially corrections or ratings. This feedback is stored and used to improve future Phi-3 outputs.
    *   **Request Body (JSON):**
        ```json
        {
          "transaction_id": "some-uuid-linking-to-original-proposal",
          "feedback_type": "correction", // e.g., "correction", "rating", "qualitative_comment"
          "feedback_content": {"comment": "The rationale needs to be more specific."}, // or a rating, or a full corrected object
          "timestamp": "2023-10-27T10:30:00Z", // Optional: server will set if not provided
          "corrected_proposal": { // Optional: provide if feedback_type is "correction"
            "ticker": "XYZ",
            "action": "adjust",
            "side": "LONG",
            "new_stop_pct": 5.5,
            "new_target_pct": 10.5,
            "confidence": 0.90,
            "rationale": "Adjusted stop loss due to recent market volatility and confirmed support level."
          }
        }
        ```
    *   **Response (JSON - success):**
        ```json
        {
          "message": "Feedback received successfully",
          "transaction_id": "some-uuid-linking-to-original-proposal"
        }
        ```
    *   **Response (JSON - error):**
        ```json
        {
          "detail": "Failed to store feedback. Error: <error_details>" // If using HTTPException for that specific error
        }
        ```

*   **GET `/health`**
    *   **Description:** Provides a health check for the service. It indicates if the models are loaded, if the Phi-3 model file exists, and on which device the service is running.
    *   **Response (JSON on success):**
        ```json
        {
          "status": "ok", // or "partial_error" if one model failed, or "error" if both failed
          "hermes_loaded": true, // boolean
          "phi3_loaded": true, // boolean
          "phi3_model_file_exists": true, //boolean
          "device": "cuda" // Or "cpu"
        }
        ```
    *   **Response (JSON on failure/model not loaded):** 
        This structure is now more detailed, see above. An "error" status implies both models failed to load.
        "partial_error" implies one of them failed.

## Phi-3 Feedback Loop

This system implements a feedback loop to improve the performance of the Phi-3 model over time. The loop consists of three main components: logging, feedback submission, and prompt augmentation.

### Logging of Proposals

When the `/propose_trade_adjustments/` endpoint is utilized:
-   The initial JSON proposal generated by Phi-3 and the subsequent textual assessment provided by Hermes are logged.
-   This log is stored in a JSONL file at `/app/phi3_feedback_log.jsonl` within the container.
-   Each log entry includes:
    -   `transaction_id`: A unique identifier for the proposal-assessment pair.
    -   `timestamp`: The UTC timestamp when the log entry was created.
    -   `phi3_proposal`: The full JSON object generated by Phi-3.
    -   `hermes_assessment`: The textual assessment from Hermes.

Example log entry snippet:
```json
{
  "transaction_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "timestamp": "YYYY-MM-DDTHH:MM:SS.ffffff",
  "phi3_proposal": {"ticker": "XYZ", "action": "adjust", ...},
  "hermes_assessment": "This proposal looks reasonable..."
}
```

### Feedback Submission and Storage

User feedback on Phi-3's outputs can be submitted via the `POST /feedback/phi3/` endpoint (see API Endpoints section for details).
-   This feedback is crucial for identifying areas where Phi-3 can improve.
-   Submitted feedback items are stored in a separate JSONL file at `/app/phi3_feedback_data.jsonl`.
-   Each entry in this file corresponds to a `FeedbackItem` Pydantic model, including the original `transaction_id`, `feedback_type`, `feedback_content`, and potentially a `corrected_proposal`.

### Prompt Augmentation

Collected feedback is used to refine the prompts sent to Phi-3, particularly for structured JSON generation:
-   When generating new proposals (e.g., via `/propose_trade_adjustments/` or `/generate/phi3/`), the system loads recent feedback items.
-   Specifically, it looks for entries where `feedback_type` is `"correction"` and a `corrected_proposal` (the corrected JSON object) is provided.
-   A few of the most recent, relevant corrected examples are formatted and prepended to the original user prompt for Phi-3.
-   This mechanism guides Phi-3 by showing it concrete examples of desired outputs based on past corrections, aiming to improve its accuracy, adherence to the schema, and overall quality of its JSON proposals over time.

## CI Pipeline

The repository includes a GitHub Actions CI pipeline defined in `.github/workflows/ci.yaml`. This pipeline automatically:
- Builds the `llm-sidecar` Docker image.
- Runs the container.
- Performs a health check against the `/health` endpoint.
- This ensures basic functionality and integration on pushes and pull requests to the `main` branch.
