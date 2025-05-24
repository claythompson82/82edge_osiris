# LLM Sidecar with Hermes-3 8B GPTQ

## Project Overview

This project provides a Dockerized sidecar service for running a Large Language Model (LLM), specifically the [Hermes-Trismegistus-III-8B-GPTQ](https://huggingface.co/TheBloke/Hermes-Trismegistus-III-8B-GPTQ) model. It exposes an API for text generation and includes a VRAM watchdog to manage GPU memory usage.

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
    -   Build the `llm-sidecar` Docker image if it doesn't exist locally (this includes downloading the Hermes-3 8B GPTQ model).
    -   Start the `llm-sidecar` service in detached mode (`-d`).

    The `llm-sidecar` service runs a FastAPI application that provides an API for interacting with the model. By default, it's accessible at `http://localhost:8000`.

## Expected VRAM Usage

The Hermes-3 8B GPTQ model (8-bit quantization) typically requires around **6-7 GB of VRAM** for operation. The system is configured with a watchdog that monitors VRAM usage to prevent out-of-memory errors.

## VRAM Watchdog

This project includes a `vram_watchdog.sh` script that monitors the GPU VRAM usage to ensure stability.
-   **Purpose:** The watchdog prevents the LLM service from crashing the system due to excessive VRAM consumption.
-   **Behavior:**
    -   It checks the GPU VRAM usage every 60 seconds.
    -   If VRAM usage exceeds **10.5 GB** for **3 consecutive checks**, the watchdog will automatically restart the `llm-sidecar` service.
-   **Service:** The watchdog runs as a separate service named `vram-watchdog`, defined in the `docker/compose.yaml` file. This service requires access to the Docker socket to manage the `llm-sidecar` container and uses `docker exec` to run `nvidia-smi` within the `llm-sidecar` container for VRAM checks.

## API Endpoints

The `llm-sidecar` service exposes the following API endpoints, implemented in `server.py`:

*   **POST `/generate/`**
    *   **Description:** Accepts a prompt and generates text using the Hermes-3 model.
    *   **Request Body (JSON):**
        ```json
        {
          "prompt": "Your text prompt here",
          "max_length": 128 // Optional, defaults to 128
        }
        ```
    *   **Response (JSON):**
        ```json
        {
          "generated_text": "The model's response..."
        }
        ```

*   **GET `/health`**
    *   **Description:** Provides a health check for the service. It indicates if the model is loaded and on which device it's running.
    *   **Response (JSON on success):**
        ```json
        {
          "status": "ok",
          "model_loaded": true,
          "device": "cuda:0" // Or "cpu" if CUDA is not available
        }
        ```
    *   **Response (JSON on failure/model not loaded):**
        ```json
        {
          "status": "error",
          "model_loaded": false,
          "message": "Model or tokenizer not loaded."
        }
        ```

## CI Pipeline

The repository includes a GitHub Actions CI pipeline defined in `.github/workflows/ci.yaml`. This pipeline automatically:
- Builds the `llm-sidecar` Docker image.
- Runs the container.
- Performs a health check against the `/health` endpoint.
- This ensures basic functionality and integration on pushes and pull requests to the `main` branch.
