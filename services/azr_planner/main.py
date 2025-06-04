from fastapi import FastAPI
import os # For path manipulation to read the file

app = FastAPI(title="AZR Planner Service")

@app.get("/plan")
async def get_plan():
    """
    Implements alpha/beta resource check and curriculum step.
    """
    patch_content = ""
    priority = ""

    # Alpha resource is always available
    alpha_available = True
    # Beta resource is never available (not explicitly used in if/else based on alpha)

    if alpha_available:
        try:
            # Construct path relative to this file's location
            # __file__ is services/azr_planner/main.py
            # We want to reach <repo_root>/resources/dummy_patch.py.txt
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            # Path from /app/services/azr_planner/main.py to /app/resources/dummy_patch.py.txt
            file_path = os.path.join(current_script_dir, "..", "..", "resources", "dummy_patch.py.txt")

            # Normalize path (e.g., resolves '..')
            normalized_file_path = os.path.normpath(file_path)

            with open(normalized_file_path, "r") as f:
                patch_content = f.read()
        except FileNotFoundError:
            patch_content = "File not found"
        priority = "high"
    else:
        # This case will not be reached in the current logic as alpha_available is True.
        patch_content = "NOP"
        priority = "low"

    return {"patch": patch_content, "priority": priority}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# To run directly for local testing: uvicorn services.azr_planner.main:app --reload --port 8001
