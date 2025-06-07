from fastapi.testclient import TestClient

# Import the FastAPI application from the azr_planner service. The
# azr_planner code lives under the `services` package in this repository,
# so the correct import path includes that prefix.
from services.azr_planner.main import app

client = TestClient(app)


def test_plan_alpha_resource_available():
    """
    Tests the /plan endpoint when the alpha resource is considered available.
    It expects the content from resources/dummy_patch.py.txt.
    """
    # Expected content from resources/dummy_patch.py.txt
    # Based on previous read_files, it is 'print("Hello from dummy patch")\n'
    expected_patch_content = 'print("Hello from dummy patch")\n'

    response = client.get("/plan")

    assert response.status_code == 200
    assert response.json() == {"patch": expected_patch_content, "priority": "high"}


# Note on "Alpha Resource Unavailable" scenario:
# The current implementation of services.azr_planner.main.get_plan
# hardcodes alpha_available = True. To test the 'unavailable' case,
# main.py would need to be refactored to allow dynamic control over
# resource availability (e.g., via dependency injection or environment variables).
# Once that's done, further test cases could be added here.


def test_health_check():
    """
    Tests the /health endpoint.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
