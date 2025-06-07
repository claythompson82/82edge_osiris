from fastapi.testclient import TestClient
# The 'azr_planner' logic appears to be within the 'advisor' module.
# This import path reflects the likely correct project structure.
from advisor.main import app as planner_app


def test_plan_alpha_resource_available():
    """
    Tests if the /plan/alpha_resource endpoint is available and returns a successful response.
    """
    client = TestClient(planner_app)
    response = client.get("/plan/alpha_resource")
    assert response.status_code == 200
    assert "task_id" in response.json()


def test_status_endpoint_for_task():
    """
    Tests creating a task and then checking its status.
    """
    client = TestClient(planner_app)
    create_response = client.get("/plan/alpha_resource")
    assert create_response.status_code == 200
    task_id = create_response.json()["task_id"]

    status_response = client.get(f"/plan/status/{task_id}")
    assert status_response.status_code == 200
    assert "status" in status_response.json()
