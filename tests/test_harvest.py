import argparse
import datetime
import json
import lancedb
import os
import pandas as pd
import pyarrow as pa
import pytest
import shutil
import tempfile
from unittest.mock import patch

from osiris.scripts.harvest_feedback import main as harvest_main


@pytest.fixture
def temp_db_path_fixture():  # Renamed to avoid conflict with possible test args
    """Create a temporary directory for LanceDB."""
    td = tempfile.mkdtemp(prefix="test_harvest_db_")
    yield td
    shutil.rmtree(td)


@pytest.fixture
def sample_data_fixture():  # Renamed
    """Provides sample data for testing."""
    now = datetime.datetime.now(datetime.timezone.utc)
    one_day_ago = now - datetime.timedelta(days=1)
    ten_days_ago = now - datetime.timedelta(days=10)

    now_ns = int(now.timestamp() * 1_000_000_000)
    one_day_ago_ns = int(one_day_ago.timestamp() * 1_000_000_000)
    ten_days_ago_ns = int(ten_days_ago.timestamp() * 1_000_000_000)
    six_days_23_hours_ago_ns = int(
        (now - datetime.timedelta(days=6, hours=23)).timestamp() * 1_000_000_000
    )

    data = [
        # Record 1: Should be included (recent, correct type, has corrected_proposal)
        {
            "id": "1",
            "feedback_type": "correction",
            "assessment": "Prompt 1",
            "proposal": "Original proposal 1",
            "corrected_proposal": json.dumps({"new_text": "Corrected text 1"}),
            "when": now_ns,
            "user_id": "user_a",
        },
        # Record 2: Should be included (assessment is None, uses proposal)
        {
            "id": "2",
            "feedback_type": "correction",
            "assessment": None,
            "proposal": "Original proposal 2",
            "corrected_proposal": json.dumps({"new_text": "Corrected text 2"}),
            "when": one_day_ago_ns,
            "user_id": "user_b",
        },
        # Record 3: Should be excluded (wrong feedback_type)
        {
            "id": "3",
            "feedback_type": "rating",
            "assessment": "Prompt 3",
            "proposal": "Original proposal 3",
            "corrected_proposal": json.dumps({"new_text": "Corrected text 3"}),
            "when": now_ns,
            "user_id": "user_c",
        },
        # Record 4: Should be excluded (corrected_proposal is None)
        {
            "id": "4",
            "feedback_type": "correction",
            "assessment": "Prompt 4",
            "proposal": "Original proposal 4",
            "corrected_proposal": None,
            "when": now_ns,
            "user_id": "user_d",
        },
        # Record 5: Should be excluded (corrected_proposal is empty string)
        {
            "id": "5",
            "feedback_type": "correction",
            "assessment": "Prompt 5",
            "proposal": "Original proposal 5",
            "corrected_proposal": "",
            "when": now_ns,
            "user_id": "user_e",
        },
        # Record 6: Should be excluded by default days_back=7 (too old)
        {
            "id": "6",
            "feedback_type": "correction",
            "assessment": "Prompt 6",
            "proposal": "Original proposal 6",
            "corrected_proposal": json.dumps({"new_text": "Corrected text 6"}),
            "when": ten_days_ago_ns,
            "user_id": "user_f",
        },
        # Record 7: Should be included (boundary for days_back=7)
        {
            "id": "7",
            "feedback_type": "correction",
            "assessment": "Prompt 7",
            "proposal": "Original proposal 7",
            "corrected_proposal": json.dumps({"new_text": "Corrected text 7"}),
            "when": six_days_23_hours_ago_ns,
            "user_id": "user_g",
        },
        # Record 8: Should be included (complex JSON in corrected_proposal)
        {
            "id": "8",
            "feedback_type": "correction",
            "assessment": "Prompt 8",
            "proposal": "Original proposal 8",
            "corrected_proposal": json.dumps({"detail": {"key": "value"}}),
            "when": now_ns,
            "user_id": "user_h",
        },
        # Record 9: Should be excluded (corrected_proposal is not valid JSON string)
        {
            "id": "9",
            "feedback_type": "correction",
            "assessment": "Prompt 9",
            "proposal": "Prop 9",
            "corrected_proposal": "this is not json",
            "when": now_ns,
            "user_id": "user_i",
        },
        # Record 10: Should be excluded (both assessment and proposal are None/empty)
        {
            "id": "10",
            "feedback_type": "correction",
            "assessment": None,
            "proposal": None,
            "corrected_proposal": json.dumps({"text": "text"}),
            "when": now_ns,
            "user_id": "user_j",
        },
        # Record 11: Assessment is empty string, proposal is valid
        {
            "id": "11",
            "feedback_type": "correction",
            "assessment": "",
            "proposal": "Original proposal 11",
            "corrected_proposal": json.dumps({"new_text": "Corrected text 11"}),
            "when": now_ns,
            "user_id": "user_k",
        },
    ]
    return data


@pytest.fixture
def db_with_data_fixture(temp_db_path_fixture, sample_data_fixture):  # Renamed
    """Populate a temporary LanceDB with sample data."""
    db = lancedb.connect(temp_db_path_fixture)
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("feedback_type", pa.string()),
            pa.field("assessment", pa.string()),
            pa.field("proposal", pa.string()),
            pa.field("corrected_proposal", pa.string()),
            pa.field("when", pa.timestamp("ns", tz="UTC")),
            pa.field("user_id", pa.string()),
        ]
    )

    df = pd.DataFrame(sample_data_fixture)
    df["when"] = pd.to_datetime(df["when"], unit="ns", utc=True)
    for col in ["assessment", "proposal", "corrected_proposal"]:
        df[col] = df[col].astype(object).where(pd.notnull(df[col]), None)

    tbl = db.create_table("phi3_feedback", data=df, schema=schema)
    return db, tbl, temp_db_path_fixture  # Return path for use in run_script_test


def run_script_test_helper(db_path_for_script_to_use, output_file, script_args_list):
    """Helper function to run the harvest_feedback script with mocking."""
    SCRIPT_DB_PATH_IN_SCRIPT_CODE = "/app/lancedb_data"

    def mock_lancedb_connect(path_arg_in_connect_call):
        # This mock redirects the hardcoded path in script to our temp test DB path
        if path_arg_in_connect_call == SCRIPT_DB_PATH_IN_SCRIPT_CODE:
            return lancedb.connect(db_path_for_script_to_use)
        return lancedb.connect(
            path_arg_in_connect_call
        )  # Original behavior for any other paths

    # Prepare arguments for ArgumentParser mock
    # Defaults from script: days_back=7, out="feedback_data.jsonl", max=None
    parsed_args_dict = {"days_back": 7, "out": "feedback_data.jsonl", "max": None}

    # Override defaults with provided script_args_list
    # Example: if script_args_list = ["--out", "custom.jsonl", "--max", "10"]
    # This will update parsed_args_dict: {"out": "custom.jsonl", "max": 10}
    it = iter(script_args_list)
    for k, v in zip(it, it):
        parsed_args_dict[k.lstrip("-").replace("-", "_")] = int(v) if v.isdigit() else v

    # Ensure 'out' is always the provided output_file for the test
    parsed_args_dict["out"] = output_file

    # Mock os.path.exists for the hardcoded DB path in the script
    with (
        patch("harvest_feedback.os.path.exists") as mock_path_exists,
        patch("harvest_feedback.lancedb.connect", side_effect=mock_lancedb_connect),
        patch.object(
            argparse.ArgumentParser,
            "parse_args",
            return_value=argparse.Namespace(**parsed_args_dict),
        ),
    ):

        mock_path_exists.return_value = (
            True  # Deceive script that "/app/lancedb_data" exists
        )
        harvest_main()

    results = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():  # Ensure not an empty line
                    results.append(json.loads(line))
    return results


def test_default_extraction(db_with_data_fixture):
    _, _, temp_db_path = db_with_data_fixture  # Get the actual temp_db_path used
    output_jsonl = os.path.join(temp_db_path, "output.jsonl")

    # Default --days-back is 7.
    # Expected records: 1, 2, 7, 8, 11 (Record 11 has empty assessment but valid proposal)
    # Record 6 (10 days old) excluded.
    # Record 3 (type), 4 (None CP), 5 (empty CP), 9 (bad JSON CP), 10 (no prompt) excluded.
    args = ["--days-back", "7"]
    results = run_script_test_helper(temp_db_path, output_jsonl, args)

    expected_prompts_and_responses = {
        "Prompt 1": json.dumps({"new_text": "Corrected text 1"}, indent=2),
        "Original proposal 2": json.dumps({"new_text": "Corrected text 2"}, indent=2),
        "Prompt 7": json.dumps({"new_text": "Corrected text 7"}, indent=2),
        "Prompt 8": json.dumps({"detail": {"key": "value"}}, indent=2),
        "Original proposal 11": json.dumps({"new_text": "Corrected text 11"}, indent=2),
    }

    assert len(results) == len(expected_prompts_and_responses)

    for res in results:
        assert res["prompt"] in expected_prompts_and_responses
        assert res["response"] == expected_prompts_and_responses[res["prompt"]]


def test_days_back_filtering(db_with_data_fixture):
    _, _, temp_db_path = db_with_data_fixture
    output_jsonl = os.path.join(temp_db_path, "output_days_back.jsonl")

    # Records 1, 2, 8, 11 are <1 day old. Record 7 is ~6 days old. Record 6 is 10 days old.
    args = [
        "--days-back",
        "3",
    ]  # Only records from last 3 days (strictly < 3 days ago, not incl. start of 3rd day)
    results = run_script_test_helper(temp_db_path, output_jsonl, args)

    # Expected: 1, 2, 8, 11 (all < 3 days old based on 'now', 'one_day_ago')
    # Note: 'when' values are 'now_ns', 'one_day_ago_ns'. Cutoff is 'now - 3 days'.
    # So these records are within the 3-day window.
    assert len(results) == 4  # 1, 2, 8, 11
    prompts_in_results = {r["prompt"] for r in results}
    assert "Prompt 1" in prompts_in_results
    assert "Original proposal 2" in prompts_in_results
    assert "Prompt 8" in prompts_in_results
    assert "Original proposal 11" in prompts_in_results
    assert "Prompt 7" not in prompts_in_results  # ~6 days old, excluded
    assert "Prompt 6" not in prompts_in_results  # 10 days old, excluded

    args_15_days = ["--days-back", "15"]
    results_15_days = run_script_test_helper(temp_db_path, output_jsonl, args_15_days)
    # Expected: 1, 2, 6, 7, 8, 11 (all 6 valid records as 15 days covers all)
    assert len(results_15_days) == 6
    prompts_in_results_15 = {r["prompt"] for r in results_15_days}
    assert "Prompt 6" in prompts_in_results_15


def test_max_records_limit(db_with_data_fixture):
    _, _, temp_db_path = db_with_data_fixture
    output_jsonl = os.path.join(temp_db_path, "output_max.jsonl")
    # Using 15 days to include all 6 valid records, but limit to 2
    args = ["--days-back", "15", "--max", "2"]
    results = run_script_test_helper(temp_db_path, output_jsonl, args)
    assert len(results) == 2


def test_empty_assessment_uses_proposal(
    db_with_data_fixture,
):  # Also covers empty string assessment
    _, _, temp_db_path = db_with_data_fixture
    output_jsonl = os.path.join(temp_db_path, "output_proposal.jsonl")

    # Default days_back=7. This includes records 1, 2, 7, 8, 11.
    # Record 2: assessment=None, proposal="Original proposal 2"
    # Record 11: assessment="", proposal="Original proposal 11"
    results = run_script_test_helper(temp_db_path, output_jsonl, [])  # Use default args

    found_record_2 = False
    found_record_11 = False
    for r in results:
        if r["prompt"] == "Original proposal 2":
            found_record_2 = True
            assert r["response"] == json.dumps(
                {"new_text": "Corrected text 2"}, indent=2
            )
        elif r["prompt"] == "Original proposal 11":
            found_record_11 = True
            assert r["response"] == json.dumps(
                {"new_text": "Corrected text 11"}, indent=2
            )

    assert found_record_2, "Record 2 (None assessment) not found or incorrect."
    assert found_record_11, "Record 11 (empty assessment) not found or incorrect."


def test_no_valid_data_produces_empty_file(temp_db_path_fixture):  # Use empty DB
    # Don't use db_with_data_fixture, create an empty one or one with no valid data
    db = lancedb.connect(temp_db_path_fixture)
    schema = pa.schema(
        [  # Define schema matching the script's expectations
            pa.field("id", pa.string()),
            pa.field("feedback_type", pa.string()),
            pa.field("assessment", pa.string()),
            pa.field("proposal", pa.string()),
            pa.field("corrected_proposal", pa.string()),
            pa.field("when", pa.timestamp("ns", tz="UTC")),
            pa.field("user_id", pa.string()),
        ]
    )
    # Create an empty table
    db.create_table("phi3_feedback", schema=schema)

    output_jsonl = os.path.join(temp_db_path_fixture, "output_empty.jsonl")
    results = run_script_test_helper(
        temp_db_path_fixture, output_jsonl, []
    )  # Default args

    assert len(results) == 0
    assert os.path.exists(output_jsonl)
    with open(output_jsonl, "r") as f:
        assert f.read().strip() == ""


def test_corrected_proposal_is_actual_json_string(db_with_data_fixture):
    _, _, temp_db_path = db_with_data_fixture
    output_jsonl = os.path.join(temp_db_path, "output_json_parsing.jsonl")

    # Test focuses on record 8: "corrected_proposal": json.dumps({"detail": {"key": "value"}})
    # This should be correctly parsed and then pretty-printed.
    args = ["--days-back", "1"]  # Default includes records 1,2,8,11 (from today)
    results = run_script_test_helper(temp_db_path, output_jsonl, args)

    found_record_8 = False
    expected_response_8 = json.dumps({"detail": {"key": "value"}}, indent=2)
    for r in results:
        if r["prompt"] == "Prompt 8":
            found_record_8 = True
            assert r["response"] == expected_response_8
            break
    assert (
        found_record_8
    ), "Record 8 with complex JSON corrected_proposal not found or incorrect."
