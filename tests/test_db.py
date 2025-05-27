from llm_sidecar.db import append_feedback, feedback_tbl
import datetime # Added for timestamp construction

def test_feedback_append():
    # Ensure the table is clean before testing, or that tests account for existing data.
    # For simplicity, we'll assert count increases. A more robust test might clear the table
    # or use a dedicated test table if the main table is persistent across tests.
    initial_count = 0
    try:
        initial_count = feedback_tbl.count_rows()
    except Exception: # Handle case where table might not exist before first append or is empty
        pass

    row = {
        "transaction_id": "test123",
        "timestamp": datetime.datetime.utcnow().isoformat(), # Using actual timestamp
        "feedback_type": "rating",
        "feedback_content": "✅"
    }
    append_feedback(row)
    
    # Verify the row was added
    final_count = feedback_tbl.count_rows()
    assert final_count > initial_count, "Row count should increase after adding feedback."

    # Optional: Query the table to verify the content of the added row.
    # This requires knowing more about how to query LanceDB, e.g., using SQL or a vector search.
    # For now, just checking the count as per the issue's test structure.
    # Example of a more specific check if data retrieval is straightforward:
    # result_df = feedback_tbl.search().limit(final_count).to_df() # Get all rows
    # assert not result_df[result_df['transaction_id'] == 'test123'].empty, "Test row not found by transaction_id"
