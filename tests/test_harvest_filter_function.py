import datetime
from osiris.scripts.harvest_feedback import record_matches_filter


def make_record(version, ftype, when_ns):
    return {"schema_version": version, "feedback_type": ftype, "when": when_ns}


def test_filter_accepts_matching_record():
    cutoff = int((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)).timestamp() * 1e9)
    rec = make_record("1.0", "correction", cutoff + 1)
    assert record_matches_filter(rec, cutoff_ns=cutoff, schema_version="1.0")


def test_filter_rejects_old_or_wrong_type():
    cutoff = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e9)
    old = make_record("1.0", "correction", cutoff - 10)
    wrong_type = make_record("1.0", "rating", cutoff + 10)
    assert not record_matches_filter(old, cutoff_ns=cutoff, schema_version="1.0")
    assert not record_matches_filter(wrong_type, cutoff_ns=cutoff, schema_version="1.0")


def test_filter_prefix_match():
    cutoff = 0
    rec = make_record("1.0.1", "correction", cutoff + 5)
    assert record_matches_filter(rec, cutoff_ns=cutoff, schema_version="1.0")
