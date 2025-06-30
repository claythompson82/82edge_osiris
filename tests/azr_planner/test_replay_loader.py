from __future__ import annotations

import pytest
import datetime
import gzip
import pandas as pd
import pyarrow as pa # type: ignore[import-untyped]
import pyarrow.parquet as pq # type: ignore[import-untyped]
from pathlib import Path
import tempfile
import csv
from typing import List, Dict, Any, Tuple, Optional
import math

from azr_planner.replay.loader import load_bars
from azr_planner.replay.schemas import Bar


# --- Test Data Fixtures ---
@pytest.fixture(scope="module")
def sample_bar_data() -> List[Dict[str, Any]]:
    return [
        {"timestamp": datetime.datetime(2023, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc), "instrument": "MESU24", "open": 4500.0, "high": 4505.0, "low": 4499.0, "close": 4502.0, "volume": 1000.0},
        {"timestamp": datetime.datetime(2023, 1, 1, 10, 15, 0, tzinfo=datetime.timezone.utc), "instrument": "MESU24", "open": 4502.0, "high": 4508.0, "low": 4501.0, "close": 4507.0, "volume": 1200.0},
        {"timestamp": datetime.datetime(2023, 1, 1, 10, 30, 0, tzinfo=datetime.timezone.utc), "instrument": "MESU24", "open": 4507.0, "high": 4510.0, "low": 4506.0, "close": 4509.0, "volume": 800.0},
        {"timestamp": datetime.datetime(2023, 1, 1, 10, 45, 0, tzinfo=datetime.timezone.utc), "instrument": "MESU24", "open": 4509.0, "high": 4512.0, "low": 4508.0, "close": 4511.0, "volume": 900.0},
    ]

@pytest.fixture(scope="function")
def parquet_file_fixture(sample_bar_data: List[Dict[str, Any]]) -> Path:
    df = pd.DataFrame(sample_bar_data)
    table = pa.Table.from_pandas(df, preserve_index=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmpfile:
        pq.write_table(table, tmpfile.name)
        return Path(tmpfile.name)

@pytest.fixture(scope="function")
def gzipped_parquet_file_fixture(parquet_file_fixture: Path) -> Path:
    gzipped_path = parquet_file_fixture.with_suffix(".parquet.gz")
    with open(parquet_file_fixture, "rb") as f_in, gzip.open(gzipped_path, "wb") as f_out:
        f_out.write(f_in.read())
    parquet_file_fixture.unlink()
    return gzipped_path

@pytest.fixture(scope="function")
def csv_file_fixture(sample_bar_data: List[Dict[str, Any]]) -> Path:
    df = pd.DataFrame(sample_bar_data)
    df_csv = df.copy()
    df_csv['timestamp'] = [ts.isoformat() for ts in df_csv['timestamp']]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='') as tmpfile:
        df_csv.to_csv(tmpfile.name, index=False)
        return Path(tmpfile.name)

@pytest.fixture(scope="function")
def gzipped_csv_file_fixture(csv_file_fixture: Path) -> Path:
    gzipped_path = csv_file_fixture.with_suffix(".csv.gz")
    with open(csv_file_fixture, "rb") as f_in, gzip.open(gzipped_path, "wb") as f_out:
        f_out.write(f_in.read())
    csv_file_fixture.unlink()
    return gzipped_path

# --- Tests for load_bars ---
def validate_loaded_bars(loaded_bars: List[Bar], expected_data: List[Dict[str, Any]]) -> None:
    assert len(loaded_bars) == len(expected_data)
    for i, bar in enumerate(loaded_bars):
        expected = expected_data[i]
        assert isinstance(bar, Bar)
        assert bar.timestamp == expected["timestamp"]
        assert bar.instrument == expected["instrument"]
        assert math.isclose(bar.open, expected["open"])
        assert math.isclose(bar.high, expected["high"])
        assert math.isclose(bar.low, expected["low"])
        assert math.isclose(bar.close, expected["close"])
        if expected.get("volume") is not None:
            assert bar.volume is not None and math.isclose(bar.volume, expected["volume"])
        else:
            assert bar.volume is None

def test_load_bars_parquet(parquet_file_fixture: Path, sample_bar_data: List[Dict[str, Any]]) -> None:
    bars = list(load_bars(parquet_file_fixture))
    validate_loaded_bars(bars, sample_bar_data)
    parquet_file_fixture.unlink()

def test_load_bars_gzipped_parquet(gzipped_parquet_file_fixture: Path, sample_bar_data: List[Dict[str, Any]]) -> None:
    bars = list(load_bars(gzipped_parquet_file_fixture))
    validate_loaded_bars(bars, sample_bar_data)
    gzipped_parquet_file_fixture.unlink()

def test_load_bars_csv(csv_file_fixture: Path, sample_bar_data: List[Dict[str, Any]]) -> None:
    bars = list(load_bars(csv_file_fixture))
    validate_loaded_bars(bars, sample_bar_data)
    csv_file_fixture.unlink()

def test_load_bars_gzipped_csv(gzipped_csv_file_fixture: Path, sample_bar_data: List[Dict[str, Any]]) -> None:
    bars = list(load_bars(gzipped_csv_file_fixture))
    validate_loaded_bars(bars, sample_bar_data)
    gzipped_csv_file_fixture.unlink()

def test_load_bars_string_path(csv_file_fixture: Path, sample_bar_data: List[Dict[str, Any]]) -> None:
    bars = list(load_bars(str(csv_file_fixture)))
    validate_loaded_bars(bars, sample_bar_data)
    csv_file_fixture.unlink()

def test_load_bars_malformed_csv_row(tmp_path: Path) -> None:
    bad_csv_content = ("timestamp,instrument,open,high,low,close,volume\n"
                       "2023-01-01T10:00:00Z,MESU24,4500,4505,4499,4502,1000\n"
                       "2023-01-01T10:15:00Z,MESU24,bad_price,4508,4501,4507,1200\n"
                       "2023-01-01T10:30:00Z,MESU24,4507,4510,4506,4509,800\n")
    bad_csv_file = tmp_path / "bad.csv"; bad_csv_file.write_text(bad_csv_content)
    loaded_bars = list(load_bars(bad_csv_file))
    assert len(loaded_bars) == 2

def test_load_bars_missing_mandatory_column_csv(tmp_path: Path) -> None:
    missing_col_csv_content = ("timestamp,instrument,open,high,low,volume\n"
                               "2023-01-01T10:00:00Z,MESU24,4500,4505,4499,1000\n")
    missing_col_csv_file = tmp_path / "missing_col.csv"; missing_col_csv_file.write_text(missing_col_csv_content)
    expected_error_msg_regex = r"Mandatory columns for fields \{'?close'?\} not found in headers. Expected: '?close'? \(e.g. \['close', 'Close', 'CLOSE', 'price', 'Price', 'c'\]\)"
    with pytest.raises(ValueError, match=expected_error_msg_regex):
        list(load_bars(missing_col_csv_file))

def test_load_bars_unsupported_file_type(tmp_path: Path) -> None:
    unsupported_file = tmp_path / "data.txt"; unsupported_file.write_text("some data")
    with pytest.raises(ValueError, match="Unsupported file type"):
        list(load_bars(unsupported_file))

def test_load_bars_empty_file(tmp_path: Path) -> None:
    empty_csv = tmp_path / "empty.csv"; empty_csv.write_text("")
    assert list(load_bars(empty_csv)) == []
    empty_parquet = tmp_path / "empty.parquet"
    empty_df = pd.DataFrame(columns=['timestamp', 'instrument', 'open', 'high', 'low', 'close', 'volume'])
    empty_df = empty_df.astype({
        'timestamp': 'datetime64[ns, UTC]', 'instrument': 'object',
        'open': 'float64', 'high': 'float64', 'low': 'float64',
        'close': 'float64', 'volume': 'float64'
    })
    empty_table = pa.Table.from_pandas(empty_df, preserve_index=False)
    pq.write_table(empty_table, str(empty_parquet))
    assert list(load_bars(empty_parquet)) == []

def test_load_bars_parquet_column_variations(tmp_path: Path, sample_bar_data: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(sample_bar_data)
    df.rename(columns={"timestamp": "TS", "instrument": "Symbol", "open": "OPEN", "close": "Price"}, inplace=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    parquet_file = tmp_path / "varied_cols.parquet"
    pq.write_table(table, str(parquet_file))
    bars = list(load_bars(parquet_file))
    validate_loaded_bars(bars, sample_bar_data)

def test_load_bars_csv_column_variations(tmp_path: Path, sample_bar_data: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(sample_bar_data)
    df_csv = df.copy()
    df_csv['timestamp'] = [ts.isoformat() for ts in df_csv['timestamp']]
    df_csv.rename(columns={"timestamp": "Date", "instrument": "SYMBOL", "open": "o", "close": "price"}, inplace=True)
    csv_file = tmp_path / "varied_cols.csv"
    df_csv.to_csv(csv_file, index=False)
    bars = list(load_bars(csv_file))
    validate_loaded_bars(bars, sample_bar_data)
