from __future__ import annotations

import csv
import gzip
import datetime
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List, Union, cast, Iterable as TypingIterable

import pyarrow.parquet as pq # type: ignore[import-untyped]
# import pyarrow as pa # Not strictly used if pandas is used for parquet batch conversion

from azr_planner.replay.schemas import Bar

EXPECTED_COLUMNS = {
    "timestamp": ["timestamp", "Timestamp", "datetime", "Datetime", "Date", "time", "TS"],
    "instrument": ["instrument", "Instrument", "symbol", "Symbol", "INSTRUMENT", "SYMBOL"],
    "open": ["open", "Open", "OPEN", "o"], # Ensured "o" is present
    "high": ["high", "High", "HIGH", "h"],
    "low": ["low", "Low", "LOW", "l"],
    "close": ["close", "Close", "CLOSE", "price", "Price", "c"],
    "volume": ["volume", "Volume", "VOLUME", "vol", "VOL", "v"],
}

def _parse_datetime(value: str) -> Optional[datetime.datetime]:
    if not value: return None
    try: return datetime.datetime.fromisoformat(value)
    except ValueError:
        try: return datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            try:
                naive_dt = datetime.datetime.fromisoformat(value)
                if naive_dt.tzinfo is None: return naive_dt.replace(tzinfo=datetime.timezone.utc)
            except ValueError: pass
    return None

def _get_row_value(
    row_dict: Dict[str, Any],
    field_variants: List[str],
    target_type: type,
    is_optional: bool = False
) -> Any:
    raw_value_str: Optional[str] = None; found_key: Optional[str] = None
    for variant in field_variants:
        if variant in row_dict: # Assumes row_dict keys are one of the variants
            raw_value_str = str(row_dict[variant]); found_key = variant; break
    if not found_key:
        if is_optional: return None
        raise ValueError(f"Required field (variants: {', '.join(field_variants)}) not found in row data provided to _get_row_value.")
    if raw_value_str is None or not raw_value_str.strip():
        if is_optional: return None
        raise ValueError(f"Required field '{found_key}' is empty or None.")
    try:
        if target_type == datetime.datetime:
            dt_val = _parse_datetime(raw_value_str)
            if dt_val is None: raise ValueError(f"Could not parse datetime: '{raw_value_str}'")
            return dt_val
        if target_type == float: return float(raw_value_str)
        if target_type == str: return str(raw_value_str)
    except ValueError as e:
        raise ValueError(f"Error converting field '{found_key}' value '{raw_value_str}' to {target_type.__name__}: {e}")
    return raw_value_str

def _map_actual_headers_to_canonical(header_row: List[str]) -> Dict[str, str]: # maps actual_header -> canonical_name
    header_to_canonical_map: Dict[str, str] = {}
    processed_canonical_fields: set[str] = set()
    for actual_header in header_row:
        for canonical_name, variants in EXPECTED_COLUMNS.items():
            if actual_header in variants:
                if canonical_name not in processed_canonical_fields:
                    header_to_canonical_map[actual_header] = canonical_name
                    processed_canonical_fields.add(canonical_name)
                break # Move to next actual_header once a canonical match is found
    return header_to_canonical_map

def _validate_found_canonical_fields(found_canonical_fields: set[str]) -> None:
    mandatory_canonical = {"timestamp", "instrument", "open", "high", "low", "close"}
    missing = mandatory_canonical - found_canonical_fields
    if missing:
        missing_details = "; ".join([f"'{mf}' (e.g. {EXPECTED_COLUMNS[mf]})" for mf in missing])
        raise ValueError(f"Mandatory columns for fields {missing} not found in headers. Expected: {missing_details}")

def load_bars(file_path_input: Union[str, Path]) -> Iterable[Bar]:
    file_path = Path(file_path_input).expanduser().resolve()
    is_gzipped = file_path.name.endswith(".gz"); actual_filename = file_path.name.replace(".gz", "")
    opener = gzip.open if is_gzipped else open

    if actual_filename.endswith(".parquet"):
        with opener(file_path, "rb") as f:
            pf = pq.ParquetFile(f)
            # Create a map from actual Parquet column names to canonical field names
            parquet_col_to_canonical_map: Dict[str,str] = _map_actual_headers_to_canonical(pf.schema.names)
            _validate_found_canonical_fields(set(parquet_col_to_canonical_map.values()))

            for batch in pf.iter_batches():
                data_dict = batch.to_pydict()
                num_rows = batch.num_rows
                for i in range(num_rows):
                    row_for_processing: Dict[str, Any] = {}
                    for col_name_in_file, col_values in data_dict.items():
                        canonical_key = parquet_col_to_canonical_map.get(col_name_in_file)
                        if canonical_key:
                            key_for_get_row_value = EXPECTED_COLUMNS[canonical_key][0]
                            row_for_processing[key_for_get_row_value] = col_values[i]
                    try:
                        yield Bar(
                            timestamp=_get_row_value(row_for_processing, EXPECTED_COLUMNS["timestamp"], datetime.datetime),
                            instrument=_get_row_value(row_for_processing, EXPECTED_COLUMNS["instrument"], str),
                            open=_get_row_value(row_for_processing, EXPECTED_COLUMNS["open"], float),
                            high=_get_row_value(row_for_processing, EXPECTED_COLUMNS["high"], float),
                            low=_get_row_value(row_for_processing, EXPECTED_COLUMNS["low"], float),
                            close=_get_row_value(row_for_processing, EXPECTED_COLUMNS["close"], float),
                            volume=_get_row_value(row_for_processing, EXPECTED_COLUMNS["volume"], float, is_optional=True))
                    except ValueError as e: print(f"Warning: Skipping Parquet row: {e}")

    elif actual_filename.endswith(".csv"):
        mode: str = "rt" if is_gzipped else "r"
        with opener(file_path, mode, newline='', encoding='utf-8') as f_text_io:
            reader = csv.reader(cast(TypingIterable[str], f_text_io))
            try: header_row_list = next(reader)
            except StopIteration: return

            actual_header_to_canonical_map = _map_actual_headers_to_canonical(header_row_list)
            _validate_found_canonical_fields(set(actual_header_to_canonical_map.values()))

            for row_list_str in reader:
                if not row_list_str or all(not s for s in row_list_str): continue
                row_dict_for_processing: Dict[str, Any] = {}
                for i, cell_value in enumerate(row_list_str):
                    if i < len(header_row_list):
                        actual_header = header_row_list[i]
                        canonical_field = actual_header_to_canonical_map.get(actual_header)
                        if canonical_field:
                            key_for_get_row_value = EXPECTED_COLUMNS[canonical_field][0]
                            row_dict_for_processing[key_for_get_row_value] = cell_value
                try:
                    yield Bar(
                        timestamp=_get_row_value(row_dict_for_processing,EXPECTED_COLUMNS["timestamp"],datetime.datetime),
                        instrument=_get_row_value(row_dict_for_processing,EXPECTED_COLUMNS["instrument"],str),
                        open=_get_row_value(row_dict_for_processing,EXPECTED_COLUMNS["open"],float),
                        high=_get_row_value(row_dict_for_processing,EXPECTED_COLUMNS["high"],float),
                        low=_get_row_value(row_dict_for_processing,EXPECTED_COLUMNS["low"],float),
                        close=_get_row_value(row_dict_for_processing,EXPECTED_COLUMNS["close"],float),
                        volume=_get_row_value(row_dict_for_processing,EXPECTED_COLUMNS["volume"],float,is_optional=True))
                except ValueError as e: print(f"Warning: Skipping CSV row: {e}")
    else:
        raise ValueError(f"Unsupported file type: {file_path}. Supported: .csv, .csv.gz, .parquet, .parquet.gz")
