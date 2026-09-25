"""
ds_proj_cc — a lightweight dataset profiling toolkit.

Provides utilities for inspecting tabular data, including:
- Dataset dimensions
- Missing values
- Duplicate rows
- Column names
- Basic numerical statistics
"""

from importlib.metadata import version as _v
from pathlib import Path
import csv
import statistics


__version__ = _v("ds-proj-cc")


def _load_csv(path: str) -> list[dict]:
    """Load a CSV file into a list of dictionaries."""

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Not a file: {path}")

    with file_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _validate_data(data: list[dict]) -> None:
    """Validate dataset structure."""

    if not isinstance(data, list):
        raise TypeError("data must be a list")

    if not all(isinstance(row, dict) for row in data):
        raise TypeError("each row must be a dictionary")


def _prepare_data(data_or_path) -> list[dict]:
    """Accept either a list of dictionaries or a CSV path."""

    if isinstance(data_or_path, (str, Path)):
        return _load_csv(str(data_or_path))

    _validate_data(data_or_path)
    return data_or_path


def profile(data_or_path) -> dict:
    """
    Generate a statistical profile of a tabular dataset.

    Parameters
    ----------
    data_or_path:
        Either:
        - a list of dictionaries, or
        - a path to a CSV file.

    Returns
    -------
    dict
        Dataset statistics.
    """

    data = _prepare_data(data_or_path)

    if not data:
        return {
            "rows": 0,
            "columns": 0,
            "missing_values": 0,
            "duplicate_rows": 0,
            "column_names": [],
            "numerical_summary": {},
        }

    # Collect all column names.
    columns = set()

    for row in data:
        columns.update(row.keys())

    columns = sorted(columns)

    # Count missing values.
    missing_values = sum(
        1
        for row in data
        for column in columns
        if row.get(column) in (None, "")
    )

    # Count duplicate rows.
    row_signatures = [
        tuple(sorted(row.items()))
        for row in data
    ]

    duplicate_rows = len(data) - len(set(row_signatures))

    # Numerical statistics.
    numerical_summary = {}

    for column in columns:
        values = []

        for row in data:
            value = row.get(column)

            if value in (None, ""):
                continue

            try:
                values.append(float(value))
            except (ValueError, TypeError):
                pass

        if values:
            numerical_summary[column] = {
                "count": len(values),
                "mean": round(statistics.mean(values), 2),
                "median": round(statistics.median(values), 2),
                "minimum": min(values),
                "maximum": max(values),
            }

    return {
        "rows": len(data),
        "columns": len(columns),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "column_names": columns,
        "numerical_summary": numerical_summary,
    }


def summarize(data_or_path) -> str:
    """
    Generate a human-readable dataset summary.
    """

    result = profile(data_or_path)

    lines = [
        "Dataset Profile",
        "────────────────────────────",
        f"Rows:              {result['rows']}",
        f"Columns:           {result['columns']}",
        f"Missing values:    {result['missing_values']}",
        f"Duplicate rows:    {result['duplicate_rows']}",
        "",
        "Columns",
        "────────────────────────────",
    ]

    for column in result["column_names"]:
        lines.append(f"- {column}")

    if result["numerical_summary"]:
        lines.extend([
            "",
            "Numerical Summary",
            "────────────────────────────",
        ])

        for column, stats in result["numerical_summary"].items():
            lines.extend([
                f"{column}:",
                f"  Count:   {stats['count']}",
                f"  Mean:    {stats['mean']}",
                f"  Median:  {stats['median']}",
                f"  Min:     {stats['minimum']}",
                f"  Max:     {stats['maximum']}",
            ])

    return "\n".join(lines)


def detect_anomalies(data_or_path) -> dict:
    """
    Detect simple anomalies in numerical columns.

    Uses the IQR (Interquartile Range) method.
    """

    data = _prepare_data(data_or_path)

    if not data:
        return {}

    columns = set()

    for row in data:
        columns.update(row.keys())

    anomalies = {}

    for column in sorted(columns):

        values = []

        for row in data:
            value = row.get(column)

            if value in (None, ""):
                continue

            try:
                values.append(float(value))
            except (ValueError, TypeError):
                continue

        if len(values) < 4:
            continue

        values.sort()

        q1 = statistics.quantiles(values, n=4)[0]
        q3 = statistics.quantiles(values, n=4)[2]

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = [
            value
            for value in values
            if value < lower_bound or value > upper_bound
        ]

        if outliers:
            anomalies[column] = {
                "outliers": outliers,
                "count": len(outliers),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
            }

    return anomalies
