import pytest

from ds_proj_cc import profile, summarize, detect_anomalies


def test_profile_basic_dataset():
    data = [
        {"name": "Alice", "age": 21},
        {"name": "Bob", "age": 22},
    ]

    result = profile(data)

    assert result["rows"] == 2
    assert result["columns"] == 2
    assert result["missing_values"] == 0
    assert result["duplicate_rows"] == 0


def test_profile_missing_values():
    data = [
        {"name": "Alice", "age": 21},
        {"name": "Bob", "age": None},
    ]

    result = profile(data)

    assert result["missing_values"] == 1


def test_profile_duplicate_rows():
    data = [
        {"name": "Alice", "age": 21},
        {"name": "Alice", "age": 21},
        {"name": "Bob", "age": 22},
    ]

    result = profile(data)

    assert result["duplicate_rows"] == 1


def test_numerical_summary():
    data = [
        {"name": "Alice", "score": 80},
        {"name": "Bob", "score": 90},
        {"name": "Charlie", "score": 100},
    ]

    result = profile(data)

    assert result["numerical_summary"]["score"]["mean"] == 90.0
    assert result["numerical_summary"]["score"]["median"] == 90.0
    assert result["numerical_summary"]["score"]["minimum"] == 80.0
    assert result["numerical_summary"]["score"]["maximum"] == 100.0


def test_summarize():
    data = [
        {"name": "Alice", "age": 21},
        {"name": "Bob", "age": 22},
    ]

    result = summarize(data)

    assert "Dataset Profile" in result
    assert "Rows:" in result
    assert "Columns:" in result

def test_anomaly_detection():
    data = [
        {"score": 10},
        {"score": 11},
        {"score": 12},
        {"score": 13},
        {"score": 12},
        {"score": 11},
        {"score": 13},
        {"score": 12},
        {"score": 11},
        {"score": 100},
    ]

    result = detect_anomalies(data)

    assert "score" in result
    assert 100.0 in result["score"]["outliers"]


def test_invalid_file_path():
    with pytest.raises(FileNotFoundError):
        profile("not a dataset")


def test_invalid_row_type():
    with pytest.raises(TypeError):
        profile([{"name": "Alice"}, "invalid row"])
