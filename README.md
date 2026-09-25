# ds-proj-cc

**TDS Profiler**

A lightweight Python toolkit for *dataset profiling, statistical summaries, missing-value detection, duplicate detection, and numerical anomaly detection.*

## Install

```bash
pip install ds-proj-cc

## Usage

```python
from ds_proj_cc import profile, summarize, detect_anomalies

data = [
    {"name": "Alice", "age": 21, "score": 85},
    {"name": "Bob", "age": 22, "score": 91},
    {"name": "Charlie", "age": None, "score": 78},
]

print(profile(data))
print(summarize(data))
print(detect_anomalies(data))


## Author
# Chirantan Chakraborty
