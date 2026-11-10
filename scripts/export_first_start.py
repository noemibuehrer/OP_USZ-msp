"""Keep only the first start (start 0) of each case from the metrics CSV.

The result has one row per case and is what the downstream dataset merge consumes. Columns that
only describe the multistart run (and are constant after filtering) are dropped.

    uv run python scripts/export_first_start.py IN_CSV OUT_CSV
"""

import sys

import pandas as pd

RUN_COLUMNS = ["plane_source", "plane", "start", "is_first_start", "is_best", "volume_region"]

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
df = df[df["is_first_start"]].drop(columns=RUN_COLUMNS)
assert df["case_id"].is_unique
df.to_csv(dst, index=False)
