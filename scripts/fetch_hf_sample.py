"""Fetch a small, reproducible slice from the public Criteo uplift dataset.

The Hugging Face datasets server exposes row-level access, which lets a
portfolio repo avoid committing the 300MB raw archive while still shipping a
real, reproducible sample for the demo run.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random
import time

import pandas as pd
import requests


BASE = "https://datasets-server.huggingface.co/rows"
PARAMS = {"dataset": "criteo/criteo-uplift", "config": "default", "split": "train"}


def fetch(offset: int, length: int = 100) -> list[dict]:
    for attempt in range(6):
        response = requests.get(BASE, params={**PARAMS, "offset": offset, "length": length}, timeout=60)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return [row["row"] for row in response.json()["rows"]]
    raise RuntimeError(f"request failed at offset {offset}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "sample" / "criteo_sample_10000.csv"
    # The source is sorted in large blocks.  Spread requests across the full
    # split so treatment/control are both represented in the portfolio sample.
    rng = random.Random(20261119)
    offsets = [rng.randrange(0, 13_979_592 - 100, 100) for _ in range(30)]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, offset): offset for offset in offsets}
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                rows.extend(future.result())
            except Exception as exc:
                print(f"skipped batch: {exc}", flush=True)
            if i % 10 == 0:
                print(f"fetched {i * 100:,} rows", flush=True)
    frame = pd.DataFrame(rows).sort_values("f0").reset_index(drop=True)
    frame.to_csv(out, index=False)
    print(f"saved {len(frame):,} rows to {out}")


if __name__ == "__main__":
    main()
