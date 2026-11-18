from pathlib import Path
import sys

import requests


URL = "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "raw" / "criteo-research-uplift-v2.1.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 100_000_000:
        print(f"exists: {out} ({out.stat().st_size:,} bytes)")
        return
    with requests.get(URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0"))
        done = 0
        with out.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    done += len(chunk)
                    if done % (20 * 1024 * 1024) < 1024 * 1024:
                        pct = f" ({done / total:.0%})" if total else ""
                        print(f"downloaded {done / 1024**2:.0f} MiB{pct}", flush=True)
    print(f"saved: {out} ({done:,} bytes)")


if __name__ == "__main__":
    main()
