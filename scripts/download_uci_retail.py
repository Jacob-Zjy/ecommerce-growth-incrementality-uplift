"""Download the public UCI Online Retail transaction sample used by the project."""

from pathlib import Path

import requests


URL = "https://raw.githubusercontent.com/dbdmg/data-science-lab/master/datasets/online_retail.csv"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "ecommerce" / "online_retail.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with out.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    print(f"saved {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
