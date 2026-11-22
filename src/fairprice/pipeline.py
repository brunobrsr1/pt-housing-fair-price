"""End-to-end entry point: `uv run -m fairprice.pipeline`.

I/O happens only here, at the edges; data.py, features.py, etc. stay pure functions of frames.
Later tasks (T3-T6) add their steps after `build_listings`.
"""

from __future__ import annotations

import logging

import pandas as pd

from fairprice import data
from fairprice.paths import PROCESSED_DIR, ensure_dirs

log = logging.getLogger(__name__)

LISTINGS = PROCESSED_DIR / "listings.parquet"
CLEANING_REPORT = PROCESSED_DIR / "cleaning_report.csv"


def build_listings() -> pd.DataFrame:
    """T1: raw CSV -> parsed -> cleaned -> data/processed/listings.parquet."""
    listings, report = data.clean(data.parse(data.load_raw()))
    listings.to_parquet(LISTINGS, index=False)
    report.to_csv(CLEANING_REPORT, index=False)
    log.info("wrote %s (%d rows) and %s", LISTINGS, len(listings), CLEANING_REPORT)
    print(report.to_string(index=False))
    return listings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ensure_dirs()
    build_listings()


if __name__ == "__main__":
    main()
