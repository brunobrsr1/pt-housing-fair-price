"""Load, parse and clean the raw listings.

Parsing (this part) only fixes types and text. It never drops a row and never decides what is
valid: an implausible price stays in the frame until a cleaning filter removes it and logs why.
Keeping the two apart means every row that disappears is accounted for in DATA_CARD.md.
"""

from __future__ import annotations

import contextlib
import logging
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from fairprice.paths import RAW_DIR

log = logging.getLogger(__name__)

RAW_CSV = RAW_DIR / "kaggle" / "portugal_listinigs.csv"  # misspelled upstream

# Upstream names are misleading: `City` holds the municipality (concelho) and `Town` the parish
# (freguesia). Renaming once here stops that confusion from leaking into every later module.
COLUMNS = {
    "Price": "price",
    "District": "district",
    "City": "municipality",
    "Town": "parish",
    "Type": "property_type",
    "EnergyCertificate": "energy_certificate",
    "Floor": "floor_label",
    "NumberOfBathrooms": "bathrooms",
    "Parking": "parking",
    "TotalArea": "total_area",
    "ConstructionYear": "construction_year",
    "EnergyEfficiencyLevel": "energy_efficiency_level",
    "PublishDate": "publish_date",
    "Garage": "garage",
    "Elevator": "elevator",
    "ElectricCarsCharging": "electric_cars_charging",
    "TotalRooms": "total_rooms",
    "NumberOfBedrooms": "bedrooms",
    "NumberOfWC": "wc",
    "ConservationStatus": "conservation_status",
    "LivingArea": "living_area",
    "LotSize": "lot_size",
    "BuiltArea": "built_area",
}

# Floor labels -> storey number. "Above 10th Floor" becomes 11, a lower bound rather than the real
# floor. Labels that describe layout or position instead of a storey (Duplex, Top Floor, Attic...)
# get NaN; the original label is kept in `floor_label`, so nothing is lost.
FLOOR_NUMBER = {
    "Basement": -1,
    "Basement Level": -1,
    "Ground Floor": 0,
    **{f"{n}{suffix} Floor": n for n, suffix in [(1, "st"), (2, "nd"), (3, "rd")]},
    **{f"{n}th Floor": n for n in range(4, 10)},
    "Above 10th Floor": 11,
}

# The two portals spell "no certificate" differently. "Not available" means the field is unknown,
# which is not the same thing, so it becomes missing.
ENERGY_SYNONYMS = {"No Certificate": "NC"}
ENERGY_UNKNOWN = ["Not available"]

TEXT_COLUMNS = ["district", "municipality", "parish", "property_type", "conservation_status"]
BOOL_COLUMNS = ["garage", "elevator", "electric_cars_charging"]
ENERGY_COLUMNS = ["energy_certificate", "energy_efficiency_level"]


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Read the CSV untouched. low_memory=False infers each column's type from the whole column."""
    return pd.read_csv(path, low_memory=False)


def fix_text(value: str | float) -> str | float:
    """Repair UTF-8 read as Latin-1 ("Ãzere" -> "Ázere"), then normalise spaces and Unicode.

    Only strings containing "Ã" or "Â" are re-decoded: those characters are the signature of this
    kind of mojibake, and title-case place names don't use them. If re-decoding fails, the text
    was not mojibake, so it is kept as is. NFC makes "é" one code point everywhere, so
    equal-looking names compare equal in joins (e.g. with INE in T6).
    """
    if not isinstance(value, str):
        return value
    if "Ã" in value or "Â" in value:
        with contextlib.suppress(UnicodeEncodeError, UnicodeDecodeError):
            value = value.encode("latin-1").decode("utf-8")
    return unicodedata.normalize("NFC", " ".join(value.split()))


def parse_floor(labels: pd.Series) -> pd.Series:
    """Storey number from the floor label (see FLOOR_NUMBER), as float so NaN fits."""
    return labels.map(FLOOR_NUMBER).astype("float64")


def parse(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename and type the raw columns. Same rows in, same rows out."""
    df = raw.rename(columns=COLUMNS)[list(COLUMNS.values())].copy()

    for col in TEXT_COLUMNS:
        df[col] = df[col].map(fix_text).astype("string")

    for col in ENERGY_COLUMNS:
        unknown = df[col].isin(ENERGY_UNKNOWN)
        df[col] = df[col].mask(unknown).replace(ENERGY_SYNONYMS).astype("string")
        log.info("parse: %s, %d 'Not available' set to missing", col, unknown.sum())

    df["floor_label"] = df["floor_label"].astype("string")
    df["floor"] = parse_floor(df["floor_label"])
    no_storey = df["floor_label"].notna() & df["floor"].isna()
    log.info(
        "parse: floor, %d labels without a storey number (kept in floor_label)", no_storey.sum()
    )

    for col in BOOL_COLUMNS:
        df[col] = df[col].astype("boolean")

    # Two formats appear (with and without milliseconds); ISO8601 accepts both.
    df["publish_date"] = pd.to_datetime(df["publish_date"], format="ISO8601")

    if len(df) != len(raw):  # parsing must never drop rows; that is the filters' job
        raise AssertionError(f"parse changed the row count: {len(raw)} -> {len(df)}")
    return df


# --------------------------------------------------------------------------------------------
# Cleaning. Every threshold below is a decision; DATA_CARD.md gives the evidence for each.
# --------------------------------------------------------------------------------------------

# Homes people live in. Estate and Farm are mostly land (median EUR/m2 on total_area ~70-390) and
# "Other - Residential" mixes rooms, ruins and garages (~500 EUR/m2 vs 1,800-3,600 for houses
# and flats). The original label is kept; grouping types together is a feature choice (T4).
IN_SCOPE_TYPES = ("Apartment", "House", "Duplex", "Studio", "Mansion", "Manor")
OUTSIDE_PORTUGAL = "Z - Fora de Portugal"  # Brazil and Dubai listings

# living_area, not total_area: total_area holds lot sizes, repeated placeholders (10,368 m2 on
# many Albufeira flats) and negative values. Below 20 m2 the rows are typos (a 7 m2 flat at
# EUR 145k); above 2,000 m2 they are lot sizes typed into the living-area field.
MIN_AREA_M2, MAX_AREA_M2 = 20, 2_000

# Monthly rents sit far below this (cheapest "sales" were 75-117 m2 flats at EUR 600-1,000).
MIN_PRICE_EUR = 10_000
# Real rural houses go down to ~200 EUR/m2; below 100 the rows are rents or area typos. Prime
# Lisbon tops out near 20k EUR/m2; above 25k the area is wrong (or the price: EUR 2 billion).
MIN_EUR_M2, MAX_EUR_M2 = 100, 25_000

# Dated listings older than this carry stale prices (INE sale prices were rising ~18% a year by
# 2026). Only 0.5% of in-scope rows are older. Undated rows (all of source A) are kept: dropping
# them would drop ~40% of the data.
MIN_PUBLISH_DATE = pd.Timestamp("2024-01-01")

# Counts above these are typos (3,020 bathrooms) or a portal's "20+" cap (bedrooms stop at 21).
# The value becomes missing; the row stays, because its price and location are still fine.
MAX_COUNTS = {"bedrooms": 15, "bathrooms": 10, "wc": 10, "total_rooms": 20}

# The columns both sources publish identically. Same values here = same home. Every key column
# is location-or-finer, so merged rows always share a parish: wrongly merging two identical
# new-build flats costs a little weight, never a leak across grouped CV folds.
DEDUP_KEY = ["price", "property_type", "district", "municipality", "parish", "living_area"]


def _log_drop(step: str, before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    log.info("%-28s removed %6d rows, %6d left", step, len(before) - len(after), len(after))
    return after


def add_source(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each row with its inferred portal. The file has no portal column, but one source never
    publishes garage, EV charging or energy level, so missing `garage` identifies it (A)."""
    return df.assign(source=df["garage"].isna().map({True: "A", False: "B"}).astype("string"))


def keep_in_scope_types(df: pd.DataFrame) -> pd.DataFrame:
    return _log_drop("keep_in_scope_types", df, df[df["property_type"].isin(IN_SCOPE_TYPES)])


def drop_outside_portugal(df: pd.DataFrame) -> pd.DataFrame:
    return _log_drop("drop_outside_portugal", df, df[df["district"].ne(OUTSIDE_PORTUGAL)])


def drop_implausible_area(df: pd.DataFrame) -> pd.DataFrame:
    """Missing living_area goes too: without it there is no EUR/m2 and no baseline (T3)."""
    keep = df["living_area"].between(MIN_AREA_M2, MAX_AREA_M2)
    return _log_drop("drop_implausible_area", df, df[keep])


def drop_rentals_and_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    return _log_drop("drop_rentals_and_placeholders", df, df[df["price"].ge(MIN_PRICE_EUR)])


def drop_implausible_eur_m2(df: pd.DataFrame) -> pd.DataFrame:
    eur_m2 = df["price"] / df["living_area"]
    return _log_drop("drop_implausible_eur_m2", df, df[eur_m2.between(MIN_EUR_M2, MAX_EUR_M2)])


def drop_stale_listings(df: pd.DataFrame) -> pd.DataFrame:
    keep = df["publish_date"].isna() | df["publish_date"].ge(MIN_PUBLISH_DATE)
    return _log_drop("drop_stale_listings", df, df[keep])


def null_impossible_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Negative or absurd room counts become missing. No rows are removed."""
    df = df.copy()
    for col, cap in MAX_COUNTS.items():
        bad = df[col].lt(0) | df[col].gt(cap)
        df.loc[bad, col] = np.nan
        log.info("%-28s %6d values of %s set to missing", "null_impossible_counts", bad.sum(), col)
    return df


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return _log_drop("drop_exact_duplicates", df, df.drop_duplicates())


def drop_relisted_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Same listing scraped again later: identical except for publish_date. Keep the latest."""
    others = [c for c in df.columns if c != "publish_date"]
    latest = df.sort_values("publish_date", ascending=False, na_position="last", kind="stable")
    out = latest.drop_duplicates(subset=others).sort_index()
    return _log_drop("drop_relisted_duplicates", df, out)


def merge_cross_source_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the same home listed on both portals into one row.

    Per column the first non-missing value wins, with source B rows first (more fields) and the
    latest date first. That keeps A's floor and B's bedrooms in the same merged row.
    `source` becomes "A", "B" or "AB".
    """
    ordered = df.sort_values(
        ["source", "publish_date"], ascending=False, na_position="last", kind="stable"
    )
    groups = ordered.groupby(DEDUP_KEY, dropna=False, sort=False)
    merged = groups.first()
    merged["source"] = groups["source"].agg(lambda s: "".join(sorted(set(s)))).astype("string")
    out = merged.reset_index()[df.columns]
    return _log_drop("merge_cross_source_duplicates", df, out)


# Order matters: scope and validity first, so duplicates are judged on valid rows only, and
# all deduplication happens before any train/test split exists.
CLEANING_STEPS = (
    keep_in_scope_types,
    drop_outside_portugal,
    drop_implausible_area,
    drop_rentals_and_placeholders,
    drop_implausible_eur_m2,
    drop_stale_listings,
    null_impossible_counts,
    drop_exact_duplicates,
    drop_relisted_duplicates,
    merge_cross_source_duplicates,
)


def clean(parsed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every cleaning step. Returns the clean frame and a per-step row-count report."""
    df = add_source(parsed)
    rows = [{"step": "parsed input", "removed": 0, "rows": len(df)}]
    for step in CLEANING_STEPS:
        before = len(df)
        df = step(df)
        rows.append({"step": step.__name__, "removed": before - len(df), "rows": len(df)})
    return df.reset_index(drop=True), pd.DataFrame(rows)
