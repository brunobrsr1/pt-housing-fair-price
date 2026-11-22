"""Load and parse the raw listings.

Parsing (this part) only fixes types and text. It never drops a row and never decides what is
valid: an implausible price stays in the frame until a cleaning filter removes it and logs why.
Keeping the two apart means every row that disappears is accounted for in DATA_CARD.md.
"""

from __future__ import annotations

import contextlib
import logging
import unicodedata
from pathlib import Path

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
