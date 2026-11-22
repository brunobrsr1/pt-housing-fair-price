import numpy as np
import pandas as pd
import pytest

from fairprice import data


def _raw(**overrides: list) -> pd.DataFrame:
    """A two-row frame with the raw schema; override columns to build each test case."""
    base = {col: [np.nan, np.nan] for col in data.COLUMNS}
    base |= {"Price": [100_000.0, 200_000.0], "District": ["Lisboa", "Porto"]}
    base |= {"City": ["Lisboa", "Porto"], "Town": ["Estrela", "Bonfim"]}
    base |= {"Type": ["Apartment", "House"], "PublishDate": [np.nan, np.nan]}
    return pd.DataFrame(base | overrides)


@pytest.mark.parametrize(
    ("raw", "fixed"),
    [
        ("Ã\x81zere", "Ázere"),  # UTF-8 bytes read as Latin-1
        ("Vila Cova Ã\xa0 Coelheira", "Vila Cova à Coelheira"),
        ("Santo Estêvão", "Santo Estêvão"),  # correct text must pass through unchanged
        ("  Lagoa   (Algarve) ", "Lagoa (Algarve)"),
        ("São Brás", "São Brás"),  # decomposed accents become single code points
    ],
)
def test_fix_text(raw: str, fixed: str) -> None:
    assert data.fix_text(raw) == fixed


def test_fix_text_leaves_missing_alone() -> None:
    assert np.isnan(data.fix_text(np.nan))


def test_parse_floor_maps_storeys_and_leaves_layout_labels_missing() -> None:
    labels = pd.Series(
        ["Basement Level", "Ground Floor", "3rd Floor", "Above 10th Floor", "Duplex", None]
    )
    out = data.parse_floor(labels)
    assert out.iloc[:4].tolist() == [-1, 0, 3, 11]
    assert out.iloc[4:].isna().all()


def test_parse_keeps_every_row_and_renames() -> None:
    out = data.parse(_raw())
    assert len(out) == 2
    assert {"municipality", "parish", "property_type", "floor"} <= set(out.columns)
    assert "City" not in out.columns


def test_parse_energy_labels() -> None:
    out = data.parse(_raw(EnergyCertificate=["No Certificate", "Not available"]))
    assert out["energy_certificate"].iloc[0] == "NC"
    assert pd.isna(out["energy_certificate"].iloc[1])


def test_parse_dates_with_and_without_milliseconds() -> None:
    out = data.parse(_raw(PublishDate=["2025-03-28 15:34:12.570", "2025-03-28 15:34:12"]))
    assert out["publish_date"].notna().all()
    assert out["publish_date"].dt.year.tolist() == [2025, 2025]


def test_parse_booleans_keep_missing() -> None:
    out = data.parse(_raw(Garage=[True, np.nan]))
    assert out["garage"].dtype == "boolean"
    assert bool(out["garage"].iloc[0])
    assert pd.isna(out["garage"].iloc[1])
