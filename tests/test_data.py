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


# --- cleaning --------------------------------------------------------------------------------


def _clean_frame(n: int = 1, **cols: list) -> pd.DataFrame:
    """Parsed-schema rows that pass every filter; override columns to break one rule at a time."""
    base = {
        "price": [300_000.0] * n,
        "property_type": ["Apartment"] * n,
        "district": ["Lisboa"] * n,
        "municipality": ["Lisboa"] * n,
        "parish": ["Estrela"] * n,
        "living_area": [100.0] * n,
        "publish_date": [pd.Timestamp("2026-05-01")] * n,
        "garage": pd.array([True] * n, dtype="boolean"),
        "floor": [np.nan] * n,
        "bedrooms": [2.0] * n,
        "bathrooms": [1.0] * n,
        "wc": [np.nan] * n,
        "total_rooms": [3.0] * n,
    }
    return data.add_source(pd.DataFrame(base | cols))


@pytest.mark.parametrize(
    ("filter_fn", "column", "values", "kept"),
    [
        (data.keep_in_scope_types, "property_type", ["Apartment", "Land", "Duplex"], 2),
        (data.drop_outside_portugal, "district", ["Lisboa", "Z - Fora de Portugal", "Faro"], 2),
        # bounds are inclusive; missing area is dropped
        (data.drop_implausible_area, "living_area", [19.9, 20.0, np.nan], 1),
        (data.drop_implausible_area, "living_area", [2_000.0, 2_000.1, 50.0], 2),
        (data.drop_rentals_and_placeholders, "price", [1.0, 9_999.0, 10_000.0], 1),
        # living_area 100: EUR/m2 of 99, 100 and 25,001
        (data.drop_implausible_eur_m2, "price", [9_900.0, 10_000.0, 2_500_100.0], 1),
    ],
)
def test_filters_edges(filter_fn, column: str, values: list, kept: int) -> None:
    df = _clean_frame(3, **{column: values})
    assert len(filter_fn(df)) == kept


def test_stale_filter_keeps_undated_rows() -> None:
    dates = [pd.Timestamp("2023-12-31"), pd.Timestamp("2024-01-01"), pd.NaT]
    out = data.drop_stale_listings(_clean_frame(3, publish_date=dates))
    assert len(out) == 2
    assert out["publish_date"].isna().sum() == 1


def test_null_impossible_counts_keeps_rows() -> None:
    out = data.null_impossible_counts(_clean_frame(3, bathrooms=[-1.0, 3_020.0, 2.0]))
    assert len(out) == 3
    assert out["bathrooms"].isna().tolist() == [True, True, False]


def test_relisted_duplicates_keep_latest() -> None:
    dates = [pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")]
    out = data.drop_relisted_duplicates(_clean_frame(2, publish_date=dates))
    assert out["publish_date"].tolist() == [pd.Timestamp("2026-01-01")]


def test_cross_source_merge_combines_fields_and_prefers_b() -> None:
    # Row 0 from source A (no garage field, has floor); row 1 from source B (has bedrooms).
    df = _clean_frame(
        2,
        garage=pd.array([pd.NA, False], dtype="boolean"),
        floor=[3.0, np.nan],
        bedrooms=[np.nan, 2.0],
        bathrooms=[9.0, 1.0],  # conflict: B wins
    )
    out = data.merge_cross_source_duplicates(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert (row["source"], row["floor"], row["bedrooms"], row["bathrooms"]) == ("AB", 3, 2, 1)


def test_cross_source_merge_keeps_different_homes() -> None:
    out = data.merge_cross_source_duplicates(_clean_frame(2, parish=["Estrela", "Belém"]))
    assert len(out) == 2


def test_clean_report_accounts_for_every_row() -> None:
    df = _clean_frame(4, property_type=["Apartment", "Land", "House", "Apartment"])
    out, report = data.clean(df.drop(columns="source"))
    assert report["rows"].iloc[-1] == len(out)
    assert report["rows"].iloc[0] - report["removed"].sum() == len(out)
