# Data card: `data/processed/listings.parquet`

Asking prices of apartments and houses for sale in Portugal, cleaned and deduplicated.
Built by `uv run -m fairprice.pipeline` (code: `src/fairprice/data.py`).

**What the target is:** the *asking* price a seller published on a portal, not a sale price and
not a valuation. The model learns what sellers ask.

## Origin

| | |
|---|---|
| Source | Kaggle [`luvathoms/portugal-real-estate-2024`](https://www.kaggle.com/datasets/luvathoms/portugal-real-estate-2024), **version 141** (pinned in `scripts/download.py`) |
| Licence | MIT |
| Raw file | `portugal_listinigs.csv` (sic), 190,867 rows × 23 columns |
| Provenance | URL, retrieval time and SHA-256 in `data/raw/MANIFEST.json` |
| Deviation from the brief | The brief's dataset only locates homes by district; this one has municipality and parish, which grouped validation (T3), the INE comparison (T6) and the demo (T7) need. |

## Period

`publish_date` is only filled for 57% of the final rows. **One portal never publishes a date**
(see "Two sources" below). Among dated rows, after cleaning:

| 2024 Q1–Q4 | 2025 Q1–Q4 | 2026 Q1–Q4 (to 3 Nov) |
|---|---|---|
| 14,273 | 8,111 | 27,651 |

So the data is mostly 2024 Q3 – 2026 Q4, weighted towards mid-2026. Undated rows come from the
weekly-refreshed dataset and are assumed to be current listings. That is an assumption, not a fact.

## Two sources

The file has no portal column, but `garage`, `electric_cars_charging` and
`energy_efficiency_level` are missing on exactly the same 68,247 raw rows, and those rows
never have `bedrooms`, `publish_date`, `conservation_status` or `lot_size` either. They also
are the only rows with a `floor`. The pipeline infers a `source` column from this pattern:

| source | meaning | final rows |
|---|---|---|
| `A` | portal without garage / date / bedrooms fields | 20,101 |
| `B` | portal with those fields | 50,421 |
| `AB` | same home found on both, merged into one row | 17,813 |

Most "sparse columns" in the raw data are this schema split, not values missing at random.

## Row counts per step

| # | Step | Rule | Removed | Rows left |
|---|---|---|---:|---:|
| 0 | parsed input | types and names fixed, no rows dropped | 0 | 190,867 |
| 1 | `keep_in_scope_types` | `Apartment`, `House`, `Duplex`, `Studio`, `Mansion`, `Manor` | 67,485 | 123,382 |
| 2 | `drop_outside_portugal` | district `Z - Fora de Portugal` (Brazil, Dubai) | 43 | 123,339 |
| 3 | `drop_implausible_area` | `living_area` missing or outside 20–2,000 m² | 747 | 122,592 |
| 4 | `drop_rentals_and_placeholders` | price < €10,000 | 1,279 | 121,313 |
| 5 | `drop_implausible_eur_m2` | price / living_area outside €100–25,000/m² | 238 | 121,075 |
| 6 | `drop_stale_listings` | published before 2024-01-01 (undated rows kept) | 621 | 120,454 |
| 7 | `null_impossible_counts` | negative counts, or bedrooms > 15, bathrooms > 10, WC > 10, total rooms > 20, set to missing (196 values; no rows removed) | 0 | 120,454 |
| 8 | `drop_exact_duplicates` | identical rows | 6,626 | 113,828 |
| 9 | `drop_relisted_duplicates` | identical except `publish_date`; latest kept | 3,679 | 110,149 |
| 10 | `merge_cross_source_duplicates` | same price, type, district, municipality, parish and living area; merged | 21,814 | **88,335** |

The same table is written to `data/processed/cleaning_report.csv` on every run.

## Why each rule

1. **Scope.** Median €/m² of Apartment 3,600, House 1,810, Duplex 4,024, Studio 5,348, Mansion
   3,196, Manor 2,890. Excluded: `Estate` and `Farm` (mostly land: €72 and €390/m² on
   `total_area`, total areas up to 331,250 m²) and `Other - Residential` (852 rows, €500/m²,
   a mix of rooms, ruins and garages). Land, commercial, garages and "transfer of lease" are
   out of scope by definition. The original type label is kept.
2. **Geography.** Madeira and the Azores are kept (170 final rows), because INE covers them.
3. **Area.** `living_area` is the area used everywhere. `total_area` is unreliable: it holds lot
   sizes (in duplicate pairs, source A's `total_area` equals source B's `lot_size`), repeated
   placeholders (10,368 m² on many Albufeira flats), 56 million m² flats and negative values.
   Below 20 m² the rows were typos (a 7 m² Lisbon flat at €145k); above 2,000 m² they were lot sizes.
4. **Rentals and placeholders.** There is no sale/rent column. The cheapest "sales" were 75–117 m²
   flats at €600–1,000, i.e. monthly rents, plus a €1 house. 470 apartments but only 16 houses
   were under €5k, the signature of rental flats.
5. **€/m².** Real rural houses go down to ~€200/m² (Montalegre, Penela). Below €100/m² the rows
   were rents or area typos. Above €25,000/m² the area or price was wrong (two houses at
   €1.38 billion and €2 billion).
6. **Stale listings.** Only 0.5% of in-scope rows were dated before 2024. Their prices are from a
   market that has since risen ~18% a year.
7. **Impossible counts.** 3,020 bathrooms, 2,751 total rooms, negative WC counts. `bedrooms`
   stops at exactly 21, which looks like a portal's "20+" option. The row's price and location
   are still valid, so the value is nulled rather than the row dropped.
8. **Deduplication** happens before any train/test split. Every key in the cross-source merge
   is location-or-finer, so merged rows always share a parish. A wrong merge (two identical
   new-build flats) costs some weight but cannot leak across geographic CV folds. In a merged
   row, each column takes the first non-missing value, with source B and the latest date first.

## Final columns

| Column | Type | Filled | Notes |
|---|---|---:|---|
| `price` | float | 100% | **Target.** Asking price, € |
| `district`, `municipality`, `parish` | string | 100% | Upstream `District`, `City`, `Town`. Parish names repeat across municipalities: identify a parish by (municipality, parish) |
| `property_type` | string | 100% | One of the six in-scope labels |
| `living_area` | float | 100% | m², 20–2,000 |
| `total_area`, `built_area`, `lot_size` | float | 100 / 48 / 32% | Unreliable (see above) |
| `bedrooms` | float | 77% | Source B only, plus merged rows |
| `total_rooms` | float | 69% | In source B usually `bedrooms + 1` |
| `bathrooms`, `wc` | float | 100 / 51% | Bathrooms is often 0 where unknown |
| `floor`, `floor_label` | float, string | 24 / 25% | Source A only. Basement −1, ground 0, "Above 10th" 11 (a lower bound); Duplex, Top Floor, Attic → NaN |
| `construction_year` | float | 90% | |
| `energy_certificate` | string | 100% | `NC` = no certificate |
| `energy_efficiency_level` | string | 77% | Looks like a copy of `energy_certificate` |
| `conservation_status` | string | 27% | |
| `garage`, `elevator`, `electric_cars_charging` | boolean | 77 / 100 / 77% | |
| `parking` | float | 100% | |
| `publish_date` | datetime | 57% | Source B only |
| `source` | string | 100% | Inferred portal: `A`, `B`, `AB` |

No column is derived from the price, so nothing here leaks the target into features.

## Known biases and limitations

- **Asking ≠ sale.** Listings over-represent homes that are hard to sell (they stay listed longer
  and are re-scraped) and may over-represent expensive segments.
- **Uneven coverage.** Viseu district has only 20 final listings, against 1,000+ for Bragança. The
  portals or the scraper barely cover it. The model will know little about Viseu.
- **Thin parishes.** 2,370 (municipality, parish) pairs, many with fewer than 5 listings.
- **Source split.** Floor exists only for source A, bedrooms and dates only for source B. A model
  can learn "which portal" from the missing-value pattern.
- **Remaining duplicates.** The same home after a price change, or with a slightly different area
  on each portal, is not merged. These stay within one parish, so they add weight but do not
  leak across grouped folds.
- **Rental filter is a threshold.** Rents above €10,000/month, and real sales under €10,000
  (rural ruins), are misclassified.
- **Undated rows** are assumed current. If source A holds old listings, its prices are stale.
- **Mixed timing.** The data spans about two and a half years of a rising market. Aligning with
  INE (T6) needs a deliberate choice of period.
