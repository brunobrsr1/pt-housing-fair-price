"""Fetch the raw inputs into data/raw/: the Kaggle listings and the INE sale-price series.

Run with `uv run scripts/download.py`. Files that already exist are kept unless `--force` is
passed, so re-running is cheap. Each download is recorded in data/raw/MANIFEST.json with its
source URL, retrieval time and SHA-256, which is what DATA_CARD.md cites as provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import time
import urllib.request
import zipfile
from datetime import UTC, datetime

from fairprice.paths import RAW_DIR, ensure_dirs

log = logging.getLogger("download")

# Kaggle serves public datasets without credentials on this endpoint. The dataset is refreshed
# weekly, so pinning the version is what keeps our training data (and every reported metric)
# from shifting under us between runs.
KAGGLE_DATASET = "luvathoms/portugal-real-estate-2024"
KAGGLE_VERSION = 141
KAGGLE_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
    f"?datasetVersionNumber={KAGGLE_VERSION}"
)
KAGGLE_DIR = RAW_DIR / "kaggle"

# INE "median sale value per m² over the last 12 months", quarterly, NUTS 2024 geography, split by
# dwelling category (Total / New / Existing). Covers every municipality, plus parishes in the large
# metro areas. Codes were found via dados.gov.pt, which indexes INE's API. Code 0014763 is the same
# series on the 2025 parish map. We use the older map because the listings' `Town` column still
# uses the pre-2025 merged parishes (e.g. "Cascais e Estoril").
INE_INDICATORS = ("0012234",)
INE_API = "https://www.ine.pt/ine/json_indicador"
INE_DIR = RAW_DIR / "ine"
INE_PAUSE_S = 1.0  # INE throttles bursts; one request at a time with a pause keeps us welcome.

MANIFEST = RAW_DIR / "MANIFEST.json"

TIMEOUT_S = 120
RETRIES = 3
USER_AGENT = "fairprice-download/0.1"


def _get(url: str) -> bytes:
    """GET with a few retries and linear backoff, because INE's API drops connections often."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return response.read()
        except OSError as error:  # URLError, timeouts and resets are all OSError subclasses
            if attempt == RETRIES:
                raise
            log.warning("GET failed (%s), retry %d/%d", error, attempt, RETRIES - 1)
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def _record(manifest: dict, name: str, url: str, payload: bytes) -> None:
    manifest[name] = {
        "url": url,
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def download_kaggle(manifest: dict, force: bool) -> None:
    """Download and unzip the listings. The zip is not kept; its manifest hash identifies it."""
    if KAGGLE_DIR.exists() and any(KAGGLE_DIR.iterdir()) and not force:
        log.info("kaggle: %s already populated, skipping (use --force to refresh)", KAGGLE_DIR)
        return

    log.info("kaggle: downloading %s v%d", KAGGLE_DATASET, KAGGLE_VERSION)
    payload = _get(KAGGLE_URL)
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        # extractall strips absolute paths and "..", so a crafted archive cannot escape KAGGLE_DIR.
        archive.extractall(KAGGLE_DIR)
        names = archive.namelist()
    _record(manifest, "kaggle", KAGGLE_URL, payload)
    log.info("kaggle: extracted %d files into %s", len(names), KAGGLE_DIR)


def ine_quarters(meta: dict) -> list[str]:
    """Quarter codes (e.g. "S5A20254" = Q4 2025) listed in an indicator's metadata, oldest first."""
    categories = meta["Dimensoes"]["Categoria_Dim"]
    periods = [
        entry
        for block in categories
        for key, entries in block.items()
        if key.startswith("Dim_Num1_")
        for entry in entries
    ]
    return [p["categ_cod"] for p in sorted(periods, key=lambda p: p["categ_ord"])]


def download_ine(manifest: dict, force: bool) -> None:
    """Save the raw API JSON, one file per indicator and quarter.

    We fetch every published quarter instead of picking one: choosing which quarter lines up
    with the listings is an analysis decision (T6), not a download decision. One file per
    quarter also makes an interrupted run resumable.
    """
    INE_DIR.mkdir(parents=True, exist_ok=True)
    for code in INE_INDICATORS:
        meta_url = f"{INE_API}/pindicaMeta.jsp?varcd={code}&lang=PT"
        meta_payload = _get(meta_url)
        meta = json.loads(meta_payload)[0]
        (INE_DIR / f"{code}_meta.json").write_bytes(meta_payload)
        _record(manifest, f"ine/{code}_meta", meta_url, meta_payload)
        quarters = ine_quarters(meta)
        log.info("ine %s: %d quarters, last %s", code, len(quarters), meta["UltimoPeriodo"])

        for quarter in quarters:
            target = INE_DIR / f"{code}_{quarter}.json"
            if target.exists() and not force:
                continue
            url = f"{INE_API}/pindica.jsp?op=2&varcd={code}&Dim1={quarter}&lang=PT"
            payload = _get(url)
            # INE answers bad requests with HTTP 200 and an error object, so check the body.
            if "Falso" in json.loads(payload)[0].get("Sucesso", {}):
                raise RuntimeError(f"INE rejected {url}: {payload[:300]!r}")
            target.write_bytes(payload)
            _record(manifest, f"ine/{target.stem}", url, payload)
            log.info("ine %s: saved %s", code, quarter)
            time.sleep(INE_PAUSE_S)


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download files that exist")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ensure_dirs()
    manifest = _load_manifest()
    try:
        download_kaggle(manifest, args.force)
        download_ine(manifest, args.force)
    finally:
        # Save whatever succeeded, so a failure halfway (INE is flaky) keeps the provenance
        # of the files that did land.
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
