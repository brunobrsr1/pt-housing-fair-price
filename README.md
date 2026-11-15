# Fair Price

Estimates the **asking price** of a home in Portugal with a calibrated 80% interval, then compares
asking prices with INE deed (sale) prices per municipality to find where sellers ask the most above
what homes sell for.

> Work in progress. The full report (data, validation protocol, results, findings and limitations)
> will live here at `v1.0`.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # install from uv.lock (Python 3.12)
uv run ruff check . && uv run ruff format --check .
uv run pytest
```
