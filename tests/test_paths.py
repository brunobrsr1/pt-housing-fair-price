from fairprice import paths


def test_root_is_repository_root() -> None:
    # If the package layout changes, parents[2] silently points somewhere else.
    assert (paths.ROOT / "pyproject.toml").is_file()


def test_data_dirs_live_under_root() -> None:
    for directory in (paths.RAW_DIR, paths.PROCESSED_DIR, paths.FIGURES_DIR, paths.MODELS_DIR):
        assert directory.is_relative_to(paths.ROOT)
