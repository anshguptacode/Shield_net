"""Fetching CICIDS2017 from Kaggle.

Credentials, in the order this module tries them:

1. ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` environment variables.
2. ``~/.kaggle/kaggle.json`` (or ``$KAGGLE_CONFIG_DIR/kaggle.json``).

Get the JSON from kaggle.com -> your profile -> Settings -> API -> "Create New Token".
On Linux and macOS it must be mode 600 or the Kaggle client refuses it; this module
fixes the mode for you if it can.

A note on ``import kaggle``: the top-level package authenticates *at import time* and
raises ``OSError`` if credentials are absent, which produces a baffling traceback from
an innocuous-looking import. We import the API class directly and call
``authenticate()`` ourselves so the error can be explained properly.
"""

from __future__ import annotations

import os
import stat
import zipfile
from pathlib import Path
from typing import List, Optional

from ..logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["ensure_credentials", "download_dataset", "CredentialsError",
           "COLAB_INSTRUCTIONS"]

COLAB_INSTRUCTIONS = """\
On Google Colab, the shortest reliable path is:

    from google.colab import files
    files.upload()                 # pick your kaggle.json
    !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

then re-run this command. notebooks/ShieldNet_Colab.ipynb does this for you.
"""


class CredentialsError(RuntimeError):
    """Raised when Kaggle credentials are missing or unusable."""


def ensure_credentials() -> str:
    """Verify Kaggle credentials are discoverable; return how they were found."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "environment variables"

    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    token = config_dir / "kaggle.json"
    if token.exists():
        if os.name == "posix":
            mode = stat.S_IMODE(token.stat().st_mode)
            if mode & 0o077:
                try:
                    token.chmod(0o600)
                    log.info("tightened %s permissions to 600", token)
                except OSError:
                    log.warning("%s is mode %o; Kaggle wants 600. Run: chmod 600 %s",
                                token, mode, token)
        return str(token)

    raise CredentialsError(
        "No Kaggle credentials found.\n\n"
        "Set KAGGLE_USERNAME and KAGGLE_KEY, or place kaggle.json at "
        f"{token}.\nCreate the token at kaggle.com -> Settings -> API -> "
        "Create New Token.\n\n" + COLAB_INSTRUCTIONS
    )


def download_dataset(
    destination: Path | str,
    *,
    slug: str = "chethuhn/network-intrusion-dataset",
    fallback_slug: Optional[str] = "cicdataset/cicids2017",
    force: bool = False,
) -> List[Path]:
    """Download and unzip a CICIDS2017 mirror into *destination*.

    Skips the download when CSVs are already present unless *force* is set - a 1 GB
    re-download because a later stage crashed is a miserable way to spend ten minutes.

    Returns the list of CSV files available afterwards.
    """
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)

    existing = sorted(target.rglob("*.csv"))
    if existing and not force:
        log.info("%d CSV file(s) already in %s - skipping download (use --force to "
                 "re-fetch)", len(existing), target)
        return existing

    source = ensure_credentials()
    log.info("authenticating with Kaggle using %s", source)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise CredentialsError(
            "the `kaggle` package is not installed. Run `pip install kaggle` "
            "(it is in requirements.txt)."
        ) from exc

    api = KaggleApi()
    api.authenticate()

    last_error: Optional[Exception] = None
    for candidate in [s for s in (slug, fallback_slug) if s]:
        try:
            log.info("downloading %s -> %s (this is roughly 1 GB; expect a few "
                     "minutes)", candidate, target)
            api.dataset_download_files(candidate, path=str(target), unzip=True,
                                       quiet=False, force=force)
            break
        except Exception as exc:  # noqa: BLE001 - Kaggle raises bare ApiException
            last_error = exc
            log.warning("could not download %s: %s", candidate, exc)
    else:
        raise CredentialsError(
            f"none of the candidate Kaggle datasets could be downloaded. Last error: "
            f"{last_error}\n\nYou can also download any CICIDS2017 mirror by hand and "
            f"unzip the CSVs into {target}."
        ) from last_error

    # Some mirrors ship a nested zip that `unzip=True` leaves alone.
    for archive in target.rglob("*.zip"):
        log.info("extracting nested archive %s", archive.name)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(archive.parent)

    csvs = sorted(target.rglob("*.csv"))
    if not csvs:
        raise CredentialsError(
            f"download reported success but no .csv files appeared under {target}. "
            "Inspect the directory - the mirror's layout may have changed."
        )
    log.info("download complete: %d CSV file(s), %.1f MB total", len(csvs),
             sum(p.stat().st_size for p in csvs) / 1e6)
    return csvs
