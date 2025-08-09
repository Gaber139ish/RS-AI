import logging
import os
from typing import Optional


def setup_logging(verbose: Optional[bool] = None) -> None:
    """Configure global logging.

    If verbose is None, read RS_VERBOSE env var ("1"/"true" case-insensitive)
    Level is DEBUG when verbose, WARNING otherwise.
    """
    if verbose is None:
        env = os.getenv("RS_VERBOSE", "0").strip().lower()
        verbose = env in {"1", "true", "yes", "on"}

    level = logging.DEBUG if verbose else logging.WARNING

    # Avoid duplicate handlers if called twice
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d [%(threadName)s] - %(message)s",
    )

    # Reduce noise from third-party libs unless verbose
    if not verbose:
        for noisy in ("urllib3", "matplotlib", "PIL", "numexpr"):
            logging.getLogger(noisy).setLevel(logging.WARNING)