"""Utility for clearing all persistent state."""

from pathlib import Path
import shutil

try:  # Support running as a script
    from .logger import get_logger
except ImportError:  # pragma: no cover - fallback when executed directly
    from logger import get_logger


def wipe(persist_dir: str | Path | None = None) -> None:
    """Delete saved memory snapshots in the persistent directory."""
    path = Path(persist_dir) if persist_dir else Path(__file__).resolve().parents[2] / "persistent"
    logger = get_logger("memory_wipe")

    if not path.exists():
        logger.info(f"{path} does not exist; nothing to wipe")
        return

    removed = False
    for filename in (
        "hippocampus.npy",
        "hippocampus.npz",
        "motor.pt",
        "angular_gyrus.pt",
        "insular.pt",
        "insula.pt",
        "semantic_flow.json",
        "semantic_flow.npy",
    ):
        target = path / filename
        if target.exists():
            target.unlink()
            logger.info(f"deleted {target}")
            removed = True

    # remove any additional files or directories except README
    for extra in list(path.iterdir()):
        if extra.name == "README.md":
            continue
        if extra.is_file():
            extra.unlink()
            logger.info(f"removed file {extra}")
            removed = True
        elif extra.is_dir():
            shutil.rmtree(extra)
            logger.info(f"removed directory {extra}")
            removed = True

    if removed:
        logger.info("memory wiped")
    else:
        logger.info("no memory snapshots found")


if __name__ == "__main__":
    wipe()
