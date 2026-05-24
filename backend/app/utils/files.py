from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def safe_stem(filename: str) -> str:
    return Path(filename).stem.replace(" ", "_").replace("-", "_")


def timestamped_name(prefix: str, suffix: str = ".csv") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid4().hex[:8]}{suffix}"
