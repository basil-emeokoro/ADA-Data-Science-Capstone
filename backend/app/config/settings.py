from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime paths and deterministic ML settings."""

    project_root: Path = Path(__file__).resolve().parents[3]
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def export_dir(self) -> Path:
        path = self.data_dir / "exports"
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
