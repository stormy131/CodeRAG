from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class RAGConfig(BaseSettings):
    target_repo: str = Field(alias="GIT_REPO")


class PathConfig(BaseSettings):
    log_root: Path      = Path("./data/logs")
    data_root: Path     = Path("./data/fetched")
    eval_root: Path     = Path("./data/eval")
    
    lang_map_path: Path = Path("./data/resources/langchain_ext_map.json")
