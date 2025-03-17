from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class RAGConfig(BaseSettings):
    api_key: str        = Field("", alias="GOOGLE_API_KEY")
    target_repo: str    = Field("https://github.com/viarotel-org/escrcpy.git",
                                alias="GIT_REPO")


class PathConfig(BaseSettings):
    logs_path: Path      = Path("./data/runs.log")
    eval_set_path: Path = Path("./data/eval/test.json")
    lang_map_path: Path = Path("./data/resources/langchain_ext_map.json")

    data_root: Path     = Path("./data/fetched")
    cache_root: Path     = Path("./data/cache")


    def __init__(self):
        super().__init__()
        self.data_root.parent.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
