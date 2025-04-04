from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class RAGConfig(BaseSettings):
    api_key: str        = Field("", alias="API_KEY")
    target_repo: str    = Field("", alias="GIT_REPO")
    encoder: str        = Field("microsoft/codebert-base", alias="ENCODER_MODEL")
    llm_slug: str       = Field("", alias="LLM")


_DATA_ROOT = Path("./data")
class PathConfig(BaseSettings):
    logs_path: Path         = _DATA_ROOT / "runs.log"
    eval_set_path: Path     = _DATA_ROOT / "eval/test.json"
    lang_map_path: Path     = _DATA_ROOT / "resources/langchain_ext_map.json"

    code_repo_root: Path    = _DATA_ROOT / "fetched"
    cache_root: Path        = _DATA_ROOT / "cache"
    prompts_root: Path      = _DATA_ROOT / "prompts"

    def __init__(self):
        super().__init__()

        self.logs_path.touch(exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.code_repo_root.parent.mkdir(parents=True, exist_ok=True)
