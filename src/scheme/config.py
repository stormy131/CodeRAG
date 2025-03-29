from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


DEFAULT_REPO = "https://github.com/viarotel-org/escrcpy.git"


class RAGConfig(BaseSettings):
    api_key: str        = Field("", alias="GOOGLE_API_KEY")
    target_repo: str    = Field(DEFAULT_REPO, alias="GIT_REPO")
    encoder: str        = Field("microsoft/codebert-base", alias="ENCODER_MODEL")
    llm_slug: str       = Field("gemini-1.5-flash", alias="LLM")


_DATA_ROOT = Path("./data")
class PathConfig(BaseSettings):
    logs_path: Path     = _DATA_ROOT / "runs.log"
    eval_set_path: Path = _DATA_ROOT / "eval/test.json"
    lang_map_path: Path = _DATA_ROOT / "resources/langchain_ext_map.json"

    data_root: Path     = _DATA_ROOT / "fetched"
    cache_root: Path    = _DATA_ROOT / "cache"
    prompts_root: Path  = _DATA_ROOT / "prompts"

    def __init__(self):
        super().__init__()
        self.data_root.parent.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
