from pathlib import Path

from scheme.config import PathConfig


path_config = PathConfig()

def make_context_prompt(files: list[str]) -> str:
    result = ""

    for f_path in files:
        with open(path_config.code_repo_root / f_path, "r") as f:
            content = f.read()
            result += f"""Contents of {f_path}:\n{content}\n\n"""

    return result


if __name__ == "__main__":
    pass