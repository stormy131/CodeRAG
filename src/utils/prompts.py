from scheme.config import PathConfig


# Load global path configuration
path_config = PathConfig()


def make_context_prompt(files: list[str]) -> str:
    """
    Generates a context prompt by reading the contents of the specified files.

    Args:
        files (list[str]): List of file paths relative to the code repository root.

    Returns:
        str: A string containing the contents of the specified files, formatted as a prompt.
    """

    result = ""

    for f_path in files:
        with open(path_config.code_repo_root / f_path, "r") as f:
            content = f.read()
            result += f"""Contents of {f_path}:\n{content}\n\n"""

    return result


if __name__ == "__main__":
    pass
