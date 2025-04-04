from itertools import chain

from scheme.config import PathConfig
from langchain_core.documents import Document


# Load global path configuration
config = PathConfig()


def load_docs(path_config: PathConfig, verbose: bool = False) -> list[Document]:
    """
    Loads documents from the code repository.

    Args:
        path_config (PathConfig): Configuration object containing repository paths.
        verbose (bool): If True, logs skipped files.

    Returns:
        list[Document]: List of documents loaded from the repository.
    """

    docs = []

    files = [
        [root / f for f in files]
        for root, _, files in path_config.code_repo_root.walk()
    ]

    for f_path in chain(*files):
        with open(f_path, "r") as f:
            try:
                docs.append(Document(
                    page_content=f.read(),
                    metadata={
                        "source": f_path.relative_to(path_config.code_repo_root).as_posix()
                    }
                ))
            except UnicodeDecodeError:
                if verbose: print("Skipping", f_path)

    return docs


if __name__ == "__main__":
    documents = load_docs(config)
