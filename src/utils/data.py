from itertools import chain

from scheme.config import PathConfig
from langchain_core.documents import Document


# TODO: move logging output into separate log stash
def load_docs(path_config: PathConfig) -> list[Document]:
    docs = []
    files = [
        [root / f for f in files]
        for root, _, files in path_config.data_root.walk()
    ]
    print("Skipped files in knowledge base:")

    for f_path in chain(*files):
        with open(f_path, "r") as f:
            try:
                docs.append(Document(
                    page_content=f.read(),
                    metadata={
                        "source": f_path.relative_to(path_config.data_root).as_posix()
                    }
                ))
            except UnicodeDecodeError:
                print(f_path)

    return docs

if __name__ == "__main__":
    config = PathConfig()
    docs = load_docs(config)
