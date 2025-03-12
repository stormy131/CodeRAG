import json
import asyncio
from pathlib import Path
from collections import defaultdict
from functools import partial

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


# TODO: chunk_size??
splitter_factory = partial(
    RecursiveCharacterTextSplitter,
    chunk_size=50,
    chunk_overlap=0,
)
SPLITTERS = defaultdict(splitter_factory)

with open("./src/resources/langchain_ext_map.json") as f:
    for ext, lang in json.load(f).items():
        SPLITTERS[ext] = RecursiveCharacterTextSplitter.from_language(
            language=Language._value2member_map_[lang], # pylint: disable=protected-access
            chunk_size=100,
            chunk_overlap=0,
        )


# TODO: move logging output into separate log stash
def load_docs(root_dir: Path) -> list[Document]:
    docs = []
    print("Skipped files in knowledge base:")
    for root, _, files in root_dir.walk():
        for path in [(root / f).as_posix() for f in files]:
            try:
                with open(path, "r") as f:
                    docs.append(
                        Document(
                            page_content=f.read(),
                            metadata={ "source": path }
                        )
                    )
            except UnicodeDecodeError:
                print(path)

    return docs


async def get_chunks(docs_pool: list[Document]) -> list[Document]:
    chunks = []
    for doc in docs_pool:
        ext = doc.metadata["source"].split(".")[-1]
        splitter = SPLITTERS[ext]
        chunks.extend( await splitter.atransform_documents([doc]) )

    return chunks


def make_vectorstore(data: list[Document]):
    pass


async def test():
    documents = load_docs( Path("./data") )
    res = await get_chunks(documents)
    breakpoint()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test())
