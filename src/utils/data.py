import json
import asyncio
from pathlib import Path
from collections import defaultdict
from itertools import chain
from functools import partial

from scheme.config import PathConfig
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


# Preload text splitter, for each available language in langchain
# TODO: chunk_size??
config = PathConfig()
splitter_factory = partial(
    RecursiveCharacterTextSplitter,
    chunk_size=50,
    chunk_overlap=0,
)
SPLITTERS = defaultdict(splitter_factory)

with open(config.lang_map_path) as f:
    for ext, lang in json.load(f).items():
        SPLITTERS[ext] = RecursiveCharacterTextSplitter.from_language(
            language=Language._value2member_map_[lang], # pylint: disable=protected-access
            chunk_size=100,
            chunk_overlap=0,
        )


# TODO: move logging output into separate log stash
def load_docs(path_config: PathConfig) -> list[Document]:
    docs = []
    files = [
        [root / f for f in files]
        for root, _, files in path_config.data_root.walk()
    ]
    print("Skipped files in knowledge base:")

    for f_path in chain(*files):
        try:
            f =  open(f_path, "r")
        except UnicodeDecodeError:
            print(f_path)
        else:
            with f:
                docs.append(
                    Document(
                        page_content=f.read(), metadata={ "source": f_path }
                    )
                )

    return docs


async def get_chunks(docs_pool: list[Document]) -> list[Document]:
    chunks: list[Document] = []
    for doc in docs_pool:
        ext = doc.metadata["source"].split(".")[-1]
        splitter = SPLITTERS[ext]
        chunks.extend( await splitter.atransform_documents([doc]) )

    return chunks


async def test():
    documents = load_docs(config)
    res = await get_chunks(documents)

    breakpoint()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test())
