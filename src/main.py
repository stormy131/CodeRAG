import os
from asyncio import get_event_loop

from git import Repo
from dotenv import load_dotenv
load_dotenv()

from rag import RAGExtractor
from utils.data import load_docs, get_chunks
from evaluation import Evaluator
from scheme.config import PathConfig, RAGConfig
from rag.utils.ast_chunker import get_chunks


path_config = PathConfig()
rag_config = RAGConfig()


# TODO: argumet switches parsing
async def main():
    if not path_config.data_root.exists():
        Repo.clone_from(rag_config.target_repo, path_config.data_root)

    documents = load_docs(path_config)
    # documents = await get_chunks(documents)
    # documents = await get_chunks(documents)
    rag = RAGExtractor(documents, path_config, load=True)

    eval = Evaluator(path_config)
    await eval.test(rag, note="hybrid + flash_rerank", verbose=True)


if __name__ == "__main__":
    loop = get_event_loop()
    loop.run_until_complete( main() )
