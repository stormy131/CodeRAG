from asyncio import get_event_loop

from git import Repo
from dotenv import load_dotenv

from rag import RAGExtractor
from utils.data import load_docs, get_chunks
from evaluation import Evaluator
from scheme.config import PathConfig, RAGConfig


load_dotenv()
path_config = PathConfig()
rag_config = RAGConfig()


# TODO: argumet switches parsing
async def main():
    if not path_config.data_root.exists():
        Repo.clone_from(rag_config.target_repo, path_config.data_root)

    documents = load_docs(path_config)
    chunked = await get_chunks(documents)
    rag = RAGExtractor(chunked, path_config, load=True)

    eval = Evaluator(path_config)
    await eval.test(rag, note="langchain chunking", verbose=True)

    breakpoint()


if __name__ == "__main__":
    loop = get_event_loop()
    loop.run_until_complete( main() )
