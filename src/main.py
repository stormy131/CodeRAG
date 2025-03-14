from asyncio import get_event_loop
from pathlib import Path

from git import Repo
from dotenv import load_dotenv

from scheme.config import PathConfig, RAGConfig
from rag.rag import RAGExtractor
from utils.data import load_docs, get_chunks


load_dotenv()
path_config = PathConfig()
rag_config = RAGConfig()


async def main():
    if not path_config.data_root.exists():
        Repo.clone_from(rag_config.target_repo, path_config.data_root)

    documents = load_docs(path_config)
    chunked = await get_chunks(documents)
    rag = RAGExtractor(chunked)

    query = "How is the wireless pairing dialog functionality implemented?"
    result = await rag.ainvoke(query)
    breakpoint()


if __name__ == "__main__":
    loop = get_event_loop() 
    loop.run_until_complete( main() )
