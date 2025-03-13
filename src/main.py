from asyncio import get_event_loop
from pathlib import Path

from git import Repo
from dotenv import load_dotenv

from rag.rag import RAGExtractor
from utils.data import load_docs, get_chunks


load_dotenv()
# TODO: Path config
GIT_REPO = "https://github.com/viarotel-org/escrcpy.git"
DATA_ROOT = Path("./data/fetched")


async def main():
    if not DATA_ROOT.exists():
        Repo.clone_from(GIT_REPO, DATA_ROOT)

    documents = load_docs(DATA_ROOT)
    chunked = await get_chunks(documents)
    rag = RAGExtractor(chunked)

    query = "How is the wireless pairing dialog functionality implemented?"
    result = await rag.ainvoke(query)
    breakpoint()


if __name__ == "__main__":
    loop = get_event_loop() 
    loop.run_until_complete( main() )
