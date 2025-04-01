from argparse import ArgumentParser, Namespace
from asyncio import get_event_loop

from git import Repo
from dotenv import load_dotenv
load_dotenv()

from rag import RAGExtractor
from utils.data import load_docs
from evaluation import Evaluator
from scheme.config import PathConfig, RAGConfig
from rag.ast_chunker import get_chunks


path_config = PathConfig()
rag_config = RAGConfig()
parser = ArgumentParser()
parser.add_argument("--mode", default="qa", help="determine RAG inference mode", choices=["qa", "evaluate"])
parser.add_argument("--verbose", action="store_true", help="sets verbosity for RAG interactions")
parser.add_argument("--expand_query", action="store_true", help="enables query expansion by LLM")
args = parser.parse_args()


async def main(args: Namespace):
    if not path_config.code_repo_root.exists():
        Repo.clone_from(rag_config.target_repo, path_config.code_repo_root)

    documents = load_docs(path_config)
    # documents = await get_chunks(documents)
    rag = RAGExtractor(documents, path_config, load=True)

    print(args.__dict__)
    match args.mode:
        case "qa":
            try:
                while True:
                    query = input("Q: ")
                    response = await rag.aanswer(query)

                    print(f"A: {response}\n")
            except KeyboardInterrupt:
                pass
        case "evaluate":
            eval = Evaluator(path_config)
            await eval.test(rag, note="hybrid + flash_rerank", verbose=args.verbose)
        case _:
            raise ValueError("Invalid mode argument.")


if __name__ == "__main__":
    loop = get_event_loop()
    loop.run_until_complete( main(args) )
