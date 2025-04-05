from asyncio import get_event_loop
from argparse import ArgumentParser, Namespace

from git import Repo
from dotenv import load_dotenv
load_dotenv()

from rag import RAGExtractor
from rag.ast_chunker import get_chunks
from utils.data import load_docs
from evaluation import Evaluator
from scheme.config import PathConfig, RAGConfig
from scheme.graph import TaskConfig


path_config, rag_config = PathConfig(), RAGConfig()

parser = ArgumentParser()
parser.add_argument("--mode", default="qa", help="determine RAG inference mode", choices=["qa", "evaluate"])
parser.add_argument("--verbose", action="store_true", help="sets verbosity for RAG interactions")
parser.add_argument("--expand_query", action="store_true", help="enables query expansion by LLM")
parser.add_argument("--build_index", action="store_true", help="builds (new) vectore store index for the documents pool")
args = parser.parse_args()


async def main(args: Namespace):
    if not path_config.code_repo_root.exists():
        print(f"Cloning the codebase for the RAG system...")
        Repo.clone_from(rag_config.target_repo, path_config.code_repo_root)

    mode = args.__dict__.pop("mode")
    task_config = TaskConfig(**args.__dict__, summarize=(mode == "qa"))
    documents = load_docs(path_config, args.verbose)
    # NOTE: uncomment the following line to chunk the documents with AST strategy
    # documents = await get_chunks(documents)

    rag = RAGExtractor(documents, path_config, task_config)
    match mode:
        case "qa":
            try:
                while True:
                    query = input("[ Q ]: ")
                    response, retrived = await rag.ainvoke(query)

                    print(f"\nRETRIEVED ITEMS: {retrived}\n")
                    print(f"[ A ]:\n{response}\n")
            except:
                pass
        case "evaluate":
            eval = Evaluator(path_config)
            await eval.test(rag, note="test run", verbose=args.verbose)
        case _:
            raise ValueError("Invalid mode argument.")


if __name__ == "__main__":
    loop = get_event_loop()
    loop.run_until_complete( main(args) )
