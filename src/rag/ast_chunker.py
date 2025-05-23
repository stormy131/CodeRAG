import json
import asyncio
from collections import defaultdict
from functools import partial

import tree_sitter_javascript as ts_js
from tree_sitter import Language as TS_Lang, Parser, Node, Tree
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from scheme.config import PathConfig


JS = TS_Lang(ts_js.language())
parser = Parser(JS)
config = PathConfig()

# Default LangChain code splitters
splitter_factory = partial(
    RecursiveCharacterTextSplitter,
    chunk_size=100,
    chunk_overlap=50,
)

SPLITTERS = defaultdict(splitter_factory)
with open(config.lang_map_path) as f:
    for ext, lang in json.load(f).items():
        SPLITTERS[ext] = RecursiveCharacterTextSplitter.from_language(
            language=Language._value2member_map_[lang],  # pylint: disable=protected-access
            chunk_size=100,
            chunk_overlap=20,
        )

# Constants for AST parsing
IGNORE = ["\n"]
TERMINAL = [
    "export_statement",
    "function_declaration",
    "variable_declaration",
    "lexical_declaration",
    "class_declaration"
]


def _parse_subtree(root: Node) -> list[Node]:
    """
    Parses a subtree of the AST starting from the given root node.

    Args:
        root (Node): The root node of the subtree.

    Returns:
        list[Node]: A list of terminal nodes found in the subtree.
    """
    subtree_nodes = []
    queue = [root]

    while queue:
        current_node = queue.pop(0)
        for child in current_node.children:
            child_type = str(child.type)
            if child_type in IGNORE:
                continue

            if child_type in TERMINAL:
                subtree_nodes.append(child)

            queue.append(child)

    return subtree_nodes


def _get_subtrees(tree: Tree) -> list[Node]:
    """
    Extracts all terminal subtrees from the given AST.

    Args:
        tree (Tree): The AST to process.

    Returns:
        list[Node]: A list of terminal nodes representing subtrees.
    """
    all_subtrees = []
    queue = [tree.root_node]

    while queue:
        current_node = queue.pop(0)
        if str(current_node.type) in TERMINAL:
            all_subtrees.append(current_node)
        else:
            subtree = _parse_subtree(current_node)
            all_subtrees.extend(subtree)
            queue.extend([x for x in current_node.children])

    return all_subtrees


async def get_chunks(documents: list[Document]) -> list[Document]:
    """
    Splits documents into smaller chunks using AST parsing or text splitting.

    Args:
        documents (list[Document]): List of documents to process.

    Returns:
        list[Document]: List of chunked documents.
    """
    result = []

    for doc in documents:
        ext = doc.metadata["source"].split(".")[-1]

        if ext == "js":  # Use AST parsing for JavaScript files
            tree = parser.parse(bytes(doc.page_content, "utf-8"))
            subtrees = _get_subtrees(tree)

            if subtrees:
                chunks = []
                for t_node in subtrees:
                    target = doc.page_content[t_node.start_byte:t_node.end_byte]
                    if target not in chunks:
                        chunks.append(target)

                result.extend([
                    Document(page_content=c, metadata=doc.metadata)
                    for c in chunks
                ])
            else:
                result.append(
                    Document(page_content=doc.page_content, metadata=doc.metadata)
                )
        else:  # Use default text splitters for other file types
            splitter = SPLITTERS[ext]
            subtrees = await splitter.atransform_documents([doc])
            result.extend(subtrees)

    return result


async def _main(text: str):
    result = await get_chunks([Document(page_content=text, metadata={"source": "test.js"})])
    print(result)


if __name__ == "__main__":
    code = """
    // program that checks if the number is positive, negative or zero
    // input from the user
    const number = parseInt(prompt("Enter a number: "));

    // check if number is greater than 0
    if (number > 0) {
        console.log("The number is positive");
    }

    // check if number is 0
    else if (number == 0) {
      console.log("The number is zero");
    }

    function test() {
        console.log("Hello world");
    }
    """

    loop = asyncio.get_event_loop()
    loop.run_until_complete(_main(code))
