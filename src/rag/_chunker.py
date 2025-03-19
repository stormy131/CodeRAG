import json
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
splitter_factory = partial(
    RecursiveCharacterTextSplitter,
    chunk_size=50,
    chunk_overlap=25,
)

TERMINAL = [
    'if_statement',
    'while_statement',
    'for_statement',
    'for_range_loop',
]
IGNORE = ["\n"]
SPLITTERS = defaultdict(splitter_factory)

with open(config.lang_map_path) as f:
    for ext, lang in json.load(f).items():
        SPLITTERS[ext] = RecursiveCharacterTextSplitter.from_language(
            language=Language._value2member_map_[lang], # pylint: disable=protected-access
            chunk_size=100,
            chunk_overlap=50,
        )


def _parse_subtree(root: Node) -> list[Node]:
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


# TODO: review TS language mapping in resources
async def get_chunks(documents: list[Document]) -> list[Document]:
    result = []

    for doc in documents:
        ext = doc.metadata["source"].split(".")[-1]

        if ext == "js":
            tree = parser.parse(bytes(doc.page_content, "utf-8"))
            subtrees = _get_subtrees(tree)
            result.extend([
                Document(
                    page_content=doc.page_content[
                        t_node.start_byte:t_node.end_byte
                    ],
                    metadata=doc.metadata,
                )
                for t_node in subtrees
            ])
        else:
            splitter = SPLITTERS[ext]
            result.extend(await splitter.atransform_documents([doc]))

    return result


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

    tree = parser.parse(bytes(code, "utf-8"))
