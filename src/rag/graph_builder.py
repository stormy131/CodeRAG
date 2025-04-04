from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from scheme.config import PathConfig, RAGConfig
from scheme.graph import RAGState
from utils.prompts import make_context_prompt


path_config, rag_config = PathConfig(), RAGConfig()

llm = ChatOpenAI(
    openai_api_key=rag_config.api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name=rag_config.llm_slug,
)


# NOTE: Chain preparation for different LLM use-cases
chains: dict[str, Runnable] = {}
chat_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("system"),
    MessagesPlaceholder("context"),
    MessagesPlaceholder("user"),
])

for f_name in path_config.prompts_root.glob("*.txt"):
    with open(f_name) as f:
        slug = f_name.name.split(".")[0]
        chains[slug] = (
            chat_template.partial(system=[f.read()])
            | llm
            | StrOutputParser()
        )


# NOTE: Graph nodes
async def _expander(state: RAGState) -> RAGState:
    query = state["question"]
    extended = await chains["expand"].ainvoke({
        "user": [ query ],
        "context": [ "" ],
    })

    if state["task_config"].verbose: print(f"Expanded query: {extended}")
    return state | {
        "question": extended,
    }


def _retrieve(retriever: BaseRetriever):
    async def helper(state: RAGState) -> RAGState:
        relevant_docs = await retriever.ainvoke(state["question"])
        return state | {
            "retrieved": [d.metadata["source"] for d in relevant_docs],
            "answer": None,
        }
    
    return helper


async def _summary(state: RAGState) -> RAGState:
    result = await chains["summarize"].ainvoke({
        "user": [ state["question"] ],
        "context": [ make_context_prompt(state["retrieved"]) ],
    })

    return state | {
        "answer": result,
    }


# NOTE: Conditional edges
def _reponse_routing(state: RAGState) -> Literal["summary", "end"]:
    return "summary" if state["task_config"].summarize else "end"


def _query_init(state: RAGState) -> Literal["expand", "retrieve"]:
    return "expand" if state["task_config"].expand_query else "retrieve"


# NOTE: graph compilation
def build_graph(rag_retriever: BaseRetriever) -> CompiledStateGraph:
    builder = StateGraph(RAGState)
    builder.add_node("llm_answer", _summary)
    builder.add_sequence([
        ("expand", _expander),
        ("retrieve", _retrieve(rag_retriever)),
    ])

    builder.add_edge("llm_answer", END)
    builder.add_conditional_edges(START, _query_init)
    builder.add_conditional_edges(
        "retrieve",
        _reponse_routing,
        {
            "summary": "llm_answer",
            "end": END,
        }
    )

    return builder.compile()


if __name__ == "__main__":
    pass
