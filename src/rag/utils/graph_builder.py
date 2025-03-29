from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from scheme.config import PathConfig, RAGConfig


path_config, rag_config = PathConfig(), RAGConfig()
llm = ChatGoogleGenerativeAI(
    model=rag_config.llm_slug,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

chains: dict[str, Runnable] = {}
chat_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("system"),
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


class RAGState(TypedDict):
    question: str
    retrieved: list[str]

    # LLM integration
    require_summary: bool
    answer: str


async def _expander(state: RAGState) -> RAGState:
    query = state["question"]
    # print(query)
    extended = await chains["expand"].ainvoke({ "user": [query] })
    # print(extended)

    return state | {
        "question": extended,
        # "question": query,
    }


def _retrieve(retriever: BaseRetriever):
    async def helper(state: RAGState) -> RAGState:
        relevant_docs = await retriever.ainvoke(state["question"])
        return state | {
            "retrieved": [d.metadata["source"] for d in relevant_docs]
        }
    
    return helper


async def _summary(state: RAGState) -> RAGState:
    result = await chains["llm_answer"].ainvoke({
        "question": state["question"],
        "context": state["retrieved"]
    })

    return state | {
        "answer": result,
    }


def _reponse_routing(state: RAGState) -> Literal["summary", "end"]:
    return "summary" if state["require_summary"] else "end"


def build_graph(rag_retriever: BaseRetriever):
    builder = StateGraph(RAGState)
    builder.add_node("llm_answer", _summary)
    builder.add_sequence([
        ("expand", _expander),
        ("retrieve", _retrieve(rag_retriever)),
    ])

    builder.add_edge(START, "expand")
    builder.add_conditional_edges(
        "expand",
        _reponse_routing,
        {
            "summary": "llm_answer",
            "end": END,
        }
    )

    return builder.compile()


if __name__ == "__main__":
    pass
