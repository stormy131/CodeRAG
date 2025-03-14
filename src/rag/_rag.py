from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, START


class RAGState(TypedDict):
    question: str
    retrieved: list[str]


class RAGExtractor:
    """
    TODO: add docstring
    """

    # TODO: caching vector store index
    def __init__(self, docs_pool: list[Document]):
        embeddings = GoogleGenerativeAIEmbeddings( model="models/text-embedding-004" )
        self._vector_store = InMemoryVectorStore(embeddings)
        self._vector_store.add_documents(docs_pool)

        builder = StateGraph(RAGState).add_node("retrieve", self._retrieve)
        builder.add_edge(START, "retrieve")
        self._graph = builder.compile()


    # TODO: search result pre-filtering
    async def _retrieve(self, state: RAGState) -> RAGState:
        relevant_docs = await self._vector_store.asimilarity_search(state["question"], k=10)
        return state | {
            "retrieved": [d.metadata["source"] for d in relevant_docs]
        }


    async def ainvoke(self, query: str) -> list[str]:
        search_result = await self._graph.ainvoke({"question": query})
        return search_result["retrieved"]


if __name__ == "__main__":
    pass