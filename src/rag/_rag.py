from typing import TypedDict

import faiss
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore import InMemoryDocstore
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, START

from rag._chunker import get_chunks
from scheme.config import PathConfig


class RAGState(TypedDict):
    question: str
    retrieved: list[str]


class RAGExtractor:
    """
    TODO: add docstring
    """

    @classmethod
    async def create(cls, documents: list[Document], path_config: PathConfig, *, load: bool = False):
        self = cls()
        builder = StateGraph(RAGState).add_node("retrieve", self._retrieve)
        builder.add_edge(START, "retrieve")
        self._graph = builder.compile()

        # TODO: generic embedder from config
        embeddings = GoogleGenerativeAIEmbeddings( model="models/text-embedding-004" )
        if load:
            self._vector_store = FAISS.load_local(
                path_config.cache_root.as_posix(),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            self._vector_store = await self._build_index(documents, embeddings) # type: ignore
            self._vector_store.save_local(path_config.cache_root.as_posix())

        return self


    async def _build_index(self, docs_pool: list[Document], embeddings: Embeddings) -> VectorStore:
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        chunks = await get_chunks(docs_pool)
        vector_store.add_documents(chunks)
        return vector_store


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