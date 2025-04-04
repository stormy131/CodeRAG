import faiss
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from scheme.config import PathConfig, RAGConfig
from scheme.graph import TaskConfig
from .graph_builder import build_graph


# Load global configuration for the RAG system
config = RAGConfig()


class RAGExtractor:
    """
    RAGExtractor is responsible for performing hybrid retrieval and query answering.
    It combines dense vector search and BM25-based keyword search using an ensemble retriever.
    """

    def __init__(
        self,
        docs_pool: list[Document],
        path_config: PathConfig,
        task_config: TaskConfig,
    ):
        """
        Initializes the RAGExtractor.

        Args:
            docs_pool (list[Document]): Pool of documents to be indexed and retrieved.
            path_config (PathConfig): Configuration for file paths and cache locations.
            task_config (TaskConfig): Configuration for retrieval tasks and indexing.
        """

        self._config = path_config
        self._task_config = task_config

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        # NOTE: Use alternative pretrained embeddings
        # embeddings = PretrainedEmbeddings(config.encoder)

        bm25_retriever = BM25Retriever.from_texts(
            [d.page_content for d in docs_pool],
            metadatas=[d.metadata for d in docs_pool],
        )
        bm25_retriever.k = 10

        if not self._task_config.build_index:
            dense_store = FAISS.load_local(
                path_config.cache_root.as_posix(),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            dense_store = self._build_dense_store(docs_pool, embeddings)

        self._retriever = EnsembleRetriever(
            retrievers=[
                dense_store.as_retriever(search_kwargs={"k": 10}),
                bm25_retriever,
            ],
            weights=[0.5, 0.5],
        )
        self._graph = build_graph(self._retriever)

    def _build_dense_store(self, docs_pool: list[Document], embeddings: Embeddings):
        """
        Builds a dense vector store using FAISS for semantic search.

        Args:
            docs_pool (list[Document]): Pool of documents to be indexed.
            embeddings (Embeddings): Embedding model for generating vector representations.

        Returns:
            FAISS: A dense vector store for document retrieval.
        """

        index = faiss.IndexHNSWFlat(len(embeddings.embed_query("hello world")), 16)
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        vector_store.add_documents(docs_pool)
        vector_store.save_local(self._config.cache_root.as_posix())

        return vector_store

    async def ainvoke(self, query: str) -> tuple[str, list[str]]:
        """
        Asynchronously invokes the retrieval graph to answer a query.

        Args:
            query (str): The input query string.

        Returns:
            tuple[str, list[str]]: A tuple containing the answer and a list of retrieved documents.
        """

        search_result = await self._graph.ainvoke({
            "question": query,
            "task_config": self._task_config,
        })

        return search_result["answer"], search_result["retrieved"]


if __name__ == "__main__":
    pass
