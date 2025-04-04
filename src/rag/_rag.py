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


config = RAGConfig()


class RAGExtractor:
    """
    TODO: add docstring
    """

    def __init__(
        self,
        docs_pool: list[Document],
        path_config: PathConfig,
        task_config: TaskConfig,
    ):
        self._config = path_config
        self._task_config = task_config
        embeddings = GoogleGenerativeAIEmbeddings( model="models/text-embedding-004" )
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
        index = faiss.IndexHNSWFlat(len(embeddings.embed_query("hello world")), 16)

        vectore_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vectore_store.add_documents(docs_pool)
        vectore_store.save_local(self._config.cache_root.as_posix())

        return vectore_store
    

    async def ainvoke(self, query: str) -> tuple[str, list[str]]:
        search_result = await self._graph.ainvoke({
            "question": query,
            "task_config": self._task_config,
        })

        return search_result["answer"], search_result["retrieved"]


if __name__ == "__main__":
    pass