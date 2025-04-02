import torch
import numpy as np
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

from scheme.config import RAGConfig


config = RAGConfig()


class PretrainedEmbeddings(Embeddings):
    def __init__(self, model: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModel.from_pretrained(model).to(self._device)


    def _encode(self, text: str) -> np.ndarray:
        tokens = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        tokens = {key: val.to(self._device) for key, val in tokens.items()}
        with torch.no_grad():
            output = self._model(**tokens)

        embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return embedding


    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [self._model.embed_documents(doc) for doc in documents]


    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)


if __name__ == "__main__":
    pass
