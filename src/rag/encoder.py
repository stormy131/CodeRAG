import torch
import numpy as np
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

from scheme.config import RAGConfig


# Load global configuration for the RAG system
config = RAGConfig()


class PretrainedEmbeddings(Embeddings):
    """
    PretrainedEmbeddings is a wrapper around transformer-based models for generating embeddings.
    It supports encoding text into dense vector representations for use in retrieval tasks.
    """

    def __init__(self, model: str):
        """
        Initializes the PretrainedEmbeddings class with a specified transformer model.

        Args:
            model (str): The name or path of the pretrained transformer model.
        """

        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModel.from_pretrained(model).to(self._device)

    def _encode(self, text: str) -> np.ndarray:
        """
        Encodes a single text string into a dense vector representation.

        Args:
            text (str): The input text to encode.

        Returns:
            np.ndarray: The dense vector representation of the input text.
        """

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

        # Compute the mean of the last hidden state to get the embedding
        embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Encodes a list of documents into dense vector representations.

        Args:
            documents (list[str]): List of document strings to encode.

        Returns:
            list[list[float]]: List of dense vector representations for each document.
        """
        return [self._encode(doc) for doc in documents]

    def embed_query(self, text: str) -> list[float]:
        """
        Encodes a query string into a dense vector representation.

        Args:
            text (str): The query string to encode.

        Returns:
            list[float]: The dense vector representation of the query.
        """
        return self._encode(text)


if __name__ == "__main__":
    embeddings = PretrainedEmbeddings("bert-base-uncased")
    print(embeddings.embed_query("What is RAG?"))
