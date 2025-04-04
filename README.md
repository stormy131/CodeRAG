# CodeRAG

CodeRAG is a Retrieval-Augmented Generation (RAG) system designed for efficient code search and question answering within a codebase. It combines semantic and keyword-based search techniques, leveraging advanced embeddings and indexing algorithms to enhance retrieval accuracy.

## Features
- **Hybrid Search**: Combines BM25 and dense vector search for improved retrieval.
- **Query Expansion**: Uses LLMs to refine user queries for better search results.
- **Advanced Embedding**: Supports BERT-based encoders for generating file embeddings.
- **Indexing Algorithms**: Implements HNSW for scalable and efficient indexing.
- **Graph-Based Retrieval**: Utilizes Graph Neural Networks (GNNs) for complex retrieval tasks.

## Code Overview
- **`src/main.py`**: Entry point for running the RAG system in QA or evaluation mode.
- **`src/rag/`**: Core RAG components, including retrievers, chunkers, and graph builders.
- **`src/scheme/`**: Configuration and data structures for RAG tasks.
- **`src/utils/`**: Utilities for data loading, chunking, and prompt generation.
- **`data/`**: Contains evaluation datasets, prompts, and configuration files.

## Getting Started
1. Clone the target repository using the `GIT_REPO` environment variable.
2. Run the system in QA mode to interactively query the codebase:
   ```bash
   python src/main.py --mode qa
3. Evaluate retrieval performance using the evaluation dataset:
   ```bash
   python src/main.py --mode evaluate

## Requirements
- Python 3.8+
- Dependencies listed in requirements.txt

## Future Enhancements
- Fine-tuning embeddings for domain-specific tasks.
- Expanding support for additional indexing and retrieval algorithms.
- Enhancing graph-based retrieval with advanced GNN models.