# CodeRAG
CodeRAG is a Retrieval-Augmented Generation (RAG) system designed for efficient code search and question answering within a codebase. It combines semantic and keyword-based search techniques, leveraging advanced embeddings and indexing algorithms to enhance retrieval accuracy.

## Features
- **Hybrid Search**: Combines BM25 and dense vector search for improved retrieval.
- **Query Expansion**: Uses LLMs to refine user queries for better search results.
- **Advanced Embedding**: Supports BERT-based encoders for generating file embeddings.
- **AST-based c hunking strategy**: Breaks down code files into meaningful chunks using Abstract Syntax Tree parsing, preserving semantic relationships and code structure for more accurate retrieval.

## Project Structure Overview
```
.
├── src/
│   ├── main.py         # Entry point for RAG system (QA/evaluation)
│   ├── rag/            # Core RAG components
│   ├── scheme/         # Configuration and data structures
│   └── utils/          # Helper utilities
└── data/              # Evaluation dataset and logging
```

## Project Configuration
The following environment variables are used in the CodeRAG system. These variables are loaded into the configuration classes (`RAGConfig` and `PathConfig`) and are critical for the system's operation. The user shoud define them in `.env` file (in the repo's root) before using the system. Example contents can be seen in `.example.env`.

#### 1. `GOOGLE_API_KEY`
- **Description**: API key for accessing Google services, such as embeddings or other AI-related APIs.
- During development, Google's `text-embedding-004` model proved to be the best for this task, so I decided to stick to it.

#### 2. `API_KEY`
- **Description**: API key for accessing OpenRouter, a platform that provides access to various LLMs.
- Used to authenticate requests to OpenRouter for tasks such as query expansion and summarization. Ensure that this key is valid and corresponds to your OpenRouter account.

#### 3. `GIT_REPO`
- **Description**: URL of the target Git repository to be cloned and analyzed.
- Used in the `PathConfig` class to specify the repository that will be fetched and processed.
- The repository is cloned into the `code_repo_root` directory (`data/fetched`) for indexing and retrieval.

#### 4. `LLM`
- **Description**: Identifier for the language model to be used via OpenRouter.
  - Specifies the model used for query expansion and summarization.

#### 5. `ENCODER_MODEL`
- **Description**: Identifier for the HuggingFace embedding model to be used for generating dense vector representations of text.
- This variable is required **only for** `PretrainedEmbeddings` usage.

## Command-Line Interface
To start using the system it is sufficien to run `src/main.py` script in the root of the project. The script provides several command-line flags to control its operation:

| Flag | Description |
|------|-------------|
| `--mode` | Sets operation mode (`qa` for question answering, `evaluate` for testing) |
| `--verbose` | Enables detailed debug logging |
| `--expand_query` | Activates LLM query expansion |
| `--build_index` | Forces new index creation instead of using cache |

## Requirements
- Python 3.8+
- Dependencies listed in requirements.txt