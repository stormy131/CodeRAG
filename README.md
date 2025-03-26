# CodeRAG

Things to try out:
- Hybrid search
    - Combine __semantic__ and __keywrod based__ search together
    - BM25 + builin search
- Query expansion
    - integrate LLM calling for generating a __refined prompt__
    - "where is the auth habdled" -> "find functions related to user's ..."
- Advanced embedding
    - Use BERT based encoders to create file embeddings
    - Requires aliignment / fine-tune ??
- Indexing algorithms
    - Hierarchical Navigable Small World (HNSW) index ?
- Graph based retrieval with GNNs ??