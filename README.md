# llama-index-vector-stores-ledgermem

LlamaIndex vector store and retriever backed by [LedgerMem](https://github.com/ledgermem/ledgermem-python). LedgerMem handles the embeddings, hybrid search, and reranking for you — no separate vector DB required.

## Install

```bash
pip install llama-index-vector-stores-ledgermem
```

## Quickstart — index + query

```python
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from ledgermem import LedgerMem
from llama_index_vector_stores_ledgermem import LedgerMemVectorStore

client = LedgerMem(api_key="lm_...", workspace_id="ws_...")
vector_store = LedgerMemVectorStore(client=client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

docs = [Document(text="LangGraph is a graph-based agent framework.")]
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

response = index.as_query_engine().query("What is LangGraph?")
print(response)
```

## Standalone retriever (skip the index)

```python
from llama_index_vector_stores_ledgermem import LedgerMemRetriever

retriever = LedgerMemRetriever(client=client, similarity_top_k=8)
nodes = retriever.retrieve("everything I've said about LangGraph")
for node in nodes:
    print(node.score, node.get_content())
```

## License

MIT — see [LICENSE](./LICENSE).
