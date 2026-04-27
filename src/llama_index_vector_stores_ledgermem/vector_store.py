"""LlamaIndex ``BasePydanticVectorStore`` backed by LedgerMem."""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from ledgermem import LedgerMem
from pydantic import PrivateAttr


class LedgerMemVectorStore(BasePydanticVectorStore):
    """Persist LlamaIndex nodes in LedgerMem.

    LedgerMem manages embeddings server-side, so this store does not require a
    local embedding model and ignores ``query_embedding`` — it sends the raw
    ``query_str`` to LedgerMem's hybrid retriever instead.
    """

    stores_text: bool = True
    is_embedding_query: bool = False
    flat_metadata: bool = True

    _client: LedgerMem = PrivateAttr()

    def __init__(self, client: LedgerMem, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client

    @classmethod
    def class_name(cls) -> str:
        return "LedgerMemVectorStore"

    @property
    def client(self) -> LedgerMem:
        return self._client

    def add(self, nodes: list[BaseNode], **add_kwargs: Any) -> list[str]:
        ids: list[str] = []
        for node in nodes:
            content = node.get_content(metadata_mode=MetadataMode.NONE)
            metadata = dict(node.metadata or {})
            metadata.setdefault("source", "llamaindex")
            metadata["llamaindex_node_id"] = node.node_id
            result = self._client.add(content, metadata=metadata)
            memory_id = getattr(result, "id", None) or node.node_id
            ids.append(memory_id)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._client.delete(ref_doc_id)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        query_str = query.query_str or ""
        limit = query.similarity_top_k or 10
        response = self._client.search(query_str, limit=limit)
        hits = getattr(response, "hits", []) or []
        nodes: list[TextNode] = []
        ids: list[str] = []
        scores: list[float] = []
        for hit in hits:
            content = getattr(hit, "content", None) or getattr(hit, "text", "")
            metadata = dict(getattr(hit, "metadata", {}) or {})
            memory_id = getattr(hit, "id", None) or metadata.get("llamaindex_node_id", "")
            nodes.append(TextNode(text=content, id_=memory_id, metadata=metadata))
            ids.append(memory_id)
            score = getattr(hit, "score", None)
            if score is not None:
                scores.append(float(score))
        return VectorStoreQueryResult(nodes=nodes, similarities=scores or None, ids=ids)
