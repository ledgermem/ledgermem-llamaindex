"""LlamaIndex ``BasePydanticVectorStore`` backed by Mnemo."""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from getmnemo import Mnemo
from pydantic import PrivateAttr


class MnemoVectorStore(BasePydanticVectorStore):
    """Persist LlamaIndex nodes in Mnemo.

    Mnemo manages embeddings server-side, so this store does not require a
    local embedding model and ignores ``query_embedding`` — it sends the raw
    ``query_str`` to Mnemo's hybrid retriever instead.
    """

    stores_text: bool = True
    is_embedding_query: bool = False
    flat_metadata: bool = True

    _client: Mnemo = PrivateAttr()

    def __init__(self, client: Mnemo, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client

    @classmethod
    def class_name(cls) -> str:
        return "MnemoVectorStore"

    @property
    def client(self) -> Mnemo:
        return self._client

    def add(self, nodes: list[BaseNode], **add_kwargs: Any) -> list[str]:
        ids: list[str] = []
        for node in nodes:
            content = node.get_content(metadata_mode=MetadataMode.NONE)
            metadata = dict(node.metadata or {})
            metadata.setdefault("source", "llamaindex")
            metadata["llamaindex_node_id"] = node.node_id
            # Persist the document/ref-doc id too so delete(ref_doc_id) can
            # find every chunk that belongs to a parent document.
            ref_doc_id = getattr(node, "ref_doc_id", None) or node.node_id
            metadata["llamaindex_ref_doc_id"] = ref_doc_id
            result = self._client.add(content, metadata=metadata)
            memory_id = getattr(result, "id", None) or node.node_id
            ids.append(memory_id)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        # The argument is a LlamaIndex *ref doc id*, not a Mnemo memory
        # id. Calling client.delete(ref_doc_id) directly was a no-op (or a
        # 404) because Mnemo keys by its own server-issued id. Walk the
        # workspace and delete every chunk whose stored metadata matches.
        cursor: str | None = None
        ids_to_delete: list[str] = []
        while True:
            page = self._client.list(limit=100, cursor=cursor)
            items = getattr(page, "items", []) or getattr(page, "memories", []) or []
            for item in items:
                meta = getattr(item, "metadata", {}) or {}
                if (
                    meta.get("llamaindex_ref_doc_id") == ref_doc_id
                    or meta.get("llamaindex_node_id") == ref_doc_id
                ):
                    memory_id = getattr(item, "id", None)
                    if memory_id is not None:
                        ids_to_delete.append(memory_id)
            cursor = getattr(page, "next_cursor", None)
            if not cursor:
                break
        for memory_id in ids_to_delete:
            self._client.delete(memory_id)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        query_str = query.query_str or ""
        limit = query.similarity_top_k or 10
        # If the caller passed metadata filters we have to retrieve more
        # than `limit` rows because the filters are applied client-side
        # below; otherwise filtering trims the result set under the user's
        # requested top-k.
        filters = getattr(query, "filters", None)
        fetch_limit = max(limit * 4, 20) if filters else limit
        response = self._client.search(query_str, limit=fetch_limit)
        hits = getattr(response, "hits", []) or []
        nodes: list[TextNode] = []
        ids: list[str] = []
        scores: list[float] = []
        for hit in hits:
            content = getattr(hit, "content", None) or getattr(hit, "text", "")
            metadata = dict(getattr(hit, "metadata", {}) or {})
            if filters and not _matches_filters(metadata, filters):
                continue
            memory_id = getattr(hit, "id", None) or metadata.get("llamaindex_node_id", "")
            nodes.append(TextNode(text=content, id_=memory_id, metadata=metadata))
            ids.append(memory_id)
            score = getattr(hit, "score", None)
            # similarities and nodes must stay parallel — appending to
            # ``scores`` only when the hit has a score produced misaligned
            # arrays whenever the backend returned mixed (None, float)
            # results, so LlamaIndex would attribute the wrong score to
            # the wrong node downstream.
            scores.append(float(score) if score is not None else 0.0)
            if len(nodes) >= limit:
                break
        # Drop similarities entirely when not a single hit had a score,
        # otherwise keep them aligned 1:1 with ``nodes``.
        any_score = any(getattr(h, "score", None) is not None for h in hits[: len(nodes)])
        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=scores if any_score else None,
            ids=ids,
        )


def _matches_filters(metadata: dict[str, Any], filters: Any) -> bool:
    """Best-effort client-side check against LlamaIndex MetadataFilters.

    Supports the common ``key == value`` case across LlamaIndex versions
    where the API has shifted between ExactMatchFilter and MetadataFilter.
    Unknown operators conservatively return True so we don't drop hits the
    caller might have wanted.
    """
    raw = getattr(filters, "filters", None) or []
    condition = str(getattr(filters, "condition", "and")).lower()
    matches: list[bool] = []
    for f in raw:
        key = getattr(f, "key", None)
        value = getattr(f, "value", None)
        op = str(getattr(f, "operator", "==")).lower()
        if key is None:
            continue
        actual = metadata.get(key)
        if op in ("==", "eq", "operator.eq"):
            matches.append(actual == value)
        elif op in ("!=", "ne"):
            matches.append(actual != value)
        elif op == "in" and isinstance(value, (list, tuple, set)):
            matches.append(actual in value)
        else:
            matches.append(True)
    if not matches:
        return True
    return all(matches) if condition == "and" else any(matches)
