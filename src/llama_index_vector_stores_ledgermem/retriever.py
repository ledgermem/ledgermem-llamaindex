"""LlamaIndex retriever that talks to Mnemo directly (no index needed)."""

from __future__ import annotations

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from getmnemo import Mnemo


class MnemoRetriever(BaseRetriever):
    """A standalone LlamaIndex retriever backed by ``Mnemo.search``.

    Useful when you want LlamaIndex's query engines and response synthesis but
    do not want to maintain a separate vector index — Mnemo is the index.
    """

    def __init__(
        self,
        client: Mnemo,
        similarity_top_k: int = 5,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        self._client = client
        self._top_k = similarity_top_k
        super().__init__(callback_manager=callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        response = self._client.search(query_bundle.query_str, limit=self._top_k)
        results: list[NodeWithScore] = []
        for hit in getattr(response, "hits", []) or []:
            content = getattr(hit, "content", None) or getattr(hit, "text", "")
            metadata = dict(getattr(hit, "metadata", {}) or {})
            memory_id = getattr(hit, "id", None) or ""
            score = getattr(hit, "score", None)
            node = TextNode(text=content, id_=memory_id, metadata=metadata)
            results.append(NodeWithScore(node=node, score=float(score) if score is not None else None))
        return results
