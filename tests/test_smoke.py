"""Smoke test: import + instantiate the LlamaIndex vector store with a mocked SDK."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _install_fake_ledgermem() -> None:
    if "ledgermem" in sys.modules:
        return
    fake = types.ModuleType("ledgermem")

    class LedgerMem:
        def __init__(self, *a, **k):
            pass

        def search(self, query, limit=5):
            return types.SimpleNamespace(hits=[])

        def add(self, content, metadata=None):
            return types.SimpleNamespace(id="mem_1")

        def delete(self, memory_id):
            return None

    class AsyncLedgerMem(LedgerMem):
        pass

    fake.LedgerMem = LedgerMem
    fake.AsyncLedgerMem = AsyncLedgerMem
    sys.modules["ledgermem"] = fake


_install_fake_ledgermem()

from llama_index.core.schema import TextNode  # noqa: E402
from llama_index.core.vector_stores.types import VectorStoreQuery  # noqa: E402
from llama_index_vector_stores_ledgermem import LedgerMemRetriever, LedgerMemVectorStore  # noqa: E402
from ledgermem import LedgerMem  # noqa: E402


def test_imports() -> None:
    assert LedgerMemVectorStore is not None
    assert LedgerMemRetriever is not None


def test_vector_store_add_and_query() -> None:
    client = LedgerMem()
    client.add = MagicMock(return_value=type("R", (), {"id": "mem_42"})())
    hit = type(
        "Hit",
        (),
        {"id": "mem_42", "content": "hello world", "metadata": {"foo": "bar"}, "score": 0.77},
    )()
    client.search = MagicMock(return_value=type("Resp", (), {"hits": [hit]})())

    store = LedgerMemVectorStore(client=client)
    ids = store.add([TextNode(text="hello world", metadata={"foo": "bar"})])
    assert ids == ["mem_42"]

    result = store.query(VectorStoreQuery(query_str="hello", similarity_top_k=3))
    assert result.nodes is not None and len(result.nodes) == 1
    assert result.nodes[0].get_content() == "hello world"
    assert result.similarities == [0.77]
