"""No-secret smoke test for ingest_docs.py.

Runs the script for real with web fetching, OpenAI and Couchbase replaced by
fakes, and checks the fetched pages end up chunked in the vector store.
"""

import json
import runpy
from pathlib import Path

import dotenv
import langchain_community.document_loaders
import pytest
from langchain_core.documents import Document

import fakes

SCRIPT = Path(__file__).resolve().parent.parent / "ingest_docs.py"
GOOD_URL = "https://docs.couchbase.com/cloud/index.html"
BAD_URL = "https://docs.couchbase.com/unreachable.html"
PAGE_TEXT = "Couchbase Capella is a database as a service. " * 60  # ~2.7k chars


class FakeWebBaseLoader:
    def __init__(self, url):
        self.url = url

    def load(self):
        if self.url == BAD_URL:
            raise ConnectionError(f"cannot reach {self.url}")
        return [Document(page_content=PAGE_TEXT, metadata={"source": self.url})]


@pytest.fixture
def ingested(tmp_path, monkeypatch, capsys):
    for name in fakes.SECRET_NAMES:
        monkeypatch.setenv(name, "smoke-placeholder")
    fakes.install(monkeypatch.setattr)
    fakes.FakeVectorStore.instances.clear()
    # Don't pick up a developer's real .env.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(
        langchain_community.document_loaders, "WebBaseLoader", FakeWebBaseLoader
    )
    (tmp_path / "file_with_urls.json").write_text(
        json.dumps([{"URL": GOOD_URL}, {"URL": BAD_URL}])
    )
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(SCRIPT), run_name="__main__")
    [store] = fakes.FakeVectorStore.instances
    return store, capsys.readouterr().out


def test_ingest_adds_chunks(ingested):
    store, out = ingested
    assert store.kwargs["index_name"] == "smoke-placeholder"
    docs = list(store.store.values())
    # 1000-char chunks with 200 overlap, so a ~2.7k char page gives several.
    assert len(docs) >= 3
    assert all(len(d["text"]) <= 1000 for d in docs)
    assert {d["metadata"]["source"] for d in docs} == {GOOD_URL}
    assert all(len(d["vector"]) == 1536 for d in docs)
    assert f"cannot reach {BAD_URL}" in out
    assert "Traceback" not in out
