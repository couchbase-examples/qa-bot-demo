"""Optional live OpenAI smoke test. Needs OPENAI_API_KEY; Couchbase is not used.

The model names are read from bot_ui.py and ingest_docs.py, so this checks
the exact models the app calls.
"""

import ast
import math
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is not set; skipping live OpenAI provider smoke test",
)


def _calls(path):
    return [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Call)]


def _chat_models():
    """The model= string literals passed to ChatOpenAI in bot_ui.py."""
    models = {
        kw.value.value
        for call in _calls(REPO / "bot_ui.py")
        if isinstance(call.func, ast.Name) and call.func.id == "ChatOpenAI"
        for kw in call.keywords
        if kw.arg == "model" and isinstance(kw.value, ast.Constant)
    }
    assert models, "no ChatOpenAI(model=...) call found in bot_ui.py"
    return sorted(models)


def _embedding_models():
    """EMBEDDING_MODEL if set, else the os.getenv default in each script."""
    if os.environ.get("EMBEDDING_MODEL"):
        return [os.environ["EMBEDDING_MODEL"]]
    models = {
        call.args[1].value
        for script in ("bot_ui.py", "ingest_docs.py")
        for call in _calls(REPO / script)
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "getenv"
        and len(call.args) == 2
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "EMBEDDING_MODEL"
    }
    assert models, "no os.getenv('EMBEDDING_MODEL', default) found"
    return sorted(models)


@pytest.mark.parametrize("model", _embedding_models())
def test_embeddings(model):
    from langchain_openai import OpenAIEmbeddings

    vector = OpenAIEmbeddings(model=model).embed_query("Couchbase")
    # The Search index in the README has a 1536-dim vector field.
    assert len(vector) == 1536
    assert all(isinstance(x, float) and math.isfinite(x) for x in vector)


@pytest.mark.parametrize("model", _chat_models())
def test_chat_model(model):
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0, max_retries=1)
    chunks = [c.content for c in llm.stream("Reply with the single word: pong")]
    assert "".join(chunks).strip()
