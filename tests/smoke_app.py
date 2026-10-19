"""Run bot_ui.py with Couchbase and OpenAI swapped for in-memory fakes.

Streamlit executes this file instead of the app so the UI can be smoke tested
without a Couchbase cluster, an OPENAI_API_KEY or a LangSmith key. The
LangSmith variables are left unset, which checks that the app runs without them.
"""

import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import langchain_couchbase.vectorstores  # noqa: E402

import fakes  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "bot_ui.py"

# Placeholders only, so the app's environment checks pass.
for name in fakes.SECRET_NAMES:
    if name not in fakes.LANGSMITH_NAMES:
        os.environ[name] = "smoke-placeholder"

fakes.install()
langchain_couchbase.vectorstores.CouchbaseSearchVectorStore = fakes.seeded_vector_store

runpy.run_path(str(APP), run_name="__main__")
