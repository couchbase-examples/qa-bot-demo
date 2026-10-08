"""Run bot_ui.py with Couchbase and OpenAI swapped for in-memory fakes.

Streamlit executes this file instead of the app so the UI can be smoke tested
without a Couchbase cluster, an OPENAI_API_KEY or a LangSmith key.
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
    os.environ[name] = "smoke-placeholder"
# The app turns LangSmith tracing on by default; keep it off.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

fakes.install()
langchain_couchbase.vectorstores.CouchbaseSearchVectorStore = fakes.seeded_vector_store

runpy.run_path(str(APP), run_name="__main__")
