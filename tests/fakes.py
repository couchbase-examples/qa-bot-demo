"""In-memory stand-ins for Couchbase and OpenAI used by the no-secret smoke tests.

Only the external services are faked. The app code, Streamlit and LangChain
run for real.
"""

import couchbase.cluster
import langchain_couchbase.chat_message_histories
import langchain_couchbase.vectorstores
import langchain_openai
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from langchain_core.vectorstores import InMemoryVectorStore

SOURCE_URL = "https://docs.couchbase.com/smoke-test.html"
SECRET_NAMES = (
    "OPENAI_API_KEY",
    "DB_CONN_STR",
    "DB_USERNAME",
    "DB_PASSWORD",
    "DB_BUCKET",
    "DB_SCOPE",
    "DB_COLLECTION",
    "CONVERSATIONAL_CACHE_COLLECTION",
    "INDEX_NAME",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_API_KEY",
    "LOGIN_PASSWORD",
)
# Optional in the app; the smoke tests run with these unset.
LANGSMITH_NAMES = ("LANGCHAIN_ENDPOINT", "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY")


class FakeCluster:
    def __init__(self, connection_string, options):
        pass

    def wait_until_ready(self, timeout):
        pass


class FakeChatMessageHistory(BaseChatMessageHistory):
    """Like CouchbaseChatMessageHistory, `messages` returns a fresh list."""

    def __init__(self, session_id, **kwargs):
        self._messages = []

    @property
    def messages(self):
        return list(self._messages)

    def add_message(self, message):
        self._messages.append(message)

    def clear(self):
        self._messages = []


class FakeVectorStore(InMemoryVectorStore):
    """Records what was added so the ingestion smoke can inspect it."""

    instances = []

    def __init__(self, embedding, **kwargs):
        super().__init__(embedding=embedding)
        self.kwargs = kwargs
        FakeVectorStore.instances.append(self)


def fake_embeddings(model, **kwargs):
    return DeterministicFakeEmbedding(size=1536)


def fake_chat_model(model, **kwargs):
    return FakeListChatModel(responses=[f"Smoke answer from {model}"])


def install(setattr=setattr):
    """Swap the external services for fakes. Pass monkeypatch.setattr to undo later."""
    setattr(couchbase.cluster, "Cluster", FakeCluster)
    setattr(langchain_couchbase.vectorstores, "CouchbaseSearchVectorStore", FakeVectorStore)
    setattr(
        langchain_couchbase.chat_message_histories,
        "CouchbaseChatMessageHistory",
        FakeChatMessageHistory,
    )
    setattr(langchain_openai, "OpenAIEmbeddings", fake_embeddings)
    setattr(langchain_openai, "ChatOpenAI", fake_chat_model)


def seeded_vector_store(embedding, **kwargs):
    """A vector store that already holds one ingested docs page."""
    store = FakeVectorStore(embedding, **kwargs)
    store.add_documents(
        [
            Document(
                page_content="Couchbase is a distributed NoSQL database.",
                metadata={"source": SOURCE_URL},
            )
        ]
    )
    return store
