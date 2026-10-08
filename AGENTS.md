# AGENTS.md

Notes for contributors and coding agents working on this repo.

## Install

```sh
python -m pip install -r requirements-dev.txt   # app requirements + pytest + playwright
python -m playwright install chromium           # add --with-deps on a fresh Linux host
```

## Test tiers

| Tier | What it covers | Secrets | When it runs |
| --- | --- | --- | --- |
| 1. Streamlit + ingestion smoke | `tests/test_streamlit_smoke.py` boots `bot_ui.py` headlessly. It checks that the title, welcome message, "Clear Chat Context" button and chat input render. It asks a question and a follow-up (which covers both retriever branches), checks for the answer and the "Sources" link, clears the chat, and checks the `AUTH_ENABLED` password gate. `tests/test_ingest_smoke.py` runs `ingest_docs.py` and checks that fetched pages are chunked into the vector store and that an unreachable URL is skipped. Couchbase, OpenAI and web fetching are replaced by in-memory fakes (`tests/fakes.py`, `tests/smoke_app.py`). Every UI test fails on a `stException` element or on `Traceback` in the page or server log. | None | Every PR and push to `main` (`Streamlit + ingestion smoke (no secrets)` job). This is the required check. |
| 2. OpenAI provider smoke | `tests/test_openai_provider.py` calls the live OpenAI API with the exact models the app uses (read from `bot_ui.py` / `ingest_docs.py`, or `EMBEDDING_MODEL` if set). Embeddings must be 1536 finite floats, matching the Search index, and the chat model must stream a non-empty reply. Couchbase is not used. | `OPENAI_API_KEY` | `OpenAI provider smoke (optional, OPENAI_API_KEY)` job, on PRs from this repo, pushes to `main`, and manual `workflow_dispatch`. If the secret is not available (not configured, fork or Dependabot PRs), the job passes with a "skipped" notice. |
| 3. Live Couchbase RAG validation | The full app and ingestion against a real cluster, Search vector index and OpenAI. LangSmith is optional. | `OPENAI_API_KEY` and the Couchbase settings | Manual only (see below). |

Commands:

```sh
# Tier 1: no secrets needed (the tests strip any real credentials from the app's environment)
python -m pytest tests/test_streamlit_smoke.py tests/test_ingest_smoke.py -v

# Tier 2: skipped unless OPENAI_API_KEY is set in the environment
python -m pytest tests/test_openai_provider.py -v -rs
```

To run tier 2 on a Dependabot branch, start the `Smoke tests` workflow with
`workflow_dispatch` on that branch, or add `OPENAI_API_KEY` as a Dependabot
secret as well.

## Secrets

Names only. Never commit values. `.streamlit/secrets.toml`, `.env` and
`file_with_urls.json` are git-ignored; keep it that way.

- `OPENAI_API_KEY`: optional GitHub Actions secret for tier 2. Also needed to run the app and the ingestion.
- `DB_CONN_STR`, `DB_USERNAME`, `DB_PASSWORD`, `DB_BUCKET`, `DB_SCOPE`,
  `DB_COLLECTION`, `CONVERSATIONAL_CACHE_COLLECTION`, `INDEX_NAME`,
  `EMBEDDING_MODEL`: Couchbase and model settings, tier 3 only. They are not in CI.
- `LANGCHAIN_ENDPOINT`, `LANGCHAIN_API_KEY`: optional LangSmith tracing. The app
  turns tracing on only when `LANGCHAIN_API_KEY` is set, and runs without it.
  No test tier needs them, and LangSmith is not required for validation right now.
- `AUTH_ENABLED`, `LOGIN_PASSWORD`: optional password gate for the UI.

## Manual live validation for dependency-update PRs

The smoke tests are guardrails. They do not replace a live end-to-end run.
Every dependency-update PR (including Dependabot PRs for `langchain-*`,
`langsmith`, `couchbase`, `streamlit`, `beautifulsoup4`, etc.) needs a full
manual validation against real services before merge, with evidence on the PR:

1. **Dependency install**: `pip install -r requirements.txt` on the PR branch succeeds. Include the resolved versions of the bumped packages.
2. **Provider credentials**: name the secrets that were set (`OPENAI_API_KEY`,
   `DB_PASSWORD`, ...), never their values. Tier 2 passes, or explain why it was skipped.
3. **Couchbase connection and index state**: the cluster at `DB_CONN_STR` is
   reachable. The `DB_BUCKET`/`DB_SCOPE`/`DB_COLLECTION` and
   `CONVERSATIONAL_CACHE_COLLECTION` exist. The Search index `INDEX_NAME` exists,
   is ready, and has a 1536-dim `embedding` vector field (see README). Note the
   document count.
4. **Ingestion**: `python ingest_docs.py` with a small `file_with_urls.json`
   (list the URLs). It completes, and the collection's document count goes up.
5. **App startup**: `streamlit run bot_ui.py` starts with no errors (paste the console output).
6. **RAG query**: ask at least one question answered by the ingested pages, and
   one follow-up. Record each question, the observed answer, and the "Sources"
   links. Confirm that "Clear Chat Context" works.
7. **Screenshots or traces** of the chat when practical (a LangSmith trace link is fine).

If a live service is unavailable, write down the exact blocker on the PR. Do not
approve it based on the smoke tests alone.

### Known issue: answers lag one question behind

`CouchbaseChatMessageHistory` (langchain-couchbase 1.0.0 to 1.1.1) reads and
clears the chat history with SQL++ at the default `not_bounded` scan
consistency. Right after `add_user_message`, `chat_history.messages` usually
doesn't include the new question yet, so the RAG chain answers the previous
question (or none). The first answer after a clear is often "I am sorry, I am
afraid I can't answer that". This happens on `main` too; it is not caused by a
dependency bump. When you validate, compare against `main` and check whether
each answer matches its own question or the one before.
