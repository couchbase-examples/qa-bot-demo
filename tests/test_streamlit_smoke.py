"""No-secret smoke test: boot the Streamlit app and drive it with Playwright."""

import pytest
from playwright.sync_api import Page, expect, sync_playwright

from fakes import SOURCE_URL

WELCOME = "Hi, I'm a chatbot who can chat with the Couchbase Docs. How can I help you?"
CHAT_PLACEHOLDER = "Questions related to Couchbase?"
ANSWER = "Smoke answer from gpt-4o"
# smoke_app.py sets every secret, including LOGIN_PASSWORD, to this placeholder.
LOGIN_PASSWORD = "smoke-placeholder"

# The first run of a fresh server imports LangChain, which can take a while.
expect.set_options(timeout=30_000)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser
        browser.close()


def _open(browser, url):
    page = browser.new_page()
    page.set_default_timeout(30_000)
    page.goto(url)
    return page


def _assert_no_errors(page: Page, log_path):
    expect(page.get_by_test_id("stException")).to_have_count(0)
    assert "Traceback" not in page.inner_text("body")
    log = log_path.read_text()
    assert "Traceback" not in log, log


def _assert_chat_ui(page: Page):
    expect(page.get_by_role("heading", name="Chat with Couchbase Docs")).to_be_visible()
    expect(page.get_by_text(WELCOME)).to_be_visible()
    expect(page.get_by_role("button", name="Clear Chat Context")).to_be_visible()
    expect(page.get_by_placeholder(CHAT_PLACEHOLDER)).to_be_visible()


def _ask(page: Page, question):
    chat = page.get_by_placeholder(CHAT_PLACEHOLDER)
    chat.fill(question)
    chat.press("Enter")
    expect(page.get_by_text(question)).to_be_visible()


@pytest.fixture(scope="module")
def app(start_streamlit, browser):
    url, log_path = start_streamlit(AUTH_ENABLED="False")
    page = _open(browser, url)
    yield page, log_path
    page.close()


def test_app_renders(app):
    page, log_path = app
    _assert_chat_ui(page)
    _assert_no_errors(page, log_path)


def test_chat_round_trip(app):
    page, log_path = app
    # First question goes straight to the retriever.
    _ask(page, "What is Couchbase?")
    expect(page.get_by_text(ANSWER)).to_have_count(1)
    expect(page.get_by_text(SOURCE_URL)).to_have_count(1)
    _assert_no_errors(page, log_path)

    # A follow-up goes through the history-summarising branch first.
    _ask(page, "Tell me more")
    expect(page.get_by_text(ANSWER)).to_have_count(2)
    _assert_no_errors(page, log_path)


def test_clear_chat_context(app):
    page, log_path = app
    page.get_by_role("button", name="Clear Chat Context").click()
    expect(page.get_by_text(ANSWER)).to_have_count(0)
    expect(page.get_by_text(WELCOME)).to_be_visible()
    _assert_no_errors(page, log_path)


def test_password_gate(start_streamlit, browser):
    url, log_path = start_streamlit(AUTH_ENABLED="True")
    page = _open(browser, url)
    expect(page.get_by_role("heading", name="Chat with Couchbase Docs")).to_be_visible()
    expect(page.get_by_placeholder(CHAT_PLACEHOLDER)).to_have_count(0)

    password = page.get_by_label("Enter password")
    password.fill("wrong-password")
    page.get_by_role("button", name="Submit").click()
    expect(page.get_by_text("Incorrect password")).to_be_visible()
    expect(page.get_by_placeholder(CHAT_PLACEHOLDER)).to_have_count(0)

    password.fill(LOGIN_PASSWORD)
    page.get_by_role("button", name="Submit").click()
    _assert_chat_ui(page)
    _assert_no_errors(page, log_path)
    page.close()
