from typing import Dict

from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions
from datetime import timedelta


from langchain_couchbase.vectorstores import CouchbaseVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableBranch,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_couchbase.chat_message_histories import CouchbaseChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage

import streamlit as st
import os


def parse_bool(value: str):
    """Parse boolean values from environment variables"""
    return value.lower() in ("yes", "true", "t", "1")


def check_environment_variable(variable_name):
    """Check if environment variable is set"""
    if variable_name not in os.environ:
        st.error(
            f"{variable_name} environment variable is not set. Please add it to the secrets.toml file"
        )
        st.stop()


@st.cache_resource(show_spinner="Connecting to Couchbase")
def connect_to_couchbase(connection_string, db_username, db_password):
    """Connect to Couchbase cluster"""

    auth = PasswordAuthenticator(db_username, db_password)
    options = ClusterOptions(auth)
    connect_string = connection_string
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster


@st.cache_resource(show_spinner="Connecting to Vector Store")
def get_vector_store(
    _cluster,
    db_bucket,
    db_scope,
    db_collection,
    _embedding,
    index_name,
) -> CouchbaseVectorStore:
    """Return the Couchbase vector store"""
    vector_store = CouchbaseVectorStore(
        cluster=_cluster,
        bucket_name=db_bucket,
        scope_name=db_scope,
        collection_name=db_collection,
        embedding=_embedding,
        index_name=index_name,
    )
    return vector_store


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def parse_retriever_input(params: Dict):
    return params["messages"][-1].content


@st.cache_resource()
def get_chat_history(_cluster, db_bucket, db_scope, db_collection):
    """Store the chat history in Couchbase"""
    chat_message_history = CouchbaseChatMessageHistory(
        cluster=_cluster,
        bucket_name=db_bucket,
        scope_name=db_scope,
        collection_name=db_collection,
        session_id="chat-session",
    )
    return chat_message_history


if __name__ == "__main__":
    st.set_page_config(
        page_title="Chat with Couchbase Docs",
        page_icon="✨",
    )
    st.title("Chat with Couchbase Docs")

    # Check auth
    AUTH_ENABLED = parse_bool(os.getenv("AUTH_ENABLED", "False"))

    if not AUTH_ENABLED:
        st.session_state.auth = True
    else:
        # Authorization
        if "auth" not in st.session_state:
            st.session_state.auth = False

        AUTH = os.getenv("LOGIN_PASSWORD")
        check_environment_variable("LOGIN_PASSWORD")

        # Authentication
        user_pwd = st.text_input("Enter password", type="password")
        pwd_submit = st.button("Submit")

        if pwd_submit and user_pwd == AUTH:
            st.session_state.auth = True
        elif pwd_submit and user_pwd != AUTH:
            st.error("Incorrect password")

    if st.session_state.auth:
        # Load environment variables
        DB_CONN_STR = os.getenv("DB_CONN_STR")
        DB_USERNAME = os.getenv("DB_USERNAME")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_BUCKET = os.getenv("DB_BUCKET")
        DB_SCOPE = os.getenv("DB_SCOPE")
        DB_COLLECTION = os.getenv("DB_COLLECTION")
        INDEX_NAME = os.getenv("INDEX_NAME")
        EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        CONVERSATIONAL_CACHE_COLLECTION = os.getenv("CONVERSATIONAL_CACHE_COLLECTION")

        # Ensure that all environment variables are set
        check_environment_variable("OPENAI_API_KEY")
        check_environment_variable("DB_CONN_STR")
        check_environment_variable("DB_USERNAME")
        check_environment_variable("DB_PASSWORD")
        check_environment_variable("DB_BUCKET")
        check_environment_variable("DB_SCOPE")
        check_environment_variable("DB_COLLECTION")
        check_environment_variable("CONVERSATIONAL_CACHE_COLLECTION")
        check_environment_variable("INDEX_NAME")
        check_environment_variable("LANGCHAIN_ENDPOINT")
        check_environment_variable("LANGCHAIN_API_KEY")

        # Setup Langsmith Client
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

        cluster = connect_to_couchbase(DB_CONN_STR, DB_USERNAME, DB_PASSWORD)

        # Fetch ingested document store
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        # Get the vector store
        vector_store = get_vector_store(
            cluster,
            DB_BUCKET,
            DB_SCOPE,
            DB_COLLECTION,
            embeddings,
            INDEX_NAME,
        )

        # Fetch documents from the vector store
        retriever = vector_store.as_retriever()

        # Chat history store for added context
        chat_history = get_chat_history(
            _cluster=cluster,
            db_bucket=DB_BUCKET,
            db_scope=DB_SCOPE,
            db_collection=CONVERSATIONAL_CACHE_COLLECTION,
        )

        # Prompt for answering questions with message history
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a chatbot that can answer questions related to Couchbase. Remember that you can only reply to questions related to Couchbase or Couchbase SDKs and follow this strictly. If the user question is not related to couchbase, simply return "I am sorry, I am afraid I can't answer that". 
                    If you cannot answer based on the context provided, respond with a generic answer.
                    Answer the question as truthfully as possible using the context below:
                    {context}""",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # Use OpenAI GPT-4o as the LLM for the RAG
        llm = ChatOpenAI(temperature=0, model="gpt-4o")

        st.session_state.messages = chat_history.messages

        # Handle messages for the UI
        if len(st.session_state.messages) == 0:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Hi, I'm a chatbot who can chat with the Couchbase Docs. How can I help you?",
                }
            )

        # Prompt to transform the message history into a single query with all details
        chat_history_summary_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="messages"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
                ),
            ]
        )

        # Chain to transform the input message history into a single query using LLM and pass to retriever
        query_transforming_retriever_chain = RunnableBranch(
            (
                lambda x: len(x.get("messages", [])) == 1,
                # If only one message, then we just pass that message's content to retriever
                (lambda x: x["messages"][-1].content) | retriever,
            ),
            # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
            chat_history_summary_prompt | llm | StrOutputParser() | retriever,
        ).with_config(run_name="chat_retriever_chain")

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            else:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Create a chain to insert relevant documents into prompt to LLM
        document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

        # Conversation chain with added context based on chat history
        conversational_retrieval_chain = RunnablePassthrough.assign(
            context=query_transforming_retriever_chain,
        ).assign(
            answer=document_chain,
        )

        clear_cache = st.button("Clear Chat Context")
        if clear_cache:
            chat_history.clear()
            st.session_state.messages = []
            st.rerun()

        # React to user input
        if question := st.chat_input("Questions related to Couchbase?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(question)

            # Add user message to chat context
            chat_history.add_user_message(question)

            # Add placeholder for streaming the response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                sources_placeholder = st.empty()

            full_response = {}

            # Stream the response from the RAG
            for chunk in conversational_retrieval_chain.stream(
                {"messages": chat_history.messages}
            ):
                for key in chunk.keys():
                    try:
                        full_response[key] += chunk[key]
                    except KeyError:
                        full_response[key] = chunk[key]

                if "answer" in full_response:
                    message_placeholder.markdown(full_response["answer"] + "▌")

            # Add source links to the chat window from the context
            source_links = set()
            source_link_string = ""
            for docs in full_response["context"]:
                source_links.add(docs.metadata["source"])
                source_link_string = "\n".join(list(source_links))

            sources_placeholder.markdown(f"Sources:\n{source_link_string}")

            # Add complete response to the chat window & message history
            message_placeholder.markdown(full_response["answer"])
            chat_history.add_ai_message(full_response["answer"])
            chat_history.add_ai_message("Sources: \n" + source_link_string)
