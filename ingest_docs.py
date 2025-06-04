import os
from langchain_community.document_loaders import WebBaseLoader
import json
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions

from datetime import timedelta

load_dotenv()
URL_LIST = "file_with_urls.json"
METADATA = "metadata.json"
OUT_FOLDER = "docs"

# Load environment variables
DB_CONN_STR = os.getenv("DB_CONN_STR")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_BUCKET = os.getenv("DB_BUCKET")
DB_SCOPE = os.getenv("DB_SCOPE")
DB_COLLECTION = os.getenv("DB_COLLECTION")
INDEX_NAME = os.getenv("INDEX_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def read_url_list(file_with_urls: str) -> list:
    """Read the list of URLs from a JSON file"""
    data = []
    try:
        with open(file_with_urls, "r") as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(e)
        return data


def save_document(url: str):
    """Save the text from the HTML page to a LangChain document"""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
    except Exception as e:
        print(e)
        return None
    return docs


# Fetch the docs
urls_to_parse = read_url_list(URL_LIST)
docs_to_embed = []

print("fetching documentation")
for index, url in tqdm(enumerate(urls_to_parse)):
    docs_to_embed.append(save_document(url["URL"]))

# Text Spliiter to split larger documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
print(f"{len(docs_to_embed)=}")


print("Connecting to couchbase...")
auth = PasswordAuthenticator(DB_USERNAME, DB_PASSWORD)
options = ClusterOptions(auth)
connect_string = DB_CONN_STR
cluster = Cluster(connect_string, options)

# Wait until the cluster is ready for use.
cluster.wait_until_ready(timedelta(seconds=5))

# Create the vector store object
vector_store = CouchbaseSearchVectorStore(
    embedding=embeddings,
    cluster=cluster,
    bucket_name=DB_BUCKET,
    scope_name=DB_SCOPE,
    collection_name=DB_COLLECTION,
    index_name=INDEX_NAME,
)

# save documents to vectorstore
print("Adding documents to vector store...")
for document in tqdm(docs_to_embed):
    try:
        if document:
            documents = text_splitter.split_documents(document)
            vector_store.add_documents(documents)
    except Exception as e:
        print(f"Document {document['output']} failed: {e}")
        continue
