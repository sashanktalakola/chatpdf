import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_API_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
ASTRA_DB_KEYSPACE = os.environ["ASTRA_DB_KEYSPACE"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


embedding = OpenAIEmbeddings()

vstore = AstraDBVectorStore(
    collection_name="file_embeddings",
    embedding=embedding,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    namespace=ASTRA_DB_KEYSPACE,
)

texts = [
    "Specify the embeddings model, database, and collection to use. If the collection does not exist, it is created automatically.",
    "Load a small dataset of philosophical quotes with the Python dataset module.",
    "Process metadata and convert to LangChain documents.",
    "Compute embeddings for each document and store in the database."
    ]

vstore.add_texts(texts)