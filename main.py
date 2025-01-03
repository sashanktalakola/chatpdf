import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore

from preprocessing import extract_text_from_pdf, get_text_chunks

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_API_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
ASTRA_DB_KEYSPACE = os.environ["ASTRA_DB_KEYSPACE"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vstore = AstraDBVectorStore(
    collection_name="file_embeddings",
    embedding=embedding,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    namespace=ASTRA_DB_KEYSPACE,
)


pdf_path = "test-files/rp.pdf"
text = extract_text_from_pdf(pdf_path)
text_chunks = get_text_chunks(text)

vstore.add_texts(text_chunks)