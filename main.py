import os
from dotenv import load_dotenv
import warnings
import argparse
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAI
from langchain import hub
from langchain_core.runnables import RunnablePassthrough

from preprocessing import extract_text_from_pdf, get_text_chunks


parser = argparse.ArgumentParser(description="ChatPDF App")
parser.add_argument("--file", type=str)
parser.add_argument("--query", type=str)
args = parser.parse_args()

warnings.filterwarnings("ignore")

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


pdf_path = args.file
text = extract_text_from_pdf(pdf_path)
text_chunks = get_text_chunks(text)

vstore.add_texts(text_chunks)

retriever = vstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = OpenAI()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

query = args.query
response = chain.invoke(query)

print(response)
