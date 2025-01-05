from langchain_openai import OpenAI
from langchain import hub
from langchain_core.runnables import RunnablePassthrough


def rag_pipeline(query, vstore):

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

    response = chain.invoke(query)

    return response