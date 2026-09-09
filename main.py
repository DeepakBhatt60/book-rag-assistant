from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore=Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm=ChatMistralAI(model="ministral-3b-2512")

# Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are a helpful AI assistant.
         Use ONLY the provided context to answer the question.
         iIf the answer is not present in the context,
         say:"I could not find the answer in the document."
         """),
         (
             "human",
             """Context:
{context}

Question:
{question}
             
             """
         )
    ]
)
print("Rag system created ")
print("press 0 to exit ")
while True:
    query=input("you: ")
    if(query=="0"):
        break
    docs=retriever.invoke(query)
    context="\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    })
    response=llm.invoke(final_prompt)
    print(f"\n AI:{response.content}")
