import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
    )

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

query = "How much did Microsoft pay to acquire GitHub?"

#retriever = db.as_retriever(search_kwargs={"k": 5})

retriever = db.as_retriever(
     search_type="similarity_score_threshold",
     search_kwargs={
         "k": 5,
         "score_threshold": 0.3  
     }
    )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
print(" Context ")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

combined_input = f"""
Answer the question using ONLY the provided documents.

Question: {query}

Documents:
{chr(10).join([doc.page_content for doc in relevant_docs])}

Answer:
"""

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "RAG App"
    }
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
print("Full result:")
print(result)
#print("Content only:")
#print(result.content)
