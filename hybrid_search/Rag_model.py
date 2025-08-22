import os
import json
from dotenv import load_dotenv
from time import sleep
from datetime import datetime
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import Ollama  # <-- Switched to your local GGUF model
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Initialize the list to store document contents
combined_docs = []

# Load JSONL file content
with open(r"C:\Users\SreejaSaiLachannagar\Documents\succesful codes\fine tuning\omni_combined.jsonl", 'r') as file:
    for line in file:
        if line.strip():
            doc = json.loads(line)
            if 'page_content' in doc:
                combined_docs.append(doc['page_content'])

# Combine all into one string
combined_docs_str = "\n".join(combined_docs)

# Split the text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(combined_docs_str)
chunk_docs = [Document(page_content=chunk) for chunk in chunks]

# Embeddings & Chroma Setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_path = "C:\\Users\\SreejaSaiLachannagar\\Documents\\succesful codes\\chroma_db"
db = Chroma(collection_name="rules_vector_db", embedding_function=embeddings, persist_directory=db_path)

if len(db.get()['ids']) == 0:
    db.add_documents(chunk_docs)
    print(f"Database initialized with {len(db.get()['ids'])} documents")
else:
    print("Database already exists")

# Set up RetrievalQA with Ollama custom GGUF model
llm = Ollama(model="ggml-model-q5_k_m.gguf")  # Replace with your fine-tuned GGUF model name in Ollama
retriever = db.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# Chat loop
chat_history = []

while True:
    user_input = input("--> User: ")

    if user_input.lower().strip() == "exit":
        print("Exiting the chat...")
        break

    # Get answer
    response = qa_chain({"query": user_input})

    print(f"\n--> Assistant: {response['result']}\n")

    # Update chat history
    chat_history.append({'role': 'user', 'message': user_input.lower()})
    chat_history.append({'role': 'assistant', 'message': response['result'].lower()})

    # Save chat history to file
    try:
        with open(r'C:\Users\SreejaSaiLachannagar\Downloads\chats.txt', 'w') as file:
            json.dump(chat_history, file, indent=4)
    except Exception as e:
        print(f"Error saving chat history: {e}")

    # Add new conversations to vector DB
    chats = []
    for i in range(0, len(chat_history), 2):
        if i + 1 < len(chat_history):
            content = f"User: {chat_history[i]['message']}\nAssistant: {chat_history[i+1]['message']}"
            chats.append(Document(page_content=content, metadata={"turn": i}))

    db.add_documents(chats)
    sleep(1)
