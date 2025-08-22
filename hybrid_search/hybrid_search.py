# ✅ Full Example: Hybrid Search + MMR + MQR + LLaMA 3.1 (Fine-Tuned) via Ollama + ChromaDB

from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil
import os

# Step 1: Load your embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 2: Load and split documents with updated chunking
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 200

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

file = r"C:\Users\SreejaSaiLachannagar\Documents\propect_rules.pdf"
pdf_model = PyPDFLoader(file)
corpus= pdf_model.load()



# Dummy corpus load (replace with your real text corpus)
# You should load actual leasing/call documents here
# corpus = [Document(page_content="Sample leasing or call status content.")]
split_docs = splitter.split_documents(corpus)
print(split_docs)

# # Step 3: Connect to ChromaDB and store split documents
# persist_dir = "./chroma_db"
# if os.path.exists(persist_dir):
#     shutil.rmtree(persist_dir)

# db = Chroma.from_documents(split_docs, embedding_model, persist_directory=persist_dir)

# # Step 4: Define advanced retrievers with updated MMR params
# mmr_retriever = db.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 6, "lambda_mult": 0.5}
# )

# llm_for_mqr = Ollama(model="llama3-8b-finetuned")
# mqr_retriever = MultiQueryRetriever.from_llm(
#     retriever=mmr_retriever,
#     llm=llm_for_mqr
# )

# # Keyword filter fallback (manual hybrid layer)
# def keyword_filter(query):
#     results = db._collection.query(
#         where_document={"$contains": query},
#         n_results=5
#     )
#     return [
#         Document(page_content=d[0], metadata={"id": i})
#         for d, i in zip(results["documents"], results["ids"])
#         if d
#     ]

# def hybrid_retrieve(query):
#     vector_docs = mqr_retriever.invoke(query)
#     keyword_docs = keyword_filter(query)

#     seen = set()
#     merged_docs = []
#     for doc in vector_docs + keyword_docs:
#         if doc.page_content not in seen:
#             seen.add(doc.page_content)
#             merged_docs.append(doc)
#     return merged_docs[:8]

# # Step 5: Load LLaMA 3.1 fine-tuned model from Ollama
# llm = Ollama(model="llama3-8b-finetuned", temperature=0.0)

# # Step 6: Define custom prompt (RAG)
# prompt_template = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
#     You are a leasing assistant. Based on the following context, answer the user's question clearly and accurately.
    
#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """
# )

# # Step 7: Fallback QA chain (can be used optionally)
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=mmr_retriever,  # fallback vector-only retrieval
#     chain_type="stuff",
#     chain_type_kwargs={"prompt": prompt_template}
# )

# # Step 8: Use Hybrid Retriever with MMR + MQR + Keyword filtering
# def answer_question(query):
#     docs = hybrid_retrieve(query)
#     context = "\n\n".join([doc.page_content for doc in docs])
#     prompt = prompt_template.format(context=context, question=query)
#     return llm.invoke(prompt)

# # 🧪 Example
# if __name__ == "__main__":
#     query = "The user gave a token last week, what is their leasing status?"
#     answer = answer_question(query)
#     print("\nAnswer:\n", answer)
