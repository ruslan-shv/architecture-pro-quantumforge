import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

docs = []
for filename in os.listdir("knowledge_base"):
    if not filename.endswith(".txt"):
        continue
    with open(f"knowledge_base/{filename}", encoding="utf-8") as f:
        text = f.read()
    docs.append(Document(
        page_content=text,
        metadata={"source": filename, "title": filename.replace(".txt", "").replace("_", " ")}
    ))

chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")
