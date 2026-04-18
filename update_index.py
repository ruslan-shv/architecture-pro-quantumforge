import os
import json
import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

LOG_FILE = "update_log.jsonl"
STATE_FILE = "indexed_files.json"
KB_DIR = "knowledge_base"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def write_log(entry):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

state = load_state()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists("faiss_index"):
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    db = None

new_docs = []
for filename in os.listdir(KB_DIR):
    if not filename.endswith(".txt"):
        continue
    filepath = os.path.join(KB_DIR, filename)
    mtime = str(os.path.getmtime(filepath))
    if state.get(filename) == mtime:
        continue
    with open(filepath, encoding="utf-8") as f:
        text = f.read()
    new_docs.append(Document(
        page_content=text,
        metadata={"source": filename, "title": filename.replace(".txt", "").replace("_", " ")}
    ))
    state[filename] = mtime

started_at = datetime.datetime.now().isoformat()
new_chunks = 0
errors = None

try:
    if new_docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(new_docs)
        new_chunks = len(chunks)
        if db is None:
            db = FAISS.from_documents(chunks, embeddings)
        else:
            db.add_documents(chunks)
        db.save_local("faiss_index")
        save_state(state)
except Exception as e:
    errors = str(e)

write_log({
    "started_at": started_at,
    "finished_at": datetime.datetime.now().isoformat(),
    "new_files": len(new_docs),
    "new_chunks": new_chunks,
    "index_size": db.index.ntotal if db else 0,
    "errors": errors,
})
