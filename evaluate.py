import json
import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_new_tokens=150, eos_token_id=2)

SYSTEM = (
    "You are a helpful assistant. Use only the context provided. "
    "If the answer is not in the context, say 'I don't know'.\n\n"
)

questions = []
with open("golden_questions.txt") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        questions.append({"question": parts[0].strip(), "expected": parts[1].strip()})

logs = []
for item in questions:
    query = item["question"]
    chunks = db.similarity_search(query, k=3)
    sources = [r.metadata["title"] for r in chunks]
    context = "\n".join(r.page_content for r in chunks)

    prompt = SYSTEM + f"Context:\n{context}\n\nQ: {query}\nA:"
    raw = llm(prompt)[0]["generated_text"]
    answer = raw[len(prompt):].strip().split("\nQ:")[0].strip()

    has_chunks = len(chunks) > 0
    success = len(answer) > 20 and "don't know" not in answer.lower()

    log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": query,
        "expected": item["expected"],
        "answer": answer,
        "answer_length": len(answer),
        "has_chunks": has_chunks,
        "sources": sources,
        "success": success,
        "correct": (item["expected"] == "known" and success) or (item["expected"] == "unknown" and not success),
    }
    logs.append(log)
    print(f"Q: {query}")
    print(f"A: {answer[:100]}")
    print(f"Status: {'✓' if log['correct'] else '✗'} (expected={item['expected']}, success={success})\n")

with open("logs.jsonl", "w") as f:
    for log in logs:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

correct = sum(1 for l in logs if l["correct"])
print(f"\nИтого: {correct}/{len(logs)} корректных ответов")
print("Лог сохранён в logs.jsonl")
