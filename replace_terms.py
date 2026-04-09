import json
import os

with open("terms_map.json", encoding="utf-8") as f:
    TERMS_MAP = json.load(f)

os.makedirs("knowledge_base", exist_ok=True)

for filename in os.listdir("raw_knowledge"):
    with open(f"raw_knowledge/{filename}", encoding="utf-8") as f:
        text = f.read()

    for original, fake in TERMS_MAP.items():
        text = text.replace(original, fake)

    # переименовываем файл тоже
    title = filename.replace(".txt", "").replace("_", " ")
    fake_title = TERMS_MAP.get(title, title)
    out_filename = f"knowledge_base/{fake_title.replace(' ', '_')}.txt"

    with open(out_filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{filename} -> {out_filename}")

print(f"\nГотово. Файлов в knowledge_base: {len(os.listdir('knowledge_base'))}")
