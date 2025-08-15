import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import re

# ตัวอย่าง service สำหรับ search
FAISS_INDEX_PATH = "faiss_db/faiss_index.bin"
METADATA_PATH = "faiss_db/faiss_metadata.json"

index = faiss.read_index(FAISS_INDEX_PATH)
# load metadata
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
print(f"Loaded FAISS index with {index.ntotal} vectors.")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def semantic_search_json(query, threshold=0.6):
    """
    ทำ semantic search ด้วย FAISS + cosine similarity
    และ merge chunks ตาม section/sub_header/header
    คืนค่าเป็น JSON structure พร้อมเนื้อหาเต็มของ section
    """
    # --- 1️⃣ search semantic ---
    query_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)
    sims, idxs = index.search(query_emb, index.ntotal)

    # เก็บ chunks ที่ผ่าน threshold
    results = []
    for sim, idx in zip(sims[0], idxs[0]):
        if sim > threshold:
            results.append({
                "score": float(sim),
                "chunk_id": metadata[idx]["chunk_id"],
                "header": metadata[idx]["header"],
                "secondary_header": metadata[idx]["secondary_header"],
                "sub_header": metadata[idx]["sub_header"],
                "page_number": metadata[idx]["page_number"],
                "content": metadata[idx]["content"]
            })

    # --- 2️⃣ merge chunks ตาม section ---
    merged = defaultdict(lambda: {"chunks": [], "content": "", "page_number": None})
    for r in results:
        key = (r["header"], r["secondary_header"], r["sub_header"])
        merged[key]["chunks"].append({
            "chunk_id": r["chunk_id"],
            "score": r["score"],
            "content": r["content"]
        })
        merged[key]["content"] += r["content"] + "\n"
        if merged[key]["page_number"] is None:
            merged[key]["page_number"] = r["page_number"]

    # --- 3️⃣ format final JSON ---
    final_results = []
    for (header, secondary_header, sub_header), data in merged.items():
        final_results.append({
            "header": header,
            "secondary_header": secondary_header,
            "sub_header": sub_header,
            "page_number": data["page_number"],
            "content": data["content"].strip(),
            "related_chunks": data["chunks"]
        })

    # sort by top score of each section
    final_results.sort(key=lambda x: max(c["score"] for c in x["related_chunks"]), reverse=True)
    return final_results
