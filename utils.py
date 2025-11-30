import os, json
from typing import List, Dict, Any
from PyPDF2 import PdfReader
import numpy as np
from sklearn.neighbors import NearestNeighbors
import openai
from tqdm import tqdm

# OpenRouter-compatible embedding model name (prefix with 'openai/')
EMBED_MODEL = "openai/text-embedding-3-small"

def read_pdfs(folder: str) -> List[Dict[str, Any]]:
    docs = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith('.pdf'):
            continue
        path = os.path.join(folder, fname)
        try:
            reader = PdfReader(path)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue
        for p in range(len(reader.pages)):
            try:
                page = reader.pages[p]
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            paras = [para.strip() for para in page_text.split('\n\n') if para.strip()]
            for i,para in enumerate(paras):
                docs.append({
                    "source": fname,
                    "page": p+1,
                    "chunk_index": i,
                    "text": para
                })
    return docs

def chunk_texts(docs: List[Dict[str,Any]], max_chars: int = 800) -> List[Dict[str,Any]]:
    new_docs = []
    for d in docs:
        txt = d['text']
        if len(txt) <= max_chars:
            new_docs.append(d)
        else:
            start = 0
            while start < len(txt):
                chunk = txt[start:start+max_chars]
                new = d.copy()
                new['text'] = chunk
                new_docs.append(new)
                start += max_chars
    return new_docs

def get_embeddings(texts: List[str]) -> np.ndarray:
    # Use OpenRouter via openai package. Read key from OPENROUTER_API_KEY environment variable.
    openai.api_key = os.getenv('OPENROUTER_API_KEY')
    
    openai.api_base = "https://openrouter.ai/api/v1"

    if openai.api_key is None:
        raise ValueError("OPENROUTER_API_KEY not set in environment. Set it before running.")
    embeddings = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i+16]
        resp = openai.Embedding.create(model=EMBED_MODEL, input=batch)
        for e in resp['data']:
            embeddings.append(e['embedding'])
    return np.array(embeddings, dtype='float32')

def build_store(docs: List[Dict[str,Any]], embeddings: np.ndarray, out_dir: str = 'store'):
    os.makedirs(out_dir, exist_ok=True)
    emb_path = os.path.join(out_dir, 'embeddings.npy')
    np.save(emb_path, embeddings)
    docs_path = os.path.join(out_dir, 'docs.json')
    with open(docs_path, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"Saved embeddings to {emb_path} and docs to {docs_path}")

class SimpleRetriever:
    def __init__(self, store_dir: str = 'store'):
        emb_path = os.path.join(store_dir, 'embeddings.npy')
        docs_path = os.path.join(store_dir, 'docs.json')
        if not os.path.exists(emb_path) or not os.path.exists(docs_path):
            raise FileNotFoundError("Run ingest.py first to create the store directory.")
        self.emb = np.load(emb_path)
        with open(docs_path, 'r', encoding='utf-8') as f:
            self.docs = json.load(f)
        self.nn = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.nn.fit(self.emb)

    def retrieve(self, query_emb: List[float], k: int = 4):
        import numpy as np
        q = np.array(query_emb).reshape(1, -1)
        dists, idxs = self.nn.kneighbors(q, n_neighbors=k)
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            results.append({
                'score': float(1 - dist),
                'doc': self.docs[idx]
            })
        return results
