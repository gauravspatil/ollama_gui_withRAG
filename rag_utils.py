import requests
import numpy as np

OLLAMA_API_URL = "http://localhost:11434"

def split_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def get_embeddings(texts, model="nomic-embed-text", progress_bar=None, progress_popup=None):
    embeddings = []
    for idx, text in enumerate(texts):
        payload = {"model": model, "prompt": text}
        try:
            resp = requests.post(f"{OLLAMA_API_URL}/api/embeddings", json=payload)
            resp.raise_for_status()
            data = resp.json()
            emb = data.get("embedding")
            if emb:
                embeddings.append(emb)
            else:
                embeddings.append(None)
        except Exception:
            embeddings.append(None)
        if progress_bar is not None:
            progress_bar["value"] = idx + 1
            if progress_popup is not None:
                progress_popup.update()
    return embeddings

def get_query_embedding(query, model="nomic-embed-text", ensure_embedding_model=None):
    if ensure_embedding_model and not ensure_embedding_model(model):
        return None
    payload = {"model": model, "prompt": query}
    try:
        resp = requests.post(f"{OLLAMA_API_URL}/api/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("embedding")
    except Exception:
        return None

def retrieve_context(query, kb_chunks, kb_embeddings, top_k=2, threshold=0.05, get_query_embedding_func=None):
    if get_query_embedding_func is None:
        return "", []
    query_emb = get_query_embedding_func(query)
    if not query_emb:
        return "", []
    def cosine_sim(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    sims = []
    valid_indices = []
    for idx, emb in enumerate(kb_embeddings):
        if emb is not None:
            sims.append(cosine_sim(query_emb, emb))
            valid_indices.append(idx)
        else:
            sims.append(float('-inf'))
    sims_np = np.array(sims)
    filtered = [(i, sims_np[i]) for i in valid_indices if sims_np[i] > threshold]
    filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]
    context_chunks = [(i, kb_chunks[i]) for i, _ in filtered]
    context = '\n'.join([chunk for _, chunk in context_chunks])
    return context, context_chunks
