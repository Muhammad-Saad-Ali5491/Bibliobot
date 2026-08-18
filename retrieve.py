import numpy as np
import pickle
from embed import get_model

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_data():
    embeddings = np.load("embeddings.npy")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return embeddings, chunks

def retrieve(query, model, embeddings, chunks, top_k=5, threshold=0.3):
    query_vec = model.encode(query)
    
    scores = []
    for i, chunk_vec in enumerate(embeddings):
        score = cosine_similarity(query_vec, chunk_vec)
        scores.append((score, i))
    
    scores.sort(reverse=True)
    top_results = scores[:top_k]
    
    # filter out weak matches
    results = [(chunks[idx], score) for score, idx in top_results if score >= threshold]
    return results
   

if __name__ == "__main__":
    model = get_model()
    embeddings, chunks = load_data()
    
    query = "What is the difference between supervised and unsupervised learning?"
    results = retrieve(query, model, embeddings, chunks)
    
    for chunk, score in results:
        print(f"\n--- Score: {score:.4f} ---")
        print(chunk[:300])