import chromadb
from sentence_transformers import CrossEncoder

def load_db():
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection("book_chunks")
    return collection

def get_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def retrieve(query, model, collection, reranker, top_k=5, fetch_k=15):
    query_vec = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=fetch_k
    )
    
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    
    if not docs:
        return []
    
    # rerank: cross-encoder scores (query, doc) pairs directly — more accurate than vector similarity alone
    pairs = [[query, doc] for doc in docs]
    rerank_scores = reranker.predict(pairs)
    
    combined = list(zip(docs, metadatas, rerank_scores))
    combined.sort(key=lambda x: x[2], reverse=True)
    
    top_results = combined[:top_k]
    # returns list of (chunk_text, {"page": N}, rerank_score)
    return top_results