from sentence_transformers import SentenceTransformer
import chromadb

def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def build_vector_db(chunks, model):
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection("book_chunks")
    
    texts = [c["text"] for c in chunks]
    pages = [c["page"] for c in chunks]
    
    print("Embedding chunks... this may take a minute")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    ids = [str(i) for i in range(len(texts))]
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"page": p} for p in pages]
    )
    return collection

if __name__ == "__main__":
    from load_book import load_pdf
    from chunker import chunk_text
    
    text, page_map = load_pdf(r"book\Hands-On_Machine_Learning_with_Scikit-Learn_Keras_and_Tensorflow_-_Aurelien_Geron.pdf")
    chunks = chunk_text(text, page_map)
    print(f"Total chunks: {len(chunks)}")
    
    model = get_model()
    build_vector_db(chunks, model)
    print("Vector DB built and saved to chroma_db/")