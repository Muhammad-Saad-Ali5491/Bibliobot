from sentence_transformers import SentenceTransformer
import numpy as np
from load_book import load_pdf
from chunker import chunk_text

def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks, model):
    print("Embedding chunks... this may take a minute")
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

if __name__ == "__main__":
    text = load_pdf(r"book\Hands-On_Machine_Learning_with_Scikit-Learn_Keras_and_Tensorflow_-_Aurelien_Geron.pdf")
    chunks = chunk_text(text)
    
    model = get_model()
    embeddings = embed_chunks(chunks, model)
    
    print(f"Embeddings shape: {embeddings.shape}")   # should be (2209, 384)
    
    # save so we don't have to re-embed every time
    np.save("embeddings.npy", embeddings)
    
    # also save chunks so we can map embeddings back to text later
    import pickle
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print("Saved embeddings.npy and chunks.pkl")