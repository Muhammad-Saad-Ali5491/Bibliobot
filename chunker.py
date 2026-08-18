def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    from load_book import load_pdf
    
    text = load_pdf(r"book\Hands-On_Machine_Learning_with_Scikit-Learn_Keras_and_Tensorflow_-_Aurelien_Geron.pdf")
    chunks = chunk_text(text)
    
    print(f"Total chunks: {len(chunks)}")
    print("\n--- Chunk 50 ---")
    print(chunks[50])
    print("\n--- Chunk 51 ---")
    print(chunks[51])