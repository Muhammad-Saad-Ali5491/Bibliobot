import os
from google import genai
from embed import get_model
from retrieve import load_data, retrieve

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def build_prompt(query, results):
    context = "\n\n---\n\n".join([chunk for chunk, score in results])
    
    prompt = f"""You are a helpful assistant answering questions about a machine learning textbook.
Answer the question using ONLY the context below. Do not use any outside knowledge, even if you know the answer.
If the context does not contain the answer, respond exactly with: "I couldn't find that in the book."

Context:
{context}

Question: {query}

Answer:"""
    return prompt

def ask(query, model, embeddings, chunks):
    results = retrieve(query, model, embeddings, chunks, top_k=3)
    
    if not results:
        return "I couldn't find that in the book."
    
    prompt = build_prompt(query, results)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

if __name__ == "__main__":
    model = get_model()
    embeddings, chunks = load_data()
    
    query = "What is the difference between supervised and unsupervised learning?"   
    answer = ask(query, model, embeddings, chunks)
    print(answer)