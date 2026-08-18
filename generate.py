import os
import re
from google import genai
from retrieve import retrieve
from math_agent import calculate

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

GREETINGS = ["hi", "hello", "hey", "good morning", "good evening", "thanks", "thank you", "bye", "goodbye"]


def route(query):
    q = query.strip().lower()
    if q in GREETINGS:
        return "greeting"
    math_signals = ["calculate", "compute", "how many", "what is the result", "solve"]
    has_numbers_and_ops = bool(re.search(r'\d', q)) and any(op in q for op in ['+', '-', '*', '/', '%', 'sqrt'])
    if any(sig in q for sig in math_signals) or has_numbers_and_ops:
        return "math"
    return "book_qa"


def handle_greeting(query):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""You are Bibliobot, a friendly assistant for the book "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow".
Respond warmly and briefly to: "{query}\""""
    )
    return response.text, []


def handle_math(query):
    extract_prompt = f"""Extract the pure mathematical expression from this question, using only numbers and operators (+ - * / ** sqrt etc), suitable for Python's eval().
Only output the expression, nothing else.

Question: {query}
Expression:"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=extract_prompt)
    expression = response.text.strip()
    result = calculate(expression)
    
    explain_prompt = f"""The question was: "{query}"
The calculated expression was: {expression}
The result is: {result}

Give a short, clear answer to the original question using this result."""
    final_response = client.models.generate_content(model="gemini-2.5-flash", contents=explain_prompt)
    return final_response.text, []


def handle_book_qa(query, model, collection, reranker, history):
    results = retrieve(query, model, collection, reranker, top_k=5)
    
    if not results:
        return "I couldn't find that in the book.", []
    
    context_parts = []
    for doc, meta, score in results:
        page = meta.get("page", "unknown")
        context_parts.append(f"[Page {page}]\n{doc}")
    context = "\n\n---\n\n".join(context_parts)
    
    history_text = ""
    for turn in history[-4:]:
        history_text += f"{turn['role']}: {turn['content']}\n"
    
    prompt = f"""You are Bibliobot, a friendly assistant for the book "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow".

Recent conversation:
{history_text}

Answer the new question using ONLY the book context below. Each excerpt is labeled with its page number — mention the relevant page number(s) in your answer, like "(see page 42)".
Do not use outside knowledge. If the book context doesn't contain the answer, respond exactly with: "I couldn't find that in the book."

Book context:
{context}

New question: {query}

Answer:"""
    
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text, results   


def ask(query, model, collection, reranker, history=None):
    if history is None:
        history = []
    
    intent = route(query)
    
    if intent == "greeting":
        return handle_greeting(query)
    elif intent == "math":
        return handle_math(query)
    else:
        return handle_book_qa(query, model, collection, reranker, history)