import os
import re
from google import genai
from google.genai.errors import ClientError
from retrieve import retrieve
from math_agent import calculate

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

MODEL_NAME = "gemini-3.5-flash-lite"

GREETINGS = ["hi", "hello", "hey", "good morning", "good evening", "thanks", "thank you", "bye", "goodbye"]


def route(query):
    q = query.strip().lower()
    if q in GREETINGS:
        return "greeting"

    math_signals = ["calculate", "compute", "what is the result of", "solve for", "evaluate"]
    has_numbers_and_ops = bool(re.search(r'\d', q)) and any(op in q for op in ['+', '-', '*', '/', '%'])

    if any(sig in q for sig in math_signals) or has_numbers_and_ops:
        return "math"
    return "book_qa"


def safe_generate(contents):
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=contents)
        return response.text
    except ClientError as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ I've hit my usage limit right now. Please wait a bit and try again."
        return f"⚠️ Something went wrong: {e}"


def handle_greeting(query):
    text = safe_generate(f"""You are Bibliobot, a friendly assistant for the book "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow".
Respond warmly and briefly to: "{query}\"""")
    return text or "Hey! Ask me anything about the book.", []


def handle_math(query):
    extract_prompt = f"""Extract the pure mathematical expression from this question, using only numbers and operators (+ - * / ** sqrt etc), suitable for Python's eval().
Only output the expression, nothing else.

Question: {query}
Expression:"""
    expression = safe_generate(extract_prompt)

    if not expression:
        return "I couldn't identify a calculation in that question. Could you rephrase it?", []

    expression = expression.strip()
    result = calculate(expression)

    explain_prompt = f"""The question was: "{query}"
The calculated expression was: {expression}
The result is: {result}

Give a short, clear answer to the original question using this result."""
    final_text = safe_generate(explain_prompt)

    if not final_text:
        return f"The calculation result is: {result}", []

    return final_text.strip(), []


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

    text = safe_generate(prompt)
    return text or "I couldn't find that in the book.", results


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


if __name__ == "__main__":
    from embed import get_model
    from retrieve import load_db, get_reranker

    model = get_model()
    collection = load_db()
    reranker = get_reranker()

    query = "What is the curse of dimensionality?"
    answer, sources = ask(query, model, collection, reranker)
    print(answer)