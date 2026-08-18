import streamlit as st
from embed import get_model
from retrieve import load_data
from generate import ask

st.set_page_config(page_title="Bibliobot", page_icon="📖")
st.title("Bibliobot")
st.caption("Chat with Hands-On Machine Learning")

@st.cache_resource
def load_everything():
    model = get_model()
    embeddings, chunks = load_data()
    return model, embeddings, chunks

model, embeddings, chunks = load_everything()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask something about the book")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(query, model, embeddings, chunks)
            st.write(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})