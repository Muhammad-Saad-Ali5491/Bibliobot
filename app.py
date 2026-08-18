import streamlit as st
from embed import get_model
from retrieve import load_db, get_reranker
from generate import ask

st.set_page_config(page_title="Bibliobot", page_icon="📖")
st.title("📖 Bibliobot")
st.caption("Chat with Hands-On Machine Learning")

@st.cache_resource
def load_everything():
    model = get_model()
    collection = load_db()
    reranker = get_reranker()
    return model, collection, reranker

model, collection, reranker = load_everything()

# --- Sidebar: DB inspector ---
with st.sidebar:
    st.header("🗄️ Vector DB Inspector")
    st.write(f"**Total chunks stored:** {collection.count()}")
    
    if st.button("Peek at random chunks"):
        sample = collection.get(limit=5, include=["documents", "metadatas"])
        for i in range(len(sample["ids"])):
            st.markdown(f"**ID {sample['ids'][i]} — Page {sample['metadatas'][i].get('page', '?')}**")
            st.caption(sample["documents"][i][:200] + "...")
            st.divider()
    
    lookup_id = st.text_input("Look up chunk by ID")
    if lookup_id:
        try:
            result = collection.get(ids=[lookup_id], include=["documents", "metadatas"])
            if result["ids"]:
                st.write(f"**Page:** {result['metadatas'][0].get('page', '?')}")
                st.write(result["documents"][0])
            else:
                st.warning("No chunk found with that ID.")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Main chat interface (unchanged) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("🔍 Retrieved chunks (top 5)"):
                for doc, meta, score in msg["sources"]:
                    st.markdown(f"**Page {meta.get('page', '?')} — relevance score: {score:.3f}**")
                    st.write(doc[:300] + "...")
                    st.divider()

query = st.chat_input("Ask something about the book")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask(query, model, collection, reranker, history=st.session_state.messages)
            st.write(answer)
            if sources:
                with st.expander("🔍 Retrieved chunks (top 5)"):
                    for doc, meta, score in sources:
                        st.markdown(f"**Page {meta.get('page', '?')} — relevance score: {score:.3f}**")
                        st.write(doc[:300] + "...")
                        st.divider()
    
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})