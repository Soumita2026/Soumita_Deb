import streamlit as st
import rag_backend

st.set_page_config(page_title="Local RAG Chatbot", layout="centered")
st.title("💬 Local Knowledge Base Chatbot")
st.caption("🚀 Powered completely offline by local embeddings & Ollama")


@st.cache_resource(show_spinner=False)
def load_engine():
    return rag_backend.get_chat_engine()


try:
    if "chat_engine" not in st.session_state:
        with st.spinner("Initializing local RAG engine..."):
            st.session_state.chat_engine = load_engine()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hello! I am running completely locally. Ask me anything about the files in your `tech_kb` folder."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_query := st.chat_input("Ask a question about your documents:"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching local files..."):
                response = st.session_state.chat_engine.chat(user_query)
                st.write(response.response)

                if hasattr(response, 'source_nodes') and response.source_nodes:
                    with st.expander("📚 View Document Citations"):
                        for node in response.source_nodes:
                            filename = node.metadata.get('file_name', 'Unknown Source')
                            st.markdown(f"**File:** `{filename}`")
                            st.caption(f"_{node.text[:180]}..._")
                            st.markdown("---")

        st.session_state.messages.append({"role": "assistant", "content": response.response})

except Exception as e:
    st.error(f"Initialization Error: Ensure the Ollama app is running in your taskbar. Details: {str(e)}")