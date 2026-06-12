import streamlit as st
import rag_backend

st.set_page_config(page_title="RAG Knowledge Chatbot", layout="centered")
st.title("💬 Advanced Knowledge Base Chatbot")

# 1. Sidebar for API Token Authentication Secure Entry
with st.sidebar:
    st.header("Settings")
    user_key = st.text_input("Enter OpenAI API Key:", type="password")
    st.info("💡 Put your documents inside the `tech_kb` folder in your PyCharm project directory.")

if not user_key:
    st.warning("🔑 Please enter your OpenAI API Key in the sidebar to initialize the chatbot engine.")
else:
    # Use st.cache_resource so the pipeline runs only ONCE when the key is provided
    @st.cache_resource(show_spinner=False)
    def load_engine(api_key):
        return rag_backend.get_chat_engine(api_key)


    try:
        with st.spinner("Building vector index and loading embedding models..."):
            chat_engine = load_engine(user_key)

        # 2. Setup conversational layout session states
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant",
                                          "content": "Chatbot initialized successfully! Ask me anything about your documents."}]

        # Display history message bubbles
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # 3. Handle live incoming prompt inputs
        if user_query := st.chat_input("What would you like to know?"):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Searching files..."):
                    response = chat_engine.chat(user_query)
                    st.write(response.response)

                    # Optional: Expandable citation inspection utility
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        with st.expander("📚 View Document Citations"):
                            for node in response.source_nodes:
                                filename = node.metadata.get('file_name', 'Unknown Source')
                                st.markdown(f"**File:** `{filename}`")
                                st.caption(f"_{node.text[:180]}..._")

            st.session_state.messages.append({"role": "assistant", "content": response.response})

    except Exception as e:
        st.error(f"An unexpected initialization error occurred: {str(e)}")