import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# --- Configuration & Environment ---
st.set_page_config(page_title="Website RAG Bot", page_icon="🤖")
st.title("🤖 Chat with any Website")

os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# --- Session State Initialization ---
# This keeps the vector database in memory so it doesn't rebuild on every question
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: URL Input ---
with st.sidebar:
    st.header("1. Setup Data")
    url = st.text_input("Enter Website URL:", placeholder="https://example.com")
    process_btn = st.button("Process Website")

# --- Logic: Document Processing ---
if process_btn and url:
    with st.spinner("Fetching and indexing content..."):
        try:
            # 1. Load docs
            loader = WebBaseLoader(url)
            docs = loader.load()
            
            # 2. Split docs (Slightly increased chunk size for better context)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) 
            chunks = text_splitter.split_documents(docs)
            
            # 3. Embed and create vector store
            embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings_model)
            
            st.success(f"Indexed {len(chunks)} chunks! You can now ask questions.")
        except Exception as e:
            st.error(f"Error processing the URL: {e}")

# --- Main UI: Chat Interface ---
st.header("2. Chat")

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about the website..."):
    if not st.session_state.vectorstore:
        st.warning("Please process a URL in the sidebar first!")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.vectorstore.as_retriever()
                retrieved_docs = retriever.invoke(prompt)
                context = "\n".join([doc.page_content for doc in retrieved_docs])
                
                llm_prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {prompt}"
                
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                result = llm.invoke(llm_prompt)
                
                st.markdown(result.content)
                st.session_state.messages.append({"role": "assistant", "content": result.content})