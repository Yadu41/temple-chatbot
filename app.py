import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Live Temple Assistant", page_icon="🛕")

# --- API SETUP ---
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("Please add GOOGLE_API_KEY to Streamlit Secrets!")
    st.stop()

# --- STEP 1: SCRAPE & PROCESS WEBSITE ---
@st.cache_resource
def prepare_website_data(url):
    try:
        # 1. Load website content
        loader = WebBaseLoader(url)
        data = loader.load()
        
        # 2. Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        docs = text_splitter.split_documents(data)
        
        # 3. FIX: Use HuggingFace instead of Google for embeddings to avoid 404 errors
        # This model is fast and runs on the Streamlit server itself
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error reading website: {e}")
        return None

# Initialize data
website_url = "https://shriramjankitemple.com/"
vector_db = prepare_website_data(website_url)

# --- STEP 2: AI BRAIN ---
def get_ai_response(user_query, vector_db):
    # Search the website data
    search_results = vector_db.similarity_search(user_query, k=3)
    context_text = "\n".join([doc.page_content for doc in search_results])
    
    # Prompt with specific instructions for temple timings
    prompt = f"""
    You are 'Mandir Sahayak', the official AI for Shri Ram Janki Temple.
    
    WEBSITE CONTEXT:
    {context_text}
    
    USER QUESTION: {user_query}
    
    STRICT RULES:
    1. Answer using ONLY the WEBSITE CONTEXT above.
    2. If the user asks for 'timings', look for the specific hours mentioned on the page (e.g., 5-11 AM, 4-9 PM).
    3. If asked in Hindi, reply in Hindi.
    4. Start with 'Jai Shree Ram'.
    5. If the answer isn't in the data, say "I don't have that specific information yet."
    """
    
    # Use Gemini 1.5 Flash for the chat response
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# --- UI INTERFACE ---
st.title("🛕 Shri Ram Janki Temple AI")
st.write(f"Live data from: {website_url}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about timings, donation, or history..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if vector_db:
            with st.spinner("Reading website..."):
                try:
                    answer = get_ai_response(prompt, vector_db)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"AI Response Error: {e}")
        else:
            st.error("Website data not available.")
