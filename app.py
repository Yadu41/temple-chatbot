import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Shri Ram Janki Temple AI", page_icon="🛕")

# --- API SETUP ---
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("Please add GOOGLE_API_KEY to Streamlit Secrets!")
    st.stop()

# --- AUTO-DETECT BEST WORKING MODEL ---
@st.cache_resource
def get_working_gemini_model():
    """Finds the best available model for your specific API key."""
    try:
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Prefer flash, then pro versions
        for preferred in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro', 'models/gemini-1.0-pro']:
            if preferred in available:
                return preferred
        return available[0] if available else None
    except:
        return 'models/gemini-pro' # Absolute fallback

CHOSEN_MODEL = get_working_gemini_model()

# --- STEP 1: SCRAPE WEBSITE ---
@st.cache_resource
def prepare_website_data(url):
    try:
        # User-agent header helps bypass some website blocks
        loader = WebBaseLoader(url)
        data = loader.load()
        
        # Split into smaller chunks so timings are easier to find
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(data)
        
        # Use local embeddings (HuggingFace) to avoid Google 404 errors
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error reading website: {e}")
        return None

website_url = "https://shriramjankitemple.com/"
vector_db = prepare_website_data(website_url)

# --- STEP 2: AI BRAIN ---
def get_ai_response(user_query, vector_db):
    # Search for top 5 matches to ensure we get the timings
    search_results = vector_db.similarity_search(user_query, k=5)
    context_text = "\n".join([doc.page_content for doc in search_results])
    
    prompt = f"""
    You are 'Mandir Sahayak', the assistant for Shri Ram Janki Temple.
    
    WEBSITE DATA:
    {context_text}
    
    USER QUESTION: {user_query}
    
    STRICT RULES:
    1. Look at WEBSITE DATA for timings. (Website says 5-11 AM and 4-9 PM).
    2. If someone asks for timings, reply with the EXACT hours found in the data.
    3. If asked in Hindi, reply in Hindi.
    4. Start with 'Jai Shree Ram'.
    5. Be humble and devotional.
    """
    
    model = genai.GenerativeModel(CHOSEN_MODEL)
    response = model.generate_content(prompt)
    return response.text

# --- UI ---
st.title("🛕 Shri Ram Janki Temple AI")
st.caption(f"Connected via: {CHOSEN_MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about timings, location, or donations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if vector_db:
            with st.spinner("Checking website..."):
                try:
                    answer = get_ai_response(prompt, vector_db)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"AI Error: {e}")
        else:
            st.error("Website data not available.")
