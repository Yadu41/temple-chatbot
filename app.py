import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup as Soup
import os
import time

# --- 1. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="Shri Ram Janki Temple AI", page_icon="🛕")

# --- 2. API SETUP ---
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("Missing GOOGLE_API_KEY in Streamlit Secrets!")
    st.stop()

# --- 3. AUTO-DETECT AVAILABLE GEMINI MODEL ---
@st.cache_resource
def get_model_name():
    try:
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for pref in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']:
            if pref in available: return pref
        return available[0]
    except: return 'models/gemini-pro'

SELECTED_MODEL = get_model_name()

# --- 4. LIVE WEBSITE CRAWLER (ALL PAGES) ---
@st.cache_resource
def build_live_knowledge_base(url):
    try:
        # RecursiveUrlLoader follows links on the site to find sub-pages (About, Donation, etc.)
        # max_depth=2 gets the homepage and everything linked from the menu
        loader = RecursiveUrlLoader(
            url=url, 
            max_depth=2, 
            extractor=lambda x: Soup(x, "html.parser").text,
            prevent_outside=True # Stays only on shriramjankitemple.com
        )
        
        st.info("Crawling website and sub-menus... please wait.")
        raw_data = loader.load()
        
        if not raw_data:
            return None

        # Break all pages into searchable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.split_documents(raw_data)
        
        # Create a local search engine (Using HuggingFace to avoid Google 404 errors)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(docs, embeddings)
        return vector_db
    except Exception as e:
        st.error(f"Website Scraping Error: {e}")
        return None

# Start Crawling
target_url = "https://shriramjankitemple.com/"
vector_db = build_live_knowledge_base(target_url)

# --- 5. AI GENERATION LOGIC ---
def ask_ai(user_query, db):
    # Find the most relevant 5 sections from the entire website
    docs = db.similarity_search(user_query, k=5)
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"""
    You are 'Mandir Sahayak', a helpful and devotional AI assistant for Shri Ram Janki Temple.
    
    YOUR KNOWLEDGE BASE (FROM LIVE WEBSITE):
    {context}
    
    USER QUESTION: {user_query}
    
    INSTRUCTIONS:
    1. Answer using ONLY the 'KNOWLEDGE BASE' provided above.
    2. If the data contains timings, location, or donation info, provide it exactly as written.
    3. If the answer is NOT in the knowledge base, say: "I'm sorry, I don't have that information. Please check our website or visit the temple."
    4. Start with a respectful greeting like 'Jai Shree Ram'.
    5. Language: Answer in the same language as the user (Hindi or English).
    """

    model = genai.GenerativeModel(SELECTED_MODEL)
    
    # Retry loop to handle 429 Quota errors
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e):
                time.sleep(8)
                continue
            return f"Error: {e}"
    return "The AI is currently busy. Please try again in a few seconds."

# --- 6. STREAMLIT UI ---
st.title("🛕 Shri Ram Janki Temple AI")
st.write(f"Source: {target_url} (All pages)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask about timings, donation, history..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if vector_db:
            with st.spinner("Searching website..."):
                answer = ask_ai(prompt, vector_db)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error("I couldn't read the website data. Please check the URL.")
