import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup as Soup
import os
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Shri Ram Janki Temple AI", page_icon="🛕")

# --- API SETUP ---
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("Please add GOOGLE_API_KEY to Streamlit Secrets!")
    st.stop()

# --- DETECT MODEL ---
@st.cache_resource
def get_working_model():
    try:
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for pref in ['models/gemini-1.5-flash', 'models/gemini-pro']:
            if pref in available: return pref
        return available[0]
    except: return 'models/gemini-pro'

CHOSEN_MODEL = get_working_model()

# --- SCRAPE ENTIRE WEBSITE (RECURSIVE) ---
@st.cache_resource
def prepare_full_website_data(url):
    try:
        # RecursiveUrlLoader follows links on the page to find sub-pages
        # max_depth=2 ensures it gets the home page and one level of sub-pages
        loader = RecursiveUrlLoader(
            url=url, 
            max_depth=2, 
            extractor=lambda x: Soup(x, "html.parser").text
        )
        data = loader.load()
        
        if not data:
            st.error("No data found on the website.")
            return None
            
        # Split all text from all pages into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(data)
        
        # Use HuggingFace embeddings (local) to avoid Google 404 errors
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"Crawling Error: {e}")
        return None

website_url = "https://shriramjankitemple.com/"
# This might take a few seconds longer as it's reading multiple pages
with st.spinner("Reading entire website and sub-pages..."):
    vector_db = prepare_full_website_data(website_url)

# --- AI LOGIC ---
def get_ai_response(user_query, vector_db):
    # Search top 5 matches across all pages
    search_results = vector_db.similarity_search(user_query, k=5)
    context = "\n".join([doc.page_content for doc in search_results])
    
    prompt = f"""
    You are 'Mandir Sahayak', the official assistant for Shri Ram Janki Temple, Village Kherawa, Sitapur, UP.
    
    WEBSITE DATA FROM ALL PAGES:
    {context}
    
    STRICT RULES:
    1. LOCATION: You are in Village Kherawa, Lakhimpur Road, Sitapur, Uttar Pradesh. 
    2. DO NOT mention Kannauj.
    3. TIMINGS: 5:00 AM to 11:00 AM and 4:00 PM to 9:00 PM.
    4. Provide details about donations, history, or activities if they are in the data.
    5. Always start with 'Jai Shree Ram'. Reply in the language of the user (Hindi/English).
    """

    model = genai.GenerativeModel(CHOSEN_MODEL)
    
    # Retry logic for Quota (429)
    for i in range(3):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e):
                time.sleep(10) # Wait 10 seconds if quota hit
                continue
            return f"Error: {e}"
    return "I am receiving too many questions right now. Please wait 30 seconds."

# --- UI ---
st.title("🛕 Shri Ram Janki Temple AI")
st.write(f"Connected to: {CHOSEN_MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about timings, donation, or any page on our site..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if vector_db:
            with st.spinner("Searching website..."):
                answer = get_ai_response(prompt, vector_db)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
