import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Live Temple Assistant", page_icon="🛕")

# --- API SETUP ---
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("Please add GOOGLE_API_KEY to Streamlit Secrets!")
    st.stop()

# --- STEP 1: SCRAPE & PROCESS WEBSITE ---
@st.cache_resource
def prepare_website_data(url):
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        
        # New import path for text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(data)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
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
    search_results = vector_db.similarity_search(user_query, k=3)
    context_text = "\n".join([doc.page_content for doc in search_results])
    
    prompt = f"""
    You are 'Mandir Sahayak', the official assistant for Shri Ram Janki Temple.
    Website Info: {context_text}
    
    User Question: {user_query}
    
    Instructions:
    - Use the info above to answer correctly (especially for timings).
    - If asked in Hindi, reply in Hindi.
    - Start with 'Jai Shree Ram'.
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# --- UI ---
st.title("🛕 Shri Ram Janki Temple AI")
st.write(f"Live data from: {website_url}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about timings..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if vector_db:
            with st.spinner("Reading website data..."):
                answer = get_ai_response(prompt, vector_db)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
