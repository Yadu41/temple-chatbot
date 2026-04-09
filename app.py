import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    """Downloads website content and prepares a searchable database."""
    try:
        # Load the website content
        loader = WebBaseLoader(url)
        data = loader.load()
        
        # Split text into small chunks so AI can find specific facts
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(data)
        
        # Convert text chunks into 'Embeddings' (mathematical search vectors)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Could not read website: {e}")
        return None

# Initialize the searchable database from the URL
website_url = "https://shriramjankitemple.com/"
vector_db = prepare_website_data(website_url)

# --- STEP 2: AI BRAIN ---
def get_ai_response(user_query, vector_db):
    # 1. Search the website data for the most relevant text
    search_results = vector_db.similarity_search(user_query, k=3)
    context_text = "\n".join([doc.page_content for doc in search_results])
    
    # 2. Build a prompt that forces the AI to use the scraped text
    prompt = f"""
    You are 'Mandir Sahayak', the official assistant for Shri Ram Janki Temple.
    Use ONLY the following information from the website to answer the user:
    
    WEBSITE DATA:
    {context_text}
    
    USER QUESTION: {user_query}
    
    INSTRUCTIONS:
    - If the answer is in the data (like timings or location), give the exact answer.
    - If asked in Hindi, reply in Hindi.
    - If the info is NOT in the data, say "I'm sorry, that specific information is not updated on our website yet."
    - Be very respectful and start with 'Jai Shree Ram'.
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# --- UI INTERFACE ---
st.title("🛕 Shri Ram Janki Temple AI")
st.write(f"Reading live from: {website_url}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about timings, donations, or history..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if vector_db:
            with st.spinner("Searching website..."):
                answer = get_ai_response(prompt, vector_db)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error("I couldn't access the website data.")
