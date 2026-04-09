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
        # Load website content
        loader = WebBaseLoader(url)
        data = loader.load()
        
        if not data:
            st.error("Website returned no data. Check if the URL is correct.")
            return None

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(data)
        
        # FIX: Updated to the latest stable embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error reading website: {e}")
        return None

# Initialize the data
website_url = "https://shriramjankitemple.com/"
vector_db = prepare_website_data(website_url)

# --- STEP 2: AI BRAIN ---
def get_ai_response(user_query, vector_db):
    # Search the website data
    search_results = vector_db.similarity_search(user_query, k=3)
    context_text = "\n".join([doc.page_content for doc in search_results])
    
    # Prompt with instructions for correct timings
    prompt = f"""
    You are 'Mandir Sahayak', the official assistant for Shri Ram Janki Temple.
    
    CONTEXT FROM WEBSITE:
    {context_text}
    
    USER QUESTION: {user_query}
    
    STRICT INSTRUCTIONS:
    1. Look at the WEBSITE DATA above for the answer.
    2. If the user asks for 'timings', look specifically for AM and PM times. 
    3. If you find multiple timings, explain them clearly.
    4. If asked in Hindi, reply in Hindi.
    5. Start with 'Jai Shree Ram'.
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# --- UI INTERFACE ---
st.title("🛕 Shri Ram Janki Temple AI")
st.write(f"Reading data from: {website_url}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about timings, location, or donations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if vector_db:
            with st.spinner("Searching website info..."):
                try:
                    answer = get_ai_response(prompt, vector_db)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"AI Error: {e}")
        else:
            st.error("The system could not read the website data. Please check the logs.")
