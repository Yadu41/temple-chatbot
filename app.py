import streamlit as st
import google.generativeai as genai

# Page Config
st.set_page_config(page_title="Shri Ram Janki Temple Assistant", page_icon="🛕")

# --- API KEY SETUP ---
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please add your Google API Key to Streamlit Secrets!")
    st.stop()

genai.configure(api_key=api_key)

# --- AUTO-DETECT WORKING MODEL ---
@st.cache_resource
def get_working_model():
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    # Try to find these in order of preference
    preferred_models = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
    
    for model_path in preferred_models:
        if model_path in available_models:
            return model_path
    
    # If none of the above, just pick the first available one
    return available_models[0] if available_models else None

WORKING_MODEL_NAME = get_working_model()

# --- TEMPLE DATA ---
temple_context = """
You are 'Mandir Sahayak', the official AI assistant for Shri Ram Janki Temple, Village Kherawa, Sitapur, UP.
Website: https://shriramjankitemple.com/

Tone: Very respectful, humble, and devotional. 
Greetings: Start with 'Jai Shree Ram' or 'Namaste'.

Knowledge Base:
1. Deities: Lord Rama, Mata Sita, Lord Hanuman.
2. Aarti Timings: Morning 6:30 AM, Evening 7:30 PM.
3. Temple Hours: 6:00 AM to 9:00 PM.
4. Donations: Help with Gaushala and temple maintenance. Direct users to the 'Donation' page on the website.
5. Location: Village Kherawa, Lakhimpur Road, Sitapur, Uttar Pradesh.
"""

if WORKING_MODEL_NAME:
    model = genai.GenerativeModel(
        model_name=WORKING_MODEL_NAME,
        system_instruction=temple_context
    )
else:
    st.error("No compatible Gemini models found for this API key.")
    st.stop()

# --- UI ---
st.title("🛕 Shri Ram Janki Temple AI")
st.caption(f"Connected via: {WORKING_MODEL_NAME}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the temple..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = model.generate_content(prompt)
            if response.text:
                answer = response.text
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.warning("The AI could not generate a response. Try rephrasing.")
        except Exception as e:
            st.error(f"Error: {e}")
