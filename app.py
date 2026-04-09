import streamlit as st
import google.generativeai as genai

# Page Config
st.set_page_config(page_title="Shri Ram Janki Temple Assistant", page_icon="🛕")

# --- API KEY SETUP ---
# Securely get API Key from Streamlit Secrets
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please add your Google API Key to Streamlit Secrets!")
    st.stop()

genai.configure(api_key=api_key)

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

Language:
- If asked in Hindi, reply in Hindi.
- If asked in English, reply in English.
- Use 'Hinglish' if appropriate.
"""

model = genai.GenerativeModel(
    model = genai.GenerativeModel('models/gemini-1.5-flash'),
    system_instruction=temple_context
)

# Chat Interface
st.title("🛕 Shri Ram Janki Temple AI")
st.write("Jai Shree Ram! Ask me anything about the temple.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = model.generate_content(prompt)
        st.markdown(response.text)
    
    st.session_state.messages.append({"role": "assistant", "content": response.text})
