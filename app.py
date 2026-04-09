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

# Using the full model path which is more reliable
try:
    model = genai.GenerativeModel(
        model_name='models/gemini-1.5-flash',
        system_instruction=temple_context
    )
except Exception as e:
    st.error(f"Error initializing model: {e}")

# Chat Interface
st.title("🛕 Shri Ram Janki Temple AI")
st.write("Jai Shree Ram! Ask me anything about the temple.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Generate response
            response = model.generate_content(prompt)
            
            # Check if response has text (handles safety filters)
            if response.text:
                answer = response.text
            else:
                answer = "I'm sorry, I cannot answer that right now. Please ask something else about the temple."
                
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            # This catches the 'NotFound' or 'API' errors gracefully
            error_msg = f"I am having trouble connecting right now. (Error: {e})"
            st.error(error_msg)
