import streamlit as st
from groq import Groq

_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def chat(prompt: str) -> str:
    response = _client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content