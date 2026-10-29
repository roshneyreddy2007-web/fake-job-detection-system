import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

# Download required NLTK data safely
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

st.title("🚨 Fake Job Posting Detection")
st.write("Paste a job description below to check if it is real or fake.")

# User text input box
job_text = st.text_area("Job Description Text", height=250, placeholder="Paste job requirements, description, or title here...")

if st.button("Analyze Posting"):
    if not job_text.strip():
        st.warning("Please paste some text first!")
    else:
        # Simple rule-based logic placeholder for testing
        # Replace this with your actual model prediction if you have a saved .pkl file!
        st.info("Analyzing text patterns...")
        
        has_suspicious_words = any(word in job_text.lower() for word in ['wire transfer', 'urgent money', 'no experience required', 'earn fast'])
        
        if has_suspicious_words:
            st.error("⚠️ Warning: This posting looks highly suspicious or FAKE.")
        else:
            st.success("✅ Safe: This posting appears to be REAL.")
