import streamlit as st
from transformers import pipeline
summarizer = pipeline("summarization")
user_input = st.text_input("Put text here", "")
if st.button('Summarize'):
  result = summarizer(user_input, max_length=130, min_length=30, do_sample=False)
  st.write(result)
