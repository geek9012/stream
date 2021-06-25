import streamlit as st
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
summarizer = pipeline("summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

user_input = st.text_input("Put text here", "")
if st.button('Summarize'):
  #result = summarizer(user_input, max_length=130, min_length=30, do_sample=False)
  inputs = tokenizer.encode("summarize: " + user_input, return_tensors="pt", max_length=512, truncation=True)
  outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
  st.write(tokenizer.decode(outputs[0]))
