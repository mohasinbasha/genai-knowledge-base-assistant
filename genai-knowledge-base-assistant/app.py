import streamlit as st
from rag_pipeline import qa_chain

st.title("✈️ Airline GenAI Assistant")
query = st.text_input("Ask a question from the airline knowledge base:")

if query:
    answer = qa_chain.run(query)
    st.write("**Answer:**", answer)