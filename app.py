import streamlit as st
from preprocessing import extract_text_from_pdf


st.markdown("# ChatPDF")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:

    # file_data = uploaded_file.getvalue()
    file_data = extract_text_from_pdf(uploaded_file)
    st.write(file_data)

query = st.text_input("Enter Question")