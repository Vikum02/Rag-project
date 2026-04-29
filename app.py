import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG System",
    page_icon="📄",
    layout="centered"
)

st.title("📄 RAG System")
st.caption("Upload a PDF and ask questions about it")