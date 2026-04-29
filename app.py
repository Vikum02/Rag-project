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

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file:
        if st.button("Upload & Ingest", type="primary"):
            with st.spinner("Ingesting document..."):
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                )
                if response.status_code == 200:
                    st.success(f"{uploaded_file.name} ingested successfully")
                else:
                    st.error("Upload failed. Check the terminal for errors.")

    st.divider()
    st.header("Ingested Documents")
    try:
        docs = requests.get(f"{API_URL}/docs-list").json()
        if docs["count"] == 0:
            st.info("No documents yet")
        else:
            for doc in docs["documents"]:
                st.markdown(f"- {doc}")
    except:
        st.warning("API not reachable")