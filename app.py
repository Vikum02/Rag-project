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

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            st.caption(f"Sources: {', '.join(msg['sources'])}")

if question := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": question}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.write(data["answer"])
                    st.caption(f"Sources: {', '.join(data['sources'])}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data["sources"]
                    })
                else:
                    st.error("Something went wrong. Try again.")
            except Exception as e:
                st.error(f"Could not reach the API: {e}")