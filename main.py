from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from ingest import ingest_pdf

app = FastAPI(title="RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "RAG system is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    save_path = f"data/{file.filename}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"Ingesting {save_path}...")
    ingest_pdf(save_path)

    return {"message": f"{file.filename} uploaded and ingested successfully"}