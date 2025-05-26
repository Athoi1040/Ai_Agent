import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain.vectorstores import FAISS
from app.ingest import load_documents, chunk_documents
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.ingest import load_policy_docs
from app.embed import embed_documents 
import yaml

with open("configs/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

def embed_and_store():
    docs = load_policy_docs(config["POLICY_DIR"])
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = [chunk for doc in docs for chunk in splitter.split_text(doc)]

    embeddings = OpenAIEmbeddings(model_name=config["EMBEDDING_MODEL"])
    vectordb = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=config["CHROMA_DB_DIR"])
    vectordb.persist()
    print("Embedding complete and stored.")

if __name__ == "__main__":
    embed_and_store()