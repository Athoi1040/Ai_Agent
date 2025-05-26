import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Paths
PDF_DIR = "data/policy_samples/Government"
VECTORSTORE_DIR = "data/vectorstore"

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

def load_documents():
    """Load all text chunks from PDF files."""
    documents = []
    files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if not files:
        print("❌ No PDF files found in:", PDF_DIR)
        return []

    for file in files:
        path = os.path.join(PDF_DIR, file)
        try:
            loader = PyPDFLoader(path)
            chunks = loader.load()
            if chunks:
                documents.extend(chunks)
                print(f"✅ Loaded {len(chunks)} chunks from: {file}")
            else:
                print(f"⚠️ No text found in: {file}")
        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")

    return documents

def ingest():
    """Convert documents to vector embeddings and store in Chroma DB."""
    docs = load_documents()

    if not docs:
        print("❌ No documents to embed. Exiting.")
        return

    try:
        vectorstore = Chroma.from_documents(
            docs, embeddings, persist_directory=VECTORSTORE_DIR
        )
        vectorstore.persist()
        print(f"✅ Ingested {len(docs)} chunks into vector store.")
    except Exception as e:
        print(f"❌ Vectorstore ingestion failed: {e}")

if __name__ == "__main__":
    ingest()
