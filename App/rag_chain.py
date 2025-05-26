import os
import yaml
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama  # or use your LLM loader
from langchain_community.vectorstores import Chroma
from app.llm_chain import load_llm
llm = load_llm(provider="ollama", model_name="mistral")


# Load environment variables
load_dotenv()

# Load settings from config file
with open(os.path.join("configs", "settings.yaml"), "r") as file:
    config = yaml.safe_load(file)

# Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=config.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    model_kwargs={"device": "cpu"}  # use "cuda" if GPU available
)

# Initialize LLM (Using Ollama for local models)
llm = ChatOllama(
    model=config.get("LLM_MODEL", "mistral"),
    temperature=0.2
)

# Function: Search + Generate Answer
def generate_answer(query: str, k: int = 3) -> str:
    """Answer a question using retrieved policy document chunks."""
    relevant_docs = search_policy_chunks(query, k, embeddings)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant answering questions based on policy documents."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ])
    
    return response.content

# Vector Search Function using Chroma
def search_policy_chunks(query: str, k: int = 3, embeddings=None):
    """Search for top-k relevant policy chunks using Chroma vector store."""
    persist_directory = os.path.join("data", "vectorstore")  # path to saved Chroma index

    # Ensure the vectorstore exists
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Chroma vectorstore not found at: {persist_directory}")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    return vectorstore.similarity_search(query, k=k)
