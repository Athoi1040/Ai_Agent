import yaml
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

with open("configs/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

embeddings = OpenAIEmbeddings(model=config["EMBEDDING_MODEL"])
vectordb = Chroma(persist_directory=config["CHROMA_DB_DIR"], embedding_function=embeddings)

def search_policy_chunks(query, k=3):
    results = vectordb.similarity_search(query, k=k)
    return results
