import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def load_llm(provider="ollama", model_name="mistral"):
    if provider == "ollama":
        return ChatOllama(model=model_name, temperature=0.2)
    elif provider == "openai":
        return ChatOpenAI(model=model_name, temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.2, google_api_key=os.getenv("GOOGLE_API_KEY"))
    else:
        raise ValueError("Unsupported LLM provider. Choose from: 'ollama', 'openai', 'gemini'")
