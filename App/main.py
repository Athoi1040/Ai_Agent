from app.llm_chain import load_llm

# Example usage: switch providers easily
llm = load_llm(provider="ollama", model_name="mistral")
# llm = load_llm(provider="openai", model_name="gpt-3.5-turbo")
# llm = load_llm(provider="gemini", model_name="gemini-pro")


from app.rag_chain import generate_answer

def main():
    print("🧠 AI-Powered Policy Navigator")
    print("Type your question or type 'exit' to quit.\n")

    while True:
        user_query = input("🔍 Ask: ")
        if user_query.lower() in ['exit', 'quit']:
            break
        response = generate_answer(user_query)
        print(f"\n📘 Answer:\n{response}\n")

if __name__ == "__main__":
    main()
