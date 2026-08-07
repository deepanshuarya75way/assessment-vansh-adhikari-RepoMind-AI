from groq import Groq
from app.config import settings
from app.core.embedding import retrieve_code_context

# Initialize Groq Client
groq_client = Groq(api_key=settings.GROQ_API_KEY)

def generate_answer_stream(query: str, repo_url: str, chat_history: list):
    # 1. Retrieve Context from ChromaDB
    docs = retrieve_code_context(query=query, repo_url=repo_url, k=4)
    
    context_str = "\n\n".join([
        f"--- File: {doc.metadata.get('file_path')} ---\n{doc.page_content}" 
        for doc in docs
    ])
    
    # 2. Build Contextual System Prompt
    system_prompt = (
        "You are an expert AI software engineer. Answer the user's question accurately using ONLY the codebase context provided below.\n"
        "If you refer to specific logic or structures, mention the file name from the context.\n"
        "If the information is not present in the code, state clearly that it is not available in the context.\n\n"
        f"### Codebase Context:\n{context_str}"
    )

    # 3. Assemble Messages Payload
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})

    # 4. Stream Response from Groq Engine
    response_stream = groq_client.chat.completions.create(
        model=settings.GROQ_MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=2048,
        stream=True,
    )

    for chunk in response_stream:
        delta_content = chunk.choices[0].delta.content
        if delta_content is not None:
            yield delta_content