from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.config import settings

# Initialize Hugging Face Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def get_vector_store():
    return Chroma(
        collection_name="github_codebase",
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR
    )

def index_repository_documents(documents):
    """Stores code chunks into ChromaDB."""
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    return len(documents)

def retrieve_code_context(query: str, repo_url: str, k: int = 4):
    """Retrieves top-k relevant code chunks filtered by repo URL."""
    vector_store = get_vector_store()
    results = vector_store.similarity_search(
        query=query,
        k=k,
        filter={"repo_url": repo_url}
    )
    return results