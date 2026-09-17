import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Hugging Face Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


def normalize_repo_url(url: str) -> str:
    """Standardizes repo URLs to prevent string mismatch issues with ChromaDB metadata."""
    if not url:
        return ""
    url = str(url).strip().rstrip('/')
    if url.endswith('.git'):
        url = url[:-4]
    return url.lower()


def get_vector_store():
    """Returns persistent Chroma vector database instance."""
    return Chroma(
        collection_name="github_codebase",
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR
    )


def index_repository_documents(documents, repo_url: str, report:dict):
    """Replaces any previously indexed chunks for the same repo, then stores the new code chunks into ChromaDB."""
    clean_url = normalize_repo_url(repo_url)
    raw_url = str(repo_url).strip().rstrip('/')
    if raw_url.endswith('.git'):
        raw_url = raw_url[:-4]

    vector_store = get_vector_store()

    if report["is_incremental"]:
        for deleted_path in report["deleted_paths"]:
            try:
                vector_store,delete(
                    where={
                        "and":[
                            {"repo_url":clean_url},
                            {"file_path":deleted_path}
                        ]
                    }
                )
            except Exception as e:
                print(f"error pruning {deleted_path}:{e}")

    else:
        vector_store.delete(where={"repo_url"}=clean_url)
        return{
            "status":"success"
            "is_incremental":report["is_incremental"]
            "added_files":report["added_count"]
            "modified_files":report["modified_count"]
            "deleted_files":report["removed_count"]
        }

    # Delete stale chunks stored under any URL variation of this repo
    for url_form in {clean_url, raw_url}:
        try:
            vector_store.delete(where={"repo_url": url_form})
        except Exception as e:
            logger.warning(f"Note during chunk deletion: {e}")

    # Ensure every document has normalized metadata attached before insertion
    for doc in documents:
        doc.metadata["repo_url"] = clean_url
        if "clean_repo_url" not in doc.metadata:
            doc.metadata["clean_repo_url"] = clean_url

    vector_store.add_documents(documents)
    return len(documents)


def get_repo_chunk_count(repo_url: str) -> int:
    """Returns how many chunks are actually stored in Chroma for a repo."""
    clean_url = normalize_repo_url(repo_url)
    raw_url = str(repo_url).strip().rstrip('/')
    if raw_url.endswith('.git'):
        raw_url = raw_url[:-4]

    vector_store = get_vector_store()
    total = 0
    for url_form in {clean_url, raw_url}:
        try:
            total += len(vector_store.get(where={"repo_url": url_form})["ids"])
        except Exception:
            continue
    return total


def retrieve_code_context(query: str, repo_url: str, k: int = 4):
    """Retrieves top-k relevant code chunks filtered by normalized repo URL."""
    clean_url = normalize_repo_url(repo_url)
    raw_url = str(repo_url).strip().rstrip('/')
    if raw_url.endswith('.git'):
        raw_url = raw_url[:-4]

    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        query=query,
        k=k,
        filter={"repo_url": clean_url}
    )

    # Fallback 1: Search using raw URL if primary filter yielded 0 results
    if not results and raw_url != clean_url:
        results = vector_store.similarity_search(
            query=query,
            k=k,
            filter={"repo_url": raw_url}
        )

    # Fallback 2: Search without metadata filter if still 0, then verify repo matches
    if not results:
        unfiltered_results = vector_store.similarity_search(query=query, k=k)
        results = [
            doc for doc in unfiltered_results
            if normalize_repo_url(doc.metadata.get("repo_url", "")) == clean_url
        ]

    print(f"[RAG] Found {len(results)} chunks for repo: {clean_url}")
    return results