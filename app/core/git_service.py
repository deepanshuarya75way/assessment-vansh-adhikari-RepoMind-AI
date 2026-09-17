import os
import stat
import shutil
import tempfile
from urllib.parse import urlparse
from git import Repo
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


SUPPORTED_EXTENSIONS = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".cpp": Language.CPP,
    ".go": Language.GO,
    ".java": Language.JAVA,
    ".rs": Language.RUST,
    ".c": Language.CPP,
    ".php": Language.PHP,
    ".html": Language.HTML,
    ".md": Language.MARKDOWN,
}

IGNORED_DIRS = {
    ".git",
    ".github",
    ".vscode",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    "dist",
    "build",
    "coverage",
}


def remove_readonly(func, path, exc_info):
    """Clear the read-only flag on files and directories to fix Windows permission errors during rmtree."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def normalize_repo_url(url: str) -> str:
    """Normalize GitHub repository URL for consistent vector metadata filtering."""
    if not url:
        return ""
    url = str(url).strip().rstrip('/')
    if url.endswith('.git'):
        url = url[:-4]
    return url.lower()


def get_git_file_operation(repo_path:str, previous_commit:str=):
    repo= Repo(repo_path)
    addded_files=set()
    modified_files=set()
    removed_files=set()

    try:
        diff_index= repo.commit(previous_commit).diff("HEAD")

        for each_file in diff_index:
            source_path=each_file.source_path.replace("\\","/") if each_file.source_path else None
            target_path=each_file.target_path.replace("\\","/") if each_file.target_path else None

            if each_file.new_file:
                added_files.add(target_path)
            elif each_file.removed_files:
                removed_files.add(source_path)
            elif each_file(source_path)!=each_file(target_path):
                removed_files.add(source_path)
                added_files.add(target_path)
            else:
                modified_files.add(target_path)


        return{
            "added":list(added_files),
            "modified":list(modified_files),
            "removed":list(removed_files)
        }
    except Exception as e:
        print(f"Git diff failed:{e}")
        return None


def extract_repo_info(repo_url: str):
    """Extract owner and repo name from GitHub URL for metadata matching."""
    clean_url = normalize_repo_url(repo_url)
    parsed = urlparse(clean_url)
    parts = [p for p in parsed.path.split('/') if p]
    
    owner = parts[-2] if len(parts) >= 2 else ""
    repo_name = parts[-1] if len(parts) >= 1 else ""
    return clean_url, owner, repo_name


def parse_file_to_chunk(file_path:str, rel_path:str, clean_url:str):
    for root, dirs, files in os.walk(temp_dir):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, temp_dir).replace("\\", "/")  # Standardize to POSIX paths

                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()

                        if not content.strip():
                            continue

                        splitter = RecursiveCharacterTextSplitter.from_language(
                            language=SUPPORTED_EXTENSIONS[ext],
                            chunk_size=1000,
                            chunk_overlap=200,
                        )

                        # Store multiple variations of metadata so ChromaDB metadata filters match regardless of implementation
                        chunk_metadata = {
                            "repo_url": clean_url,            # Lowercase normalized
                            "clean_repo_url": clean_url,      # Explicit normalized key
                            "raw_repo_url": original_url,     # Raw input URL
                            "repo_name": repo_name,            # Repository name
                            "repo_owner": owner,              # Owner handle
                            "file_path": rel_path,            # Standardized path (e.g. src/app.py)
                        }

                        chunks = splitter.create_documents(
                            texts=[content],
                            metadatas=[chunk_metadata],
                        )
                        documents.extend(chunks)
                        parsed_files_count += 1

                    except Exception as e:
                        print(f"[INGEST] Error reading {rel_path}: {e}")
                        continue

        print(f"[INGEST] Successfully processed {parsed_files_count} files into {len(documents)} chunks.")

        if not documents:
            print("[INGEST WARNING] 0 chunks were created! Check if target files match SUPPORTED_EXTENSIONS.")

        return documents

    except Exception as e:
        print(f"[INGEST ERROR] Failed during repo cloning/parsing: {e}")
        return []



def clone_and_parse_incremental(repo_url: str, previous_commit:str="HEAD~1"):
    clean_url, owner, repo_name = extract_repo_info(repo_url)
    original_url = str(repo_url).strip().rstrip('/')
    temp_dir = tempfile.mkdtemp()

    print(f"[INGEST] Cloning repository: {repo_url}...")
    print(f"[INGEST] Metadata filters -> clean_repo_url: '{clean_url}' | repo_name: '{repo_name}'")

    report = {
        "is_incremental": False,
        "added_count":0
        "modified_count":0
        "removed_count":0
    }

    try:
        # Depth=1 shallow clone to save bandwidth and speed up parsing
        Repo.clone_from(repo_url, temp_dir, depth=1)
        if diff_changes is not None:
            report["is_incremental"]= True
            report["added_counts"]= len(diff_changes["added"])
            report["modified_count"]= len(diff_changes["modified"])
            report["removed_count"]= len(diff_changes["removed"])

        documents = []
        parsed_files_count = 0
        for rel_path in files_to process:
            full_path = os.path.join(temp_dir,rel_path)
            if os.path.exist(full_path):
                chunks = parse_file_to_chuns(full_path,rel_path,clean_url)
        report["chunks"]= documents

        else:
            report["is_incremental"]=False
            documents= []
            for root , dirs, files in os.walk(temp_dir):
                dirs[:]= [d for d in dirs if d is not in IGNORED_DIRS]
                for file in files:
                    file_path= os.path.join(root,file)
                    rel_path= os.path.relpath(file_path, temp_dir).replace("\\", "/")
                    chunks= parse_file_to_chunk(file_path,rel_path, clean_url)
                    documents.extend(chunks)
            report["chunks"]= documents
       
    finally:
        # Safe cleanup for Windows OS
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, onerror=remove_readonly)