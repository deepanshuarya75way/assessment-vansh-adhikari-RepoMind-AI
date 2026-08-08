import os
import stat
import shutil
import tempfile
from git import Repo
from langchain_community.document_loaders.parsers import LanguageParser
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


def clone_and_parse_repo(repo_url: str):
    temp_dir = tempfile.mkdtemp()

    try:
        # Depth=1 shallow clone to save bandwidth and speed up parsing
        Repo.clone_from(repo_url, temp_dir, depth=1)

        documents = []
        for root, dirs, files in os.walk(temp_dir):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, temp_dir)

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
                        chunks = splitter.create_documents(
                            texts=[content],
                            metadatas=[{"repo_url": repo_url, "file_path": rel_path}],
                        )
                        documents.extend(chunks)

                    except Exception:
                        continue

        return documents

    finally:
        # Safe cleanup for Windows OS
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, onerror=remove_readonly)
