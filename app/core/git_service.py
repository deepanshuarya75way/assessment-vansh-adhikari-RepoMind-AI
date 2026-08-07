import os
import shutil
import tempfile
from git import Repo
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language


SUPPORTED_EXTENSIONS = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".cpp": Language.CPP,
    ".go": Language.GO,
    ".java": Language.JAVA,
    ".rs": Language.RUST,
    ".c": Language.CPP,
    ".go": Language.GO,
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


def clone_and_parse_repo(repo_url: str):

    temp_dir= tempfile.mkdtemp()

    try:
        Repo.clone_from(repo_url, temp_dir)

        documents = []
        for root, dirs, files in os.walk(temp_dir):
             dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
             for file in files:
                ext = os.path.splitext(file)[1]
                if ext in SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, temp_dir)
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        splitter= RecursiveCharacterTextSplitter.from_language(
                            Language= SUPPORTED_EXTENSIONS[ext],
                            chunk_size= 1000,
                            chunk_overlap= 200
                        )
                        chunks= splitter.create_documents(
                            texts=[content],
                            metadatas= [{repo_url: repo_url, "file_path": rel_path}]
                        )
                        documents.extend(chunks)
                       
                    except Exception as e:
                        continue
    finally:
        shutil.rmtree(temp_dir)
