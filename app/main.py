import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.database import engine, Base, get_db, AsyncSessionLocal
from app.model import Repository, ChatSession, ChatMessage
from app.core.git_service import clone_and_parse_repo
from app.core.embedding import index_repository_documents
from app.core.rag import generate_answer_stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create database tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="GitHub Repo RAG Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request Schemas ---
class IngestRequest(BaseModel):
    repo_url: HttpUrl


class ChatRequest(BaseModel):
    session_id: str
    query: str


# --- Ingestion Background Task ---
async def process_repository_task(repo_id: str, repo_url: str, db_factory):
    async with db_factory() as db:
        try:
            # Offload synchronous heavy CPU/IO git parsing to a thread pool
            loop = asyncio.get_running_loop()
            docs = await loop.run_in_executor(None, clone_and_parse_repo, repo_url)
            
            # Offload ChromaDB embedding indexing to a thread pool
            await loop.run_in_executor(None, index_repository_documents, docs)

            # Update DB Status
            result = await db.execute(select(Repository).where(Repository.id == repo_id))
            repo = result.scalar_one_or_none()
            if repo:
                repo.status = "COMPLETED"
                repo.total_files = len(docs)
                await db.commit()
        except Exception as e:
            await db.rollback()
            result = await db.execute(select(Repository).where(Repository.id == repo_id))
            repo = result.scalar_one_or_none()
            if repo:
                repo.status = f"FAILED: {str(e)}"
                await db.commit()


# --- Endpoints ---
@app.post("/api/v1/repos/ingest")
async def ingest_repository(
    payload: IngestRequest, 
    background_tasks: BackgroundTasks, 
    db: AsyncSession = Depends(get_db)
):
    url_str = str(payload.repo_url)
    repo_name = url_str.rstrip("/").split("/")[-1]

    # Check if exists
    result = await db.execute(select(Repository).where(Repository.repo_url == url_str))
    existing_repo = result.scalar_one_or_none()
    
    if existing_repo and not existing_repo.status.startswith("FAILED"):
        return {
            "message": "Repository already indexed or processing", 
            "repo_id": existing_repo.id, 
            "status": existing_repo.status
        }

    # Create or update DB Entry
    if existing_repo:
        new_repo = existing_repo
        new_repo.status = "PROCESSING"
    else:
        new_repo = Repository(repo_url=url_str, repo_name=repo_name, status="PROCESSING")
        db.add(new_repo)

    await db.commit()
    await db.refresh(new_repo)

    # Trigger Background Ingestion Pipeline
    background_tasks.add_task(process_repository_task, new_repo.id, url_str, AsyncSessionLocal)

    return {"message": "Ingestion started", "repo_id": new_repo.id, "status": "PROCESSING"}


@app.post("/api/v1/chat/stream")
async def chat_stream(payload: ChatRequest, db: AsyncSession = Depends(get_db)):
    # Fetch Session & Repo Info
    result = await db.execute(select(ChatSession).where(ChatSession.id == payload.session_id))
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    repo_result = await db.execute(select(Repository).where(Repository.id == session.repo_id))
    repo = repo_result.scalar_one_or_none()

    if not repo:
        raise HTTPException(status_code=404, detail="Associated repository not found")

    # Get Chat History
    msg_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.timestamp.asc())
    )
    history_records = msg_result.scalars().all()
    chat_history = [{"role": msg.sender, "content": msg.content} for msg in history_records]

    # Save User Message to PostgreSQL
    user_msg = ChatMessage(session_id=session.id, sender="user", content=payload.query)
    db.add(user_msg)
    await db.commit()

    # Thread-safe Stream Response Generator Function
    async def event_generator():
        full_response = ""
        loop = asyncio.get_running_loop()

        try:
            # Wrap generator invocation in thread pool executor so it doesn't block the ASGI loop
            def fetch_stream():
                return list(generate_answer_stream(payload.query, repo.repo_url, chat_history))

            chunks = await loop.run_in_executor(None, fetch_stream)

            for chunk in chunks:
                full_response += chunk
                yield chunk
                await asyncio.sleep(0.01)  # Yield control back to event loop for smooth network delivery
                
        except Exception as e:
            err_msg = f"\n[Streaming Error: {str(e)}]\n"
            full_response += err_msg
            yield err_msg

        finally:
            # Save Assistant Response to DB after streaming completes
            async with AsyncSessionLocal() as save_db:
                assistant_msg = ChatMessage(session_id=session.id, sender="assistant", content=full_response)
                save_db.add(assistant_msg)
                await save_db.commit()

    return StreamingResponse(event_generator(), media_type="text/plain")