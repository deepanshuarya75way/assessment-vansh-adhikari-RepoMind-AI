from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import settings


engine = create_async_engine(settings.DATABASE_URL, echo=True, future=True)
AsyncSessionLocal= async_sessionmaker(
    bind=engine, 
    expire_on_commit=False,
    class_=AsyncSession)

class Base(DeclarativeBase):
    pass
#fastest way to open or close a Db helper in web application and is to use dependency injection. In this case, we can use the get_db function to create a new session for each request and close it when the request is finished. This way, we can avoid connection leaks and ensure that each request has its own session.
async def get_db():
    #This opens a safe workspace called session. It automatically closes the workspace when your code is done, even if an error happens.
    async with AsyncSessionLocal() as session:
        yield session


