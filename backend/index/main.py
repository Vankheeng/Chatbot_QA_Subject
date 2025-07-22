from __future__ import annotations

from api.routers.indexer import router as file_router
from fastapi import FastAPI

app = FastAPI()

app.include_router(file_router, prefix='/index', tags=['index'])
