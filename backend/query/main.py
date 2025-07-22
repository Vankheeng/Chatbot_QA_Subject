from __future__ import annotations

from fastapi import FastAPI
from api.routers.query import router as query_router


app = FastAPI()

app.include_router(query_router, tags=["query"])
