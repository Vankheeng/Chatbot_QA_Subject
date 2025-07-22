from __future__ import annotations

import os
import sys
from typing import Any
from typing import Dict

from application.index_service import IndexService
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import UploadFile
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from helpers.save_file import save_file
from models.file_model import Chunk

router = APIRouter()
embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')
persist_directory = os.path.join(os.path.dirname(__file__), '..', 'vector_store')


@router.post('/upload/')
async def upload_file(file: UploadFile) -> Dict[str, Any]:
    """
    Handles file upload, saves the file, processes it into chunks, and stores metadata.

    This function:
    1. Creates a file entry in the database.
    2. Processes the file to extract chunks.
    3. Stores chunk metadata in the database.

    Args:
        file (UploadFile): The uploaded file object.

    Returns:
        Dict[str, Any]: A dictionary containing the saved file's metadata.

    Raises:
        HTTPException: If the file metadata fails to save in the database.
    """
    try:
        save_file_dir = r'assets\data'
        file_path = save_file(file, save_file_dir)
        # Save the file to a specific location
        print(file_path)

        index_service = IndexService()
        index_service.process(file_path)

        chunk_documents = index_service.mongo_documents


        for chunk in chunk_documents:
            chunk.save()
        return {'file_name': file.filename}
    except Exception as e:
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail='Failed to save file metadata to database')



@router.get('/chunks/')
async def get_chunks(QUERY: str):
    vecto_db_2 = Chroma(collection_name='document_chunk_2', embedding_function=embedding, persist_directory=persist_directory)
    retriever = vecto_db_2.as_retriever()
    docs = vecto_db_2.get(include=['documents', 'embeddings'])
    print(len(docs['documents']))
    # Truy vấn với similarity_search_with_score
    # QUERY = """Hiệu trưởng"""
    results = vecto_db_2.similarity_search_with_score(QUERY, k = 10)

    # In kết quả
    print("Number of relevant documents:", len(results))
    for doc, score in results:
        print("Nội dung:", doc.page_content)
        # print("Metadata:", doc.metadata['source'])
        print("Điểm số:", 1- score)  # Điểm số thường là khoảng cách (giá trị càng nhỏ càng tốt)
        print("-------------------------------------------------------\n\n")
    return results