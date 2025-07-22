from __future__ import annotations

import os

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
persist_directory = os.path.join(os.path.dirname(__file__), '..', 'vector_store')
embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
chroma_db = Chroma(collection_name='document_chunk_2', embedding_function=embedding, persist_directory=persist_directory)

class Indexer:
    def process(self, list_data: list[Document]) -> bool:
        """
        This function will process the list of data and index it.

        Args:
            list_data: list of data to be indexed.

        Returns:
            bool: True if indexing is successful, False otherwise.
        """
        try:
            chroma_db.add_documents(list_data)            
            return True
        except Exception as e:
            raise Exception(f"Failed to index data: {str(e)}")
