from __future__ import annotations

from domain.chunker import Chunker
from domain.indexer import Indexer
from domain.parsers import Parser


class IndexService:
    def __init__(self):
        self.data = None
        self.headers = None
        self.chunks = None
        self.mongo_documents = None
        self.chroma_sub_chunks = None
        self.index_data = None

    @property
    def parser(self):
        return Parser()

    @property
    def chunker(self):
        return Chunker()

    @property
    def indexer(self):
        return Indexer()
    
    def get_mongo_documents(self):
        return self.mongo_documents

    def process(self, path_data: str) -> str:
        """
        This function will process the file and index the data.

        Args:
            path_data: path to the file.

        Returns:
            str: message if indexing is successful or failed.
        """
        try:
            self.data = self.parser.process(path_data)
            self.headers, self.chunks = self.chunker.process_data(self.data)
            self.mongo_documents, self.chroma_sub_chunks = self.chunker.process_chunks(self.headers, self.chunks)
            self.index_data = self.indexer.process(self.chroma_sub_chunks)

            if self.index_data:
                return '-> Indexing successful <-'
            return '-> Indexing failed <-'

        except Exception as e:
            raise Exception(f"Failed to process data: {str(e)}")
