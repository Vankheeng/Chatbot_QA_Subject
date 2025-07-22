from __future__ import annotations

from langchain_community.document_loaders import PyPDFLoader
import pdfplumber

class Parser:
    def __init__(self):
        pass

    def process(self, path_data: str) -> str:
        """
        This function will process the file with path_data and return the data.

        Args:
            path_data: path to the file.

        Returns:
            str: data read from the
        """
        try:
            with pdfplumber.open(path_data) as pdf:
                full_text = "\n".join([page.extract_text() for page in pdf.pages[7:]])
            return full_text 
        except Exception as e:
            raise Exception(f"Failed to parse data: {str(e)}")
