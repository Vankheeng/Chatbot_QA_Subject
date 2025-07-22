from __future__ import annotations

import json
import re
import os  # noqa
import sys  # noqa
from typing import List

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.api_core import exceptions
from api.models.file_model import Chunk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../llm')))  # noqa
from script import AnswerGenerator  # noqa


generator = AnswerGenerator()


def summarize_2(prompt: str, text: str) -> str:
    """
    Generates a summary of the given text based on the provided prompt.

    This function uses a model to generate a response by concatenating the prompt and text.

    Args:
        prompt (str): The prompt that guides the summarization.
        text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """
    response = generator.Chat(prompt + text)
    return response


def split_text(text: str) -> str:
    """
    Extracts and returns the substring enclosed in square brackets from the given text.

    This function finds the first occurrence of '[' and the last occurrence of ']'
    and returns the substring between them, including the brackets.

    Args:
        text (str): The input text containing square brackets.

    Returns:
        str: The extracted substring enclosed in square brackets.
    """
    start = text.find('[')
    end = text.rfind(']')
    return text[start:end + 1]


class Chunker:
    def __init__(self):
        self.prompt_semantic_chunk_7 = f""" Tôi đang làm về RAG phần index, tôi cần index 1 giáo trình, đoạn văn sau đây là chỉ là 1 mục trong giáo trình, hãy tách đoạn văn theo kiểu semantic chunk, vì size của các chunk nếu quá khác nhau sẽ ảnh hưởng đến kết quả truy xuất, nên khi tách hãy chú ý :
            1. Bạn không được lược bỏ hay thêm bất cứ câu nào của đoạn văn cần tách, tức là sau khi chia các chunk hợp nhất lại phải ra đủ số câu của đoạn văn ban đầu.
            2. Giữ nguyên cách diễn đạt gốc từ đầu vào nếu có thể
            3. Tôi đã chia các mục trước đó của giáo trình thành các chunk có độ dài từ 3 đến 5 câu. Hãy chia các chunk có size gần giống nhau để tránh ảnh hưởng đến kết quả truy xuất.
            4. Các chunk là các đoạn ngữ nghĩa dựa trên sự chuyển đổi tự nhiên của nội dung (ví dụ: giới thiệu, phương pháp, ví dụ, kết quả, kết luận, câu hỏi ôn tập).
            5. Trình bày kết quả dưới dạng danh sách chuỗi, định dạng bằng JSON, đồng thời trình bày chủ đề chính của từng chunk . Kết qua trả về có dạng như sau :[{{'topic': nội dung topic, 'chunk' : nội dung chunk}}].
            Biết semantic chunk có các đặc điểm sau : một **semantic chunk** là một đoạn văn bản có ý nghĩa liên kết chặt chẽ về mặt ngữ nghĩa.  Nó không chỉ là một đoạn văn bản có độ dài cố định (ví dụ, 100 từ), mà được tách ra từ văn bản nguồn dựa trên các yếu tố ngữ nghĩa như:

            * **Sự liên kết chủ đề:** Các câu trong một semantic chunk thường xoay quanh một chủ đề cụ thể.
            * **Cấu trúc câu:** Các câu trong một semantic chunk thường có liên quan về mặt cấu trúc, tạo thành một ý tưởng hoàn chỉnh.
            * **Mối quan hệ logic:** Các câu trong một semantic chunk thường có mối quan hệ logic với nhau, ví dụ như quan hệ nguyên nhân - kết quả, đối lập, liệt kê.

            Đoạn văn:
            """
        
    def split_paragraphs(self, full_text: str) -> tuple[list[str], list[str]]:
        """
        Splits the input text into paragraphs based on heading patterns, and associates headings with their respective content.

        Args:
            full_text (str): The input text to be split, which may contain sections identified by numbered headings (e.g., "1.1", "1.2.3").

        Returns:
            tuple: A tuple containing two lists:
                - List of strings representing the headings found in the text.
                - List of strings representing the content associated with each heading.
        
        The headings are identified using a regular expression pattern that matches common numbered heading formats.
        """
        heading_pattern = r'^\d+\.\d+(?:\.\d+)?\.?\s*'

        # Tách dữ liệu thành các đoạn dựa trên pattern
        segments = re.split(heading_pattern, full_text, flags=re.MULTILINE)

        headers = []
        for i, segment in enumerate(segments):
            headers.append(segment.splitlines()[0].strip())

        # Loại bỏ phần tử rỗng (nếu có) và kết hợp lại tiêu đề với nội dung
        result = []
        for i in range(len(segments)):
            if segments[i].strip():  # Kiểm tra nếu đoạn không rỗng
                # Nếu là đoạn đầu tiên hoặc có tiêu đề trước đó
                if i > 0 and re.match(heading_pattern, full_text.splitlines()[sum(len(s.splitlines()) for s in segments[:i])]):
                    # Lấy tiêu đề từ dòng trước đó
                    lines = full_text.splitlines()
                    header_index = sum(len(s.splitlines()) for s in segments[:i])
                    header = lines[header_index].strip()
                    result.append(f"{header} {segments[i].strip()}")
                else:
                    result.append(segments[i].strip())
        return headers, result
    
    def prompt_8(self, topic: str, header: str) -> str:
        """
        Generates a prompt to split a given text into semantic chunks based on a specified topic and section header.
        
        Args:
            topic (str): The topic discussed in the given section of the textbook.
            header (str): The header of the section in the textbook where the topic appears.
        
        Returns:
            str: A formatted string that can be used to guide the splitting of the input text into semantic chunks.
        """
        prompt_semantic_chunk_8 = f""" Tôi đang làm về RAG phần index, tôi cần index giáo trình "Hệ điều hành", đoạn văn sau đây nói về {topic} thuộc phần {header} trong giáo trình, hãy tách đoạn văn theo kiểu semantic chunk, vì size của các chunk nếu quá khác nhau sẽ ảnh hưởng đến kết quả truy xuất, nên khi tách hãy chú ý :
        1. Bạn không được lược bỏ hay thêm bất cứ câu nào của đoạn văn cần tách, tức là sau khi chia các chunk hợp nhất lại phải ra đủ số câu của đoạn văn ban đầu.
        2. Giữ nguyên cách diễn đạt gốc từ đầu vào nếu có thể
        3. Tôi đã chia các mục trước đó của giáo trình thành các chunk có độ dài từ 3 đến 5 câu. Hãy chia các chunk có size gần giống nhau để tránh ảnh hưởng đến kết quả truy xuất.
        4. Các chunk là các đoạn ngữ nghĩa dựa trên sự chuyển đổi tự nhiên của nội dung (ví dụ: giới thiệu, phương pháp, ví dụ, kết quả, kết luận, câu hỏi ôn tập).
        5. Trình bày kết quả dưới dạng danh sách chuỗi, định dạng bằng JSON, đồng thời trình bày chủ đề chính của từng chunk . Kết qua trả về có dạng như sau :[{{'topic': nội dung topic, 'chunk' : nội dung chunk}}].
        Biết semantic chunk có các đặc điểm sau : một **semantic chunk** là một đoạn văn bản có ý nghĩa liên kết chặt chẽ về mặt ngữ nghĩa.  Nó không chỉ là một đoạn văn bản có độ dài cố định (ví dụ, 100 từ), mà được tách ra từ văn bản nguồn dựa trên các yếu tố ngữ nghĩa như:

        * **Sự liên kết chủ đề:** Các câu trong một semantic chunk thường xoay quanh một chủ đề cụ thể.
        * **Cấu trúc câu:** Các câu trong một semantic chunk thường có liên quan về mặt cấu trúc, tạo thành một ý tưởng hoàn chỉnh.
        * **Mối quan hệ logic:** Các câu trong một semantic chunk thường có mối quan hệ logic với nhau, ví dụ như quan hệ nguyên nhân - kết quả, đối lập, liệt kê.

        Đoạn văn:
        """
        return prompt_semantic_chunk_8

    def split_docs(self, documents: str) -> tuple[list[str], list[dict]]:
        """
        Splits a document into smaller semantic chunks based on predefined semantic chunking rules.
        
        Args:
            documents (str): A single document as a string to be split into semantic chunks.
        
        Returns:
            tuple: A tuple containing:
                - A list of headers extracted from the input document.
                - A list of dictionaries, each containing the topic and content of a chunk.
        
        Raises:
            JSONDecodeError: If the semantic chunk output cannot be parsed as valid JSON.
        """

        #Split the documents into smaller paragraphs
        headers, results = self.split_paragraphs(documents)
        chunks = []

        for segment in results:
            #Initialize the text splitter 
            output_split = summarize_2(self.prompt_semantic_chunk_7, segment)
            output_split = split_text(output_split)
            
            try:
                data = json.loads(output_split)
            except json.JSONDecodeError as e:
                print(f"Lỗi JSON ở output_split: {e} - Dữ liệu: {output_split}")
                data = [{"topic": "", "chunk": output_split}]
            
            chunks.append(data)
        return headers, chunks
    
    def split_sub_chunks(self, headers: list[str], chunks: list[dict]) -> tuple[list[Chunk], list[Document]]:
        """
        Splits the chunks into further sub-chunks, summarizing and processing each one for further indexing.

        Args:
            headers (list[str]): A list of headers associated with the chunks to guide the splitting process.
            chunks (list[dict]): A list of chunks to be processed, each containing a topic and content.

        Returns:
            tuple: A tuple containing:
                - A list of `Chunk` objects representing the further split and summarized data.
                - A list of `Document` objects that can be used for further storage or retrieval.
        
        Raises:
            JSONDecodeError: If the sub-chunks cannot be parsed as valid JSON.
        """

        mongo_documents = []
        chroma_sub_chunks = []
        for i, data in enumerate(chunks):
            for item in data:
                out_put_2_split = summarize_2(self.prompt_8(item["topic"], headers[i]), item["chunk"])
                out_put_2_split = split_text(out_put_2_split)
                
                try:
                    sub_chunks = json.loads(out_put_2_split)
                except json.JSONDecodeError as e:
                    print(f"Lỗi JSON ở out_put_2_split: {e} - Dữ liệu: {out_put_2_split}")
                    sub_chunks = [{"topic": "", "chunk": out_put_2_split}]
                
                # Tạo final_data (danh sách sub_sub_chunks)
                final_data = [f'[{headers[i]} - {item["topic"]} - {chunk["topic"]}]' + chunk["chunk"] for chunk in sub_chunks]

                mongo_documents.append(Chunk(
                    header=headers[i],
                    topic=item["topic"],
                    chunk=item["chunk"],
                    sub_chunks=final_data
                ))

                chroma_sub_chunks.extend([Document(page_content=chunk) for chunk in final_data])


        return mongo_documents, chroma_sub_chunks


    def process_data(self, data: str) :
        """
        Processes the input data by splitting it into chunks and extracting relevant information.

        Args:
            data (str): The data to be processed and chunked.

        Returns:
            list[dict]: A list of dictionaries, each containing a chunk's topic and content.

        Raises:
            Exception: If the chunking process fails, an exception is raised with a descriptive error message.
        """
        try:
            headers, chunks = self.split_docs(data)
            return headers, chunks
        except Exception as e:
            raise Exception(f"Failed to chunk data: {str(e)}")

    def process_chunks(self, header, chunks) -> list[Document]:
        """
        Processes the chunks by splitting them into smaller sub-chunks and generating documents for each.

        Args:
            header (str): The header to associate with the chunks.
            chunks (list[dict]): A list of chunks to be further processed.

        Returns:
            list[Document]: A list of `Document` objects representing the final sub-chunks for storage or retrieval.

        Raises:
            Exception: If the chunk splitting process fails, an exception is raised with a descriptive error message.
        """
        try:
            mongo_documents, chroma_sub_chunks = self.split_sub_chunks(header, chunks)
            return mongo_documents, chroma_sub_chunks
        except Exception as e:
            raise Exception(f"Failed to split chunks: {str(e)}")
