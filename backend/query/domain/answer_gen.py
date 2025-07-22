import os
import sys
import json

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../index/api')))  # noqa
from models.file_model import Chunk


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../llm')))  # noqa
from script import AnswerGenerator  # noqa

# current_dir = os.path.dirname(__file__)
# file_path = os.path.join(current_dir, 'chunks_export.json')

# Chunk_Pydantic = pydantic_model_creator(Chunk, name="Chunk")

generator = AnswerGenerator()
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
persist_directory = os.path.join(os.path.dirname(__file__), '../../index/api', 'vector_store')
vector_db = Chroma(collection_name='document_chunk_2', embedding_function=embedding, persist_directory=persist_directory)



def split_text(text: str) -> str:
  
    """
    Extracts a portion of the text between the first '[' and the last ']'.

    Args:
        text (str): The input text to split.

    Returns:
        str: The extracted portion of the text between the brackets.
    """
    start = text.find('[')
    end = text.rfind(']')
    return text[start:end+1]

def set_prompt_answer(context: str) -> str:
    """
    Generates a prompt to guide the assistant in answering the user's query.

    Args:
        context (str): The knowledge context the assistant will use to answer the query.

    Returns:
        str: A formatted string with the context and a placeholder for the user's question.
    """
    prompt_answer = f"""Bạn là 1 chuyên gia trong lĩnh vực hệ điều hành hỗ trợ câu hỏi của sinh viên về lĩnh vực này. Bạn chỉ được phép dựa vào các kiến thức được cung cấp dưới đây để trả lời câu hỏi, hãy đưa ra câu trả lời đầy đủ và chi tiết nhất có thể đồng thời hãy chỉ ra bạn đã đoạn kiến thức mà bạn đã dùng để trả lời để người dùng có thể tham khảo thêm. Nếu thông tin thuộc các kiến thức được cung cấp không liên quan đến câu hỏi. hãy trả lời không có thông tin.
                    chú ý định dạng trả về của câu hỏi ở dạng json như sau:
                    ví dụ: hệ điều hành là gì?
                    [{{"Câu trả lời" : "Hệ điều hành là một hệ thống phần mềm đóng vai trò trung gian giữa người sử dụng và phần cứng máy tính.  Vai trò chính của nó là tạo ra một môi trường thuận tiện để thực hiện các chương trình.  Nhiệm vụ của hệ điều hành là làm cho việc sử dụng hệ thống máy tính được tiện lợi và hiệu quả.  Để đạt được điều này, hệ điều hành thực hiện hai chức năng cơ bản: quản lý tài nguyên và quản lý việc thực hiện các chương trình.  Hệ điều hành cũng quản lý và đảm bảo việc sử dụng phần cứng của máy tính được hiệu quả.",
                    "Trích dẫn" : ['[KHÁI NIỆM HỆ ĐIỀU HÀNH - Khái niệm hệ điều hành - Vai trò của Hệ điều hành]',
                                    '[KHÁI NIỆM HỆ ĐIỀU HÀNH - Chức năng của hệ điều hành - Hai chức năng cơ bản của hệ điều hành]',
                                    '[CÁC THÀNH PHẦN CỦA HỆ THỐNG MÁY TÍNH - Vai trò của hệ điều hành - Nhiệm vụ chính của hệ điều hành]',
                                    '[KHÁI NIỆM HỆ ĐIỀU HÀNH - Chức năng của hệ điều hành - Chức năng quản lý tài nguyên của hệ điều hành]']
                    }}]

                    Kiến thức: {context}
                    câu hỏi: """
    return prompt_answer



class AnswerGenerator:
    def __init__(self):
        self.prompt_5_2 = """Tôi đang ôn tập lại kiến thức môn Hệ Điều Hành. Để trả lời được câu hỏi sau, tôi cần nắm rõ ít nhất những mục kiến thức nào trong giáo trình. Hãy đưa ra các bước lần lượt tôi nên đọc phần nào trước tiên, cụ thể mục nào trong phần đó (nhiều nhất là 3 bước).Hãy đưa ra danh sách các mục cần tìm hiểu lần lượt dưới dạng json.
                            Chú ý : Vì các items trong câu trả lời sẽ được dùng để truy xuất những kiến thức liên quan trong database nên, nghĩa các items phải được tách biệt không phụ thuộc vào ngữ cảnh, phi ngữ cảnh hóa items bằng cách thêm chủ thể cần tìm hiểu của items vào items.
                            ví dụ :câu hỏi: So sánh ưu nhược điểm của các phương pháp xử lý theo mẻ, đa chương trình không chia sẻ thời gian, và đa chương trình có chia sẻ thời gian (đa nhiệm).


                            ```json
                            [
                            {
                                "step": 1,
                                "title": "Các phương pháp xử lý chương trình",
                                "items": [
                                "Xử lý theo mẻ (Batch Processing): Định nghĩa, ưu điểm (hiệu quả cho các công việc lớn, không cần sự tương tác của người dùng), nhược điểm (thời gian phản hồi chậm, không linh hoạt).",
                                "Đa chương trình không chia sẻ thời gian: Định nghĩa, cách thức hoạt động (các chương trình được thực thi tuần tự, nhưng nhiều chương trình cùng nằm trong bộ nhớ), ưu điểm (tận dụng CPU tốt hơn xử lý theo mẻ), nhược điểm (thời gian phản hồi vẫn chậm, một chương trình lỗi có thể ảnh hưởng toàn bộ hệ thống).",
                                "Đa chương trình có chia sẻ thời gian (Đa nhiệm): Định nghĩa, cách thức hoạt động (time-slicing, context switching), ưu điểm (thời gian phản hồi nhanh, khả năng tương tác cao, khả năng xử lý nhiều công việc cùng lúc), nhược điểm (overhead do context switching, cần quản lý tiến trình phức tạp)."
                                ]
                            },
                            {
                                "step": 2,
                                "title": "So sánh các phương pháp",
                                "items": [
                                "So sánh thời gian phản hồi của ba phương pháp (Xử lý theo mẻ (Batch Processing), Đa chương trình không chia sẻ thời gian, Đa chương trình có chia sẻ thời gian (Đa nhiệm)).",
                                "So sánh mức độ sử dụng CPU của ba phương pháp (Xử lý theo mẻ (Batch Processing), Đa chương trình không chia sẻ thời gian, Đa chương trình có chia sẻ thời gian (Đa nhiệm)).",
                                "So sánh mức độ tương tác người dùng của ba phương pháp (Xử lý theo mẻ (Batch Processing), Đa chương trình không chia sẻ thời gian, Đa chương trình có chia sẻ thời gian (Đa nhiệm)).",
                                "So sánh độ phức tạp của quản lý hệ thống cho mỗi phương pháp (Xử lý theo mẻ (Batch Processing), Đa chương trình không chia sẻ thời gian, Đa chương trình có chia sẻ thời gian (Đa nhiệm)).",
                                ]
                            },
                            {
                                "step": 3,
                                "title": "Ví dụ minh họa",
                                "items": [
                                "Đưa ra ví dụ cụ thể minh họa sự khác biệt giữa ba phương pháp (Xử lý theo mẻ (Batch Processing), Đa chương trình không chia sẻ thời gian, Đa chương trình có chia sẻ thời gian (Đa nhiệm)), ví dụ như xử lý một loạt ảnh, xử lý yêu cầu từ nhiều người dùng cùng lúc.",
                                "Phân tích trường hợp nào phù hợp với từng phương pháp(Xử lý theo mẻ (Batch Processing), Đa chương trình không chia sẻ thời gian, Đa chương trình có chia sẻ thời gian (Đa nhiệm))."
                                ]
                            }
                            ]
                            ```
                            ```câu hỏi : """
        # """Sinh viên đang ôn tập lại kiến thức môn Hệ Điều Hành. Các câu hỏi của sinh viên được chia thành 2 dạng như sau:
        #                 Dạng 1: Các câu hỏi không rõ ràng hoặc không liên quan đến môn hệ điều hành:
        #                 Bạn có thể trả lời dựa trên ngữ cảnh "Bạn là 1 trợ lý ảo của 'Tôi Yêu PTIT'. Bạn có duy nhất 1 nhiệm vụ là giúp đỡ các bạn sinh viên trả lời các câu hỏi liên quan đến môn hệ điều hành".
        #                 Hãy đảm bảo các câu trả lời phải lịch sự, thân thiện và chỉ trả về câu trả lời  như sau:
        #                 ví dụ 1: câu hỏi: "xin chào"
        #                     "Xin chào, tôi là trợ lý ảo của 'Tôi Yêu PTIT' giúp đỡ bạn tìm hiểu môn hệ điều hành, bạn có câu hỏi nào cần tôi giúp đỡ không?"
        #                 ví dụ 2: câu hỏi :"Cờ hiệu"
        #                     "Bạn có thể cung cấp câu trả lời rõ ràng hơn không".
        #                 Dạng 2: Các câu hỏi liên quan đến kiến thức của hệ điều hành:
        #                 Để trả lời được câu hỏi này, sinh viên cần nắm rõ ít nhất những mục kiến thức nào trong giáo trình. Hãy đưa ra các bước lần lượt sinh viên nên đọc phần nào trước tiên, cụ thể mục nào trong phần đó (nhiều nhất là 3 bước).Hãy đưa ra danh sách các mục cần tìm hiểu lần lượt dưới dạng json.
        #                 Chú ý : Vì các items trong câu trả lời sẽ được dùng để truy xuất những kiến thức liên quan trong database nên, nghĩa các items phải được tách biệt không phụ thuộc vào ngữ cảnh, phi ngữ cảnh hóa items bằng cách thêm chủ thể cần tìm hiểu của items vào items.
        #                 ví dụ :câu hỏi: So sánh ưu nhược điểm của các phương pháp xử lý theo mẻ, đa chương trình không chia sẻ thời gian, và đa chương trình có chia sẻ thời gian (đa nhiệm).


        #                 ```json
        #                 [
        #                 {
        #                     "step": 1,
        #                     "title": "Các phương pháp xử lý chương trình",
        #                     "items": [
        #                     "Xử lý theo mẻ (Batch Processing): Định nghĩa, ưu điểm (hiệu quả cho các công việc lớn, không cần sự tương tác của người dùng), nhược điểm (thời gian phản hồi chậm, không linh hoạt).",
        #                     "Đa chương trình không chia sẻ thời gian: Định nghĩa, cách thức hoạt động (các chương trình được thực thi tuần tự, nhưng nhiều chương trình cùng nằm trong bộ nhớ), ưu điểm (tận dụng CPU tốt hơn xử lý theo mẻ), nhược điểm (thời gian phản hồi vẫn chậm, một chương trình lỗi có thể ảnh hưởng toàn bộ hệ thống).",
        #                     "Đa chương trình có chia sẻ thời gian (Đa nhiệm): Định nghĩa, cách thức hoạt động (time-slicing, context switching), ưu điểm (thời gian phản hồi nhanh, khả năng tương tác cao, khả năng xử lý nhiều công việc cùng lúc), nhược điểm (overhead do context switching, cần quản lý tiến trình phức tạp)."
        #                     ]
        #                 },
        #                 {
        #                     "step": 2,
        #                     "title": "So sánh các phương pháp",
        #                     "items": [
        #                     "So sánh thời gian phản hồi của ba phương pháp (Xử lý theo mẻ (Batch Processing), Đa chương trình không chia sẻ thời gian, Đa chương trình có chia sẻ thời gian (Đa nhiệm)).",
        #                     "So sánh mức độ sử dụng CPU của ba phương pháp (Xử lý theo mẻ (Batch Processing), Đa chương trình không chia sẻ thời gian, Đa chương trình có chia sẻ thời gian (Đa nhiệm)).",
        #                     "So sánh mức độ tương tác người dùng của ba phương pháp (Xử lý theo mẻ (Batch Processing), Đa chương trình không chia sẻ thời gian, Đa chương trình có chia sẻ thời gian (Đa nhiệm)).",
        #                     "So sánh độ phức tạp của quản lý hệ thống cho mỗi phương pháp (Xử lý theo mẻ (Batch Processing), Đa chương trình không chia sẻ thời gian, Đa chương trình có chia sẻ thời gian (Đa nhiệm)).",
        #                     ]
        #                 },
        #                 {
        #                     "step": 3,
        #                     "title": "Ví dụ minh họa",
        #                     "items": [
        #                     "Đưa ra ví dụ cụ thể minh họa sự khác biệt giữa ba phương pháp (Xử lý theo mẻ (Batch Processing), Đa chương trình không chia sẻ thời gian, Đa chương trình có chia sẻ thời gian (Đa nhiệm)), ví dụ như xử lý một loạt ảnh, xử lý yêu cầu từ nhiều người dùng cùng lúc.",
        #                     "Phân tích trường hợp nào phù hợp với từng phương pháp(Xử lý theo mẻ (Batch Processing), Đa chương trình không chia sẻ thời gian, Đa chương trình có chia sẻ thời gian (Đa nhiệm))."
        #                     ]
        #                 }
        #                 ]
        #                 ```
        #                 ```câu hỏi : """

        self.chunks_json = []
        chunks = Chunk.objects()
        for chunk in chunks:
            self.chunks_json.append({
                "_id": str(chunk.id),  # Convert ObjectId to string
                "header": chunk.header,
                "topic": chunk.topic,
                "chunk": chunk.chunk,
                "page": chunk.page,
                "sub_chunks": chunk.sub_chunks
            })

    def summarize(self, prompt: str, text: str) -> str:
        """
        Summarizes the input text using the given prompt.

        Args:
            prompt (str): The prompt to guide the summary.
            text (str): The text to summarize.

        Returns:
            str: The summarized response from the generator.
        """
        response = generator.Chat(prompt + text)
        return response
    
    def find_page_containing_sub_chunk(self, header_sub_chunk):

        # Duyệt qua từng bản ghi
        for record in self.chunks_json:
            for sub_chunk in record['sub_chunks']:
                if sub_chunk.startswith(header_sub_chunk):
                    # print("Chunk chứa sub_chunk với tiêu đề khớp:", record['chunk'])
                    return record['page']

        print("Không tìm thấy sub_chunk khớp.")
        return None

    def answer_1(self, query: str) -> str:
        """
        Generates an answer based on a detailed study plan for answering a question.

        Args:
            query (str): The query for which the study steps and answer are required.

        Returns:
            str: A formatted string with the study plan and the answer.
        """
        output_split = self.summarize(self.prompt_5_2, query)  # các bước tìm hiểu kiến thức để trả lời câu hỏi
        print("output_split: ", output_split)
        try:
            output_split1 = split_text(output_split)
            data = json.loads(output_split1)
        except:
            return output_split
        context = ""
        for x in data:
            for y in x["items"]:
                y = x["title"] +"-" + y
                context += f"kiến thức liên quan đến : " + x["title"] + ": \n"
                results = vector_db.similarity_search_with_score(y, k = 3)
                for doc, score in results:
                    context += " - " + doc.page_content + "\n"
                context += '\n'
                


        prompt_answer = set_prompt_answer(context)
        print("prompt_answer: ", prompt_answer)
        output_split = self.summarize(prompt_answer, query)
        print("output_split: ", output_split)
        try:
            output_split1 = split_text(output_split)
            data = json.loads(output_split1)
        except:
            # output_split = self.summarize(prompt_answer, query)
            print("return: ", output_split)
            return output_split

        answer = data[0]['Câu trả lời']
        quote = data[0]['Trích dẫn']
        linked_quotes = []
        seen_links = set()  # dùng để tránh trùng link
        link_prefix = "https://mozilla.github.io/pdf.js/web/viewer.html?file=https%3A%2F%2Fraw.githubusercontent.com%2Fdaothihuyen64%2FLLM-PDF-QA%2Ffeature%2FRAG_Without_UI%2FH%E1%BB%87%2520%C4%91i%E1%BB%81u%2520h%C3%A0nh%2520-%25202015.pdf#page="

        for q in quote:
            page = self.find_page_containing_sub_chunk(q)
            if page is not None:
                full_link = f"{link_prefix}{page}"
                if full_link not in seen_links:
                    seen_links.add(full_link)
                    linked = f"[{q}]({full_link})"
                    linked_quotes.append(linked)
            else:
                linked_quotes.append(q)  # fallback nếu không tìm được trang
        quote_text = "\n\n".join(linked_quotes)

        return answer + "\n\n" + quote_text

    def process(self, query: str) -> str:
        """
        Processes the user's query and generates an appropriate answer.

        Args:
            query (str): The user's question.

        Returns:
            str: The final answer generated for the query.
        """
        template = """ Bạn là một trợ lý ảo hữu ích của Tôi Yêu PTIT (TYP), nhiệm vụ của bạn là trả lời các câu hỏi một Ngắn gọn, rõ ràng, Thân thiện, lịch sự, Đúng trọng tâm và đồng thời cung cấp nguồn tham khảo (nếu có).
        Nếu không biết câu trả lời:
        "Xin lỗi, tôi hiện không có thông tin về vấn đề này. Bạn có thể cung cấp thêm chi tiết hoặc thử tìm kiếm từ nguồn khác."
        Nếu câu hỏi mơ hồ:
        "Bạn có thể làm rõ câu hỏi của mình không? Tôi muốn hỗ trợ bạn tốt hơn."

        câu hỏi: {query}
        """
        answer = self.answer_1(query)
        return answer