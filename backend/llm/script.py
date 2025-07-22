# API sử dùng LiteLLM proxy server (nên dùng)
from __future__ import annotations

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class AnswerGenerator:
    def __init__(self):
        self.model = ChatOpenAI(
            model='gemini-2.0-flash-001',
            openai_api_base='http://127.0.0.1:4000',  # LiteLLM Proxy
            openai_api_key='sk-1234',  # API key của LiteLLM Proxy
            temperature=0.7,
        )

    def Chat(self, query: str) -> str:
        """
        Xử lý câu hỏi và trả về câu trả lời.
        """
        template = """Bạn là một trợ lý ảo hữu ích của Tôi Yêu PTIT (TYP),
        nhiệm vụ của bạn là trả lời các câu hỏi một cách:
        - Ngắn gọn, rõ ràng
        - Thân thiện, lịch sự
        - Đúng trọng tâm
        - Cung cấp nguồn tham khảo (nếu có).

        Nếu không biết câu trả lời:
        "Xin lỗi, tôi hiện không có thông tin về vấn đề này.
        Bạn có thể cung cấp thêm chi tiết hoặc thử tìm kiếm từ nguồn khác."

        Nếu câu hỏi mơ hồ:
        "Bạn có thể làm rõ câu hỏi của mình không?
        Tôi muốn hỗ trợ bạn tốt hơn."

        Câu hỏi: {query}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.model | StrOutputParser()
        answer = chain.invoke({'query': query})
        return answer


# Khởi tạo chatbot với LiteLLM Proxy
chatbot = AnswerGenerator()
answer = chatbot.Chat('Bạn là ai?')
print(answer)

# RUNNING on http://0.0.0.0:4000
