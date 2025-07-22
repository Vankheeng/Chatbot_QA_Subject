import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../domain')))  # noqa
from answer_gen import AnswerGenerator


class QueryService:
    def __init__(self):
        pass

    async def process(self, query: str) -> str:
        """
        Asynchronously processes the user's query and returns a generated answer.

        Args:
            query (str): The user's input question or message.

        Returns:
            str: The generated response from the assistant.
        """
        answer_generator = AnswerGenerator()
        return answer_generator.process(query)
