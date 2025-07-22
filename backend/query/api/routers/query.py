from __future__ import annotations

import logging
import time

from fastapi import APIRouter
from pydantic import BaseModel

from application.query_service import QueryService


class Request(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    role: str
    content: str


router = APIRouter(
    prefix='/chat',
    tags=['chat'],
)
bot = QueryService()


@router.post('/', response_model=ChatResponse)
async def get_anwser(request: Request):
    """
    Handle incoming chat requests and return a response from the assistant.

    Args:
        request (Request): The incoming request containing the user's message.

    Returns:
        ChatResponse: A response object containing the assistant's message.

    Raises:
        Exception: If there is an error during processing the message.
    """
    try:
        start_time = time.time()
        # get response
        print(request.message)
        response = await bot.process(request.message)
        end_time = time.time()
        # logging
        logging.info(f'Thời gian trả lời {end_time - start_time: .5f} (s)')
        return ChatResponse(role='assistant', content=response)
    except Exception as e:
        raise Exception(f'Failed to chat with user: {str(e)}')