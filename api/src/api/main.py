from fastapi import FastAPI

from .models.model import Message
from .routers.chat_service import get_message

app = FastAPI()


@app.post("/send_message")
async def send_message(message: Message):
    response = await get_message(message.message)

    return {"response": response}
