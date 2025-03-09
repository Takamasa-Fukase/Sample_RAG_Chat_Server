import json, threading
import system_prompts
import vector_stores
import asyncio
import random
from starlette.responses import FileResponse
from typing import List
from sqlalchemy.orm import Session
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
from callback_handler import CallbackHandler
from chat_assistant import ChatAssistant
from data_models import AnswerResponseQueue, SendQuestionRequest, StreamAnswerResponseData, StreamErrorResponseData, Category, CategoryListResponse
from chat_assistant import ChatAssistant
from callback_handler import CallbackHandler
from database import get_db
from category_repository import CategoryRepository
from env import Env

app = FastAPI()

# CORSの設定（フロントエンドアプリからのアクセスのみ許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[Env.APP_URL],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ローカル画像を配信（より簡単な方法）
# app.mount('/images', StaticFiles(directory='images'), name='images')


# ローカル画像を配信（デバッグ用に遅延処理など入れる場合はこちらを使う）
@app.get('/images/{file_path}')
async def get_image(file_path: str):
    # 画像ごとに微妙に取得時間を変えることでリアルな見た目でデバッグできるようにする
    # delay_time = random.uniform(0.5, 1.0)
    # await asyncio.sleep(delay_time)
    return FileResponse(path=f'images/{file_path}')


@app.get('/ping')
def ping():
    return {'data': {'message': 'OK'}}


@app.get('/categories')
def get_categories(
    db: Session = Depends(get_db)
) -> CategoryListResponse:
    repository = CategoryRepository()
    categories = repository.get_all_categories(db=db)
    return CategoryListResponse(data=categories)


@app.post('/chat')
def get_answer(
        request: Request,
        body: SendQuestionRequest,
):
    # print(f'chat api body: {body}, id: {body.category_id}, text: {body.text}, previous_messages: {body.previous_messages}')

    async def receive_answer_with_streamed_chat_completion_api():
        channel = AnswerResponseQueue()
        task = threading.Thread(
            target=handle_question,
            args=(channel, body)
        )
        task.start()

        answer_texts = []
        while True:
            if await request.is_disconnected():
                # print("client disconnected")
                return

            # chatbotから回答が送られてくるまで待機
            # print("waiting for chatbot answer")
            data = channel.get()
            # print("chatbot answer received")

            # 送られてきたデータがStopIterationなら終了
            if isinstance(data, StopIteration):
                # print("chatbot stream closed")
                break

            # 送られてきたデータがException系ならraiseして脱出
            if isinstance(data, StreamErrorResponseData):
                error_response = StreamAnswerResponseData(
                    answer_type_id=2,  # 2: part_of_final_answer_text
                    part_of_final_answer_text=data.message,
                    status_code=data.status_code
                )
                # print("chatbot stream closed with error")
                yield json.dumps(error_response.dict())
                raise data

            # 会話ログに保存するために追加
            if isinstance(data, StreamAnswerResponseData) \
                    and data.part_of_final_answer_text is not None:
                answer_texts.append(data.part_of_final_answer_text)

            # 普通のAIからの返答なら、ユーザー側に返す
            yield json.dumps(data.dict())
            # print(f"chatbot stream data sent: {data.dict()}")

    return EventSourceResponse(receive_answer_with_streamed_chat_completion_api())


def handle_question(
        sender: AnswerResponseQueue,
        body: SendQuestionRequest,
):
    # print("handle_question started")
    match body.category_id:
        case 0:
            system_role_prompt_text = system_prompts.CATEGORY_0_SYSTEM_PROMPT
            vector_store = vector_stores.vector_store_2025
        case 1:
            system_role_prompt_text = system_prompts.CATEGORY_1_SYSTEM_PROMPT
            vector_store = vector_stores.vector_store_2022
        case 2:
            system_role_prompt_text = system_prompts.CATEGORY_2_SYSTEM_PROMPT                            
            vector_store = vector_stores.vector_store_2019
    
    try:
        assistant = ChatAssistant(
            callback_handler=CallbackHandler(queue=sender),
            sendQuestionRequest=body,
            vector_store=vector_store,
            model_name='gpt-4o-mini',
            temperature=0.7,
            use_latest_information=True,
            is_enabled_web_and_index_data_integrated_mode=False,
            system_role_prompt_text=system_role_prompt_text
        )
        assistant.get_answer()
    
        sender.close()
        # print("handle_question finished")

    except HTTPException as e:
        sender.send_error(e)
        raise e   

    except BaseException as e:
        sender.send_error(e)
        raise e