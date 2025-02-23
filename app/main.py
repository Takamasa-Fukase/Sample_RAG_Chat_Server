import json, threading
from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
from callback_handler import CallbackHandler
from chat_assistant import ChatAssistant
from data_models import AnswerResponseQueue, SendQuestionRequest, StreamAnswerResponseData, StreamErrorResponseData
from chat_assistant import ChatAssistant
from vector_stores import spain_fukase_vector_store
from callback_handler import CallbackHandler

app = FastAPI()

# CORSを回避するために追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/ping')
def ping():
    return {'data': {'message': 'OK'}}


@app.post('/chat')
def get_answer(
        request: Request,
        body: SendQuestionRequest,
):
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
                print("client disconnected")
                return

            # chatbotから回答が送られてくるまで待機
            print("waiting for chatbot answer")
            data = channel.get()
            print("chatbot answer received")

            # 送られてきたデータがStopIterationなら終了
            if isinstance(data, StopIteration):
                print("chatbot stream closed")
                break

            # 送られてきたデータがException系ならraiseして脱出
            if isinstance(data, StreamErrorResponseData):
                error_response = StreamAnswerResponseData(
                    answer_type_id=2,  # 2: part_of_final_answer_text
                    part_of_final_answer_text=data.message,
                    status_code=data.status_code
                )
                print("chatbot stream closed with error")
                yield json.dumps(error_response.dict())
                raise data

            # 会話ログに保存するために追加
            if isinstance(data, StreamAnswerResponseData) \
                    and data.part_of_final_answer_text is not None:
                answer_texts.append(data.part_of_final_answer_text)

            # 普通のAIからの返答なら、ユーザー側に返す
            yield json.dumps(data.dict())
            print("chatbot stream data sent")

    return EventSourceResponse(receive_answer_with_streamed_chat_completion_api())


def handle_question(
        sender: AnswerResponseQueue,
        body: SendQuestionRequest,
):
    print("handle_question started")
    try:
        assistant = ChatAssistant(
            callback_handler=CallbackHandler(queue=sender),
            question_body=body.get_question_body(),
            vector_store=spain_fukase_vector_store,
            model_name='gpt-3.5-turbo',
            temperature=0.7,
            use_latest_information=True,
            system_role_prompt_text=''
        )
        assistant.get_answer()
    
        sender.close()
        print("handle_question finished")

    except HTTPException as e:
        sender.send_error(e)
        raise e   

    except BaseException as e:
        sender.send_error(e)
        raise e