import queue
import json, threading
from typing import Optional, Union
from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
from app.callback_handler import CallbackHandler
from app.chat_assistant import ChatAssistant
from app.vector_stores import spain_fukase_vector_store
from app.data_models import SendQuestionRequest, StreamAnswerResponseData, StreamErrorResponseData

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


class AnswerResponseQueue:
    def __init__(self):
        self.queue = queue.Queue()

    def send(self, data: StreamAnswerResponseData):
        # answerを受取側に送信
        self.queue.put(data)

    def send_error(
        self,
        e: Union[BaseException, Exception, KeyboardInterrupt, HTTPException],
        message: str = "申し訳ありません。もう少し表現を変えていただくか、再度お試しください。",
        status_code: int = None
    ):
        # トークン上限エラーをチェック
        if "maximum context length" in str(e):
            message = "文脈を読み込める上限に達しています。新規チャットで開きなおしてください。"

        if isinstance(e, HTTPException):
            if e.status_code == 500:
                message = "サーバー側でエラーが発生しました。\n 管理者へお問い合わせください。"
                status_code = e.status_code
                
        kwargs = {'e': e, 'message': message}
        if status_code is not None:
            kwargs['status_code'] = status_code
        
        self.queue.put(StreamErrorResponseData(**kwargs))
        print("error sent")

    def get(self) -> Union[StreamAnswerResponseData, Exception, KeyboardInterrupt, StopIteration]:
        return self.queue.get()

    def close(self):
        # Streamの終了を知らせる
        self.queue.put(StopIteration())
        print("answer stream closed")


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