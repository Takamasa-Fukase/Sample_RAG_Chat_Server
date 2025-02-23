from typing import List, Optional, Union
from pydantic import BaseModel


class SendQuestionRequest(BaseModel):
    category_id: int
    text: str
    previous_messages: List[str]


class ActionInfo(BaseModel):
    action_prefix: Optional[str]  # 外部データから検索しています / 自社データから検索しています / 文脈を整理しています　など
    part_of_action_input_text: Optional[str]  # LLMがtokenという単位で出力する断片的な文字列のうち、action_inputのもの


# Streamの中で下記3パターンのtypeの値をアプリに渡すための共通クラス
class StreamAnswerResponseData(BaseModel):
    answer_type_id: int  # 0: action_info, 1:source_url_list, 2: part_of_final_answer_text, 3: approaching_answer_text, 4: action_input_generation_completed, 5: web_contents_scraping_progress
    action_info: Optional[ActionInfo]
    source_url_list: Optional[List[str]]
    part_of_final_answer_text: Optional[str]  # LLMがtokenという単位で出力する断片的な文字列のうち、最終回答用のもの
    status_code: Optional[int]
    web_contents_scraping_progress: Optional[int]


class StreamErrorResponseData:
    e: Union[BaseException, Exception, KeyboardInterrupt]
    message: str
    status_code: Optional[int]

    def __init__(
        self,
        e: Union[BaseException, Exception, KeyboardInterrupt],
        message: str,
        status_code : Optional[int] = None
        ):
        self.e = e
        self.message = message
        self.status_code = status_code