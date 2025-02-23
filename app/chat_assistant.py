import openai
import json
import re
import tiktoken
import logging
import asyncio

from typing import List, Optional

from langchain.vectorstores import VectorStore

from app.chat_gpt.assistant_function import AssistantFunctionType, parse_function_type_from_string
from app.chat_gpt.callback_handler import CallbackHandler
from app.chat_gpt.app_api_model import QuestionBody
from app.libraries.vector_indices.vector_index import VectorIndex
from app.libraries.repositories.usage_token import UsageTokenRepository
from app.models.agent import Agent
from app.models.company import Company
from app.settings.env import Env

_logger = logging.getLogger(__name__)

# pythonのOpenAIラッパーライブラリに環境変数からAPIキーをセットする
openai.api_key = Env.OPENAI_API_KEY

class ChatAssistant():
    usage_token_repo: UsageTokenRepository
    company: Company
    user_id: int
    callback_handler: CallbackHandler
    question_body: QuestionBody
    model_name: str
    temperature: int
    vector_store: VectorStore
    vector_index: Optional[VectorIndex]
    agent: Optional[Agent]
    functions = []
    messages = []
    # 質疑応答の入力に使用したトークン数を保持する
    prompt_token_count: int = 0
    # 質疑応答の出力に使用したトークン数を保持する
    completion_token_count: int = 0
    is_enabled_deep_web_search_mode: bool

    def __init__(
            self,
            usage_token_repo: UsageTokenRepository,
            company: Company,
            user_id: int,
            callback_handler: CallbackHandler,
            question_body: QuestionBody,
            model_name: str = 'gpt-3.5-turbo',
            temperature: int = 0.7,
            use_latest_information: bool = False,
            is_enabled_deep_web_search_mode: bool = False,
            use_index: bool = False,
            is_enabled_web_and_index_data_integrated_mode: bool = False,
            vector_index: Optional[VectorIndex] = None,
            agent: Optional[Agent] = None,
            system_role_prompt_text: Optional[str] = None,
        ):
        self.usage_token_repo = usage_token_repo
        self.company = company
        self.user_id = user_id
        self.callback_handler = callback_handler
        self.question_body = question_body
        self.model_name = model_name
        self.temperature = temperature
        self.functions.clear()
        self.messages.clear()
        self.prompt_token_count = 0
        self.completion_token_count = 0
        self.is_enabled_deep_web_search_mode = is_enabled_deep_web_search_mode

        # 受け取ったパラメータに合わせてfunctionの情報を配列に追加する
        if use_latest_information:
            self.functions.append(AssistantFunctionType.Search_On_Web.get_function_info())

        if use_index:
            self.functions.append(AssistantFunctionType.Search_On_Index_Data.get_function_info(company=company))
            # vector_indexとagentを元にvector_storeを取得してセットする（indexData検索の場合に使う）
            self.vector_store = vector_index.get_vector_store(agent)

        if is_enabled_web_and_index_data_integrated_mode:
            self.functions.clear() # 他のfunctionを消して統合検索だけにする
            self.functions.append(AssistantFunctionType.Search_On_Web_And_Index_Data.get_function_info(company=company))
        
        # もしsystem_role_prompt_textがあった場合は1番目のmessageとして挿入しておく
        if system_role_prompt_text:
            _logger.debug(f'system_role_prompt_textがある:\n{system_role_prompt_text}')
            self.messages.append({
                "role": "system",
                "content": system_role_prompt_text
            })
        
        
    def get_answer(self):
        # 会話履歴を文脈に追加する
        previous_messages = self._make_history(previous_messages=self.question_body.previous_messages)
        self.messages.extend(previous_messages)

        # ユーザーからの入力を文脈に格納する
        self.messages.append({
            "role": "user",
            "content": self.question_body.text
        })

        #【トークン計算処理】回答を格納する前の時点のmessagesがprompt扱いのトークン数なので加算しておく
        self.prompt_token_count += self._get_tokens_from_messages(messages=self.messages)
        #【トークン計算処理】渡しているfunctionsのトークン数もprompt扱いのトークン数なので加算しておく
        self.prompt_token_count += self._get_tokens_from_functions(functions=self.functions)

        # 暫定対応 もっと良いやり方があれば直したい
        # （リクエスト箇所でfunctionsを使わない場合、空配列もNoneもNGで、キー自体を落とさないといけないのでやむなく分岐している）
        if self.functions:
            # 1回目のリクエストを送信（functionsあり）
            streamed_response = openai.ChatCompletion.create(
                model=self.model_name,
                # 回答のランダム性（0から1の範囲で設定可能）
                temperature=self.temperature,
                # 文脈情報を渡す（[system_roleでのプロンプト指示（任意）, これまでの会話, 今回のユーザー入力]）
                messages=self.messages,
                # 呼び出し可能なfunctionsとして受け渡す
                functions=self.functions,
                # autoの場合、「functionが必要かどうか、どのfunctionが必要か」をGPTが自動で判断する設定を適用
                function_call='auto',
                stream=True,
            )
        else:
            # 1回目のリクエストを送信（functionsなし）
            streamed_response = openai.ChatCompletion.create(
                model=self.model_name,
                # 回答のランダム性（0から1の範囲で設定可能）
                temperature=self.temperature,
                # 文脈情報を渡す（[system_roleでのプロンプト指示（任意）, これまでの会話, 今回のユーザー入力]）
                messages=self.messages,
                stream=True,
            )

        collected_messages = []
        is_function_call = False
        function_type: AssistantFunctionType = None

        # Streamのレスポンスを順番に処理する
        for chunk in streamed_response:
            # 断片として受け取ったオブジェクトを取り出して配列に格納（最終回答もしくは呼びたいfunctionの情報などが断片で送られてくる）
            chunk_message = chunk['choices'][0]['delta']
            collected_messages.append(chunk_message)

            # functionの呼び出しを要求しているレスポンスの場合
            if (function_call_object := chunk_message.get('function_call')) is not None:
                # function自体の実行は引数の入力値を全て受け取った後なので、後でfunction_callかどうか判定できる様にフラグをTrueにしておく
                is_function_call = True
                if not function_type:
                    # GPTから実行を要求されたfunctionがどれかわかる様に保持しておく
                    function_type = parse_function_type_from_string(function_name=function_call_object.get('name'))
                    # function_callが選ばれた時点でアプリに処理工程を表示するためにcallbackを呼ぶ。「外部データを検索」「自社データから検索」など
                    self.callback_handler.on_function_selected(action_prefix=function_type.action_prefix)

                # 「〜を検索」の後に続いて「検索する内容」をstreamでアプリに表示するためにcallbackを呼ぶ
                self.callback_handler.on_part_of_function_input_generated(text=function_call_object.get('arguments'))

            # 通常の返答レスポンスの場合
            else:
                # function_callじゃない場合はそれが最終回答になるので、streamでアプリに表示するためにcallbackを呼ぶ
                if (content := chunk_message.get('content')) is not None:
                    self.callback_handler.on_part_of_answer_generated(text=content)

        # function_callが呼ばれている場合
        if is_function_call:
            # アプリにアクション情報の出力が完了したことを通知
            self.callback_handler.on_function_input_generation_completed()

            # function_callの情報のjsonが断片で送られてくるため、arguments部分を配列から取り出してjson形式の文字列に戻す
            full_reply_arguments_text = ''.join([chunk_message.get('function_call', {}).get('arguments', '') for chunk_message in collected_messages])
            _logger.debug(f"get_answer function_callの場合 すべてのレスポンスを受け取った:\n - full_reply_arguments_text: {full_reply_arguments_text}")

            completion_message = {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "arguments": full_reply_arguments_text,
                    "name": function_type.value
                }
            }

            #【トークン計算処理】回答はcompletion扱いトークン数なので加算しておく
            self.completion_token_count += self._get_tokens_from_messages(messages=[completion_message])

            # assistantからの返答を文脈に追加
            self.messages.append(completion_message)

            # 選択されたfunctionの処理の実行を委託し、得られた参考情報を含んだ文脈情報を元に当初のユーザーからの入力に対して再度応答させる
            asyncio.run(
                self._get_second_answer(
                    selected_function_type=function_type,
                    full_reply_arguments_text=full_reply_arguments_text
                )
            )
            
        else:
            # 返答が断片で送られてくるため、配列から取り出して連結した文字列に戻す
            full_reply_content = ''.join([chunk_message.get('content', '') for chunk_message in collected_messages])
            _logger.debug(f"get_answer function_callじゃない場合 すべてのレスポンスを受け取った:\n - full_reply_content: {full_reply_content}")

            completion_message = {
                "role": "assistant",
                "content": full_reply_content,
            }

            #【トークン計算処理】回答はcompletion扱いトークン数なので加算しておく
            self.completion_token_count += self._get_tokens_from_messages(messages=[completion_message])

            # assistantからの返答を文脈に追加
            self.messages.append(completion_message)

            # 【最終トークンDB保存処理】最終回答が得られたのでこの時点で今回のリクエストに対するユーザーの使用トークンをDB保存する。
            self._add_token_usage()


    # function_callが要求された場合に最終回答を生成させるために使う
    async def _get_second_answer(
            self,
            selected_function_type: AssistantFunctionType,
            full_reply_arguments_text: str
        ):
        # 選択されたFunctionの処理を実行し、結果の文字列を取得
        function_response_text = await self._execute_selected_function(
            function_type=selected_function_type,
            full_reply_arguments_text=full_reply_arguments_text
        )

        # functionの結果として得られた参考情報を文脈に追加
        self.messages.append({
            "role": "function",
            "name": selected_function_type.value,
            "content": function_response_text,
        })

        #【トークン計算処理】回答を格納する前の時点のmessagesがprompt扱いのトークン数なので加算しておく
        self.prompt_token_count += self._get_tokens_from_messages(messages=self.messages)

        streamed_second_response = openai.ChatCompletion.create(
            model=self.model_name,
            temperature=self.temperature,
            # 文脈情報を渡す（[system_roleでのプロンプト指示（任意）, これまでの会話, 今回のユーザー入力, function_call情報, functionによって取得された参考情報]）
            messages=self.messages,
            stream=True,
        )

        collected_messages = []
        for chunk in streamed_second_response:
            chunk_message = chunk['choices'][0]['delta']
            collected_messages.append(chunk_message)
            # streamでアプリに表示するためにcallbackを呼ぶ
            if (content := chunk_message.get('content')) is not None:
                self.callback_handler.on_part_of_answer_generated(text=content)

        # 返答が断片で送られてくるため、配列から取り出して連結した文字列に戻す
        full_reply_content = ''.join([chunk_message.get('content', '') for chunk_message in collected_messages])
        _logger.debug(f"_get_second_answer すべてのレスポンスを受け取った:\n - full_reply_content: {full_reply_content}")

        completion_message = {
            "role": "assistant",
            "content": full_reply_content,
        }

        #【トークン計算処理】回答はcompletion扱いトークン数なので加算しておく
        self.completion_token_count += self._get_tokens_from_messages(messages=[completion_message])

        # assistantからの返答を文脈に追加
        self.messages.append(completion_message)

        #【最終トークンDB保存処理】最終回答が得られたのでこの時点で今回のリクエストに対するユーザーの使用トークンをDB保存する。
        self._add_token_usage()
    

    async def _execute_selected_function(
            self, 
            function_type: AssistantFunctionType, 
            full_reply_arguments_text: str
        ) -> str:
        # json形式に戻す
        arguments = json.loads(full_reply_arguments_text)
        # 外部データ検索の場合
        if function_type == AssistantFunctionType.Search_On_Web:
            # GPTから文脈を踏まえた上で引数として渡された検索クエリを元に外部データ検索結果を取得する
            function_response = await AssistantFunctionType.Search_On_Web(
                query=arguments.get('query'),
                callback_handler=self.callback_handler,
                is_enabled_deep_search_mode=self.is_enabled_deep_web_search_mode,
            )
            print(f'function_response: {function_response}')
            source_url_list = function_response[0]
            # 検索結果のURLリストを参考文献としてアプリに表示するためにcallbackを呼ぶ
            self.callback_handler.on_source_url_list_extracted(source_url_list)
            function_response_text = function_response[1]
        
        # 組織内データ検索の場合
        elif function_type == AssistantFunctionType.Search_On_Index_Data:
            # GPTから文脈を踏まえた上で引数として渡された検索クエリを元に組織内データ検索結果を取得する
            function_response_text = AssistantFunctionType.Search_On_Index_Data(
                query=arguments.get('query'),
                vector_store=self.vector_store,
            )
        
        # 組織内外データ統合検索の場合
        elif function_type == AssistantFunctionType.Search_On_Web_And_Index_Data:
            # GPTから文脈を踏まえた上で引数として渡された検索クエリを元に組織内外データ統合検索結果を取得する
            function_response = await AssistantFunctionType.Search_On_Web_And_Index_Data(
                index_data_search_query=arguments.get('index_data_search_query', ''),
                web_search_query=arguments.get('web_search_query', ''),
                vector_store=self.vector_store,
                callback_handler=self.callback_handler,
                is_enabled_deep_search_mode=self.is_enabled_deep_web_search_mode,
            )
            print(f'function_response: {function_response}')
            source_url_list = function_response[0]
            # 検索結果のURLリストを参考文献としてアプリに表示するためにcallbackを呼ぶ
            self.callback_handler.on_source_url_list_extracted(source_url_list)
            function_response_text = function_response[1]

        # 結果の文字列を返却
        return function_response_text


    def _make_history(self, previous_messages: List[str]):
        histories = []
        for message in previous_messages:
            if message.startswith('Human:'):
                histories.append({
                    "role": "user",
                    "content": self._remove_author_prefix(message)
                })
            elif message.startswith('AI:'):
                histories.append({
                    "role": "assistant",
                    "content": self._remove_author_prefix(message)
                })
        return histories


    def _remove_author_prefix(self, text):
        return re.sub(r'^(AI:|Human:)\s*', '', text)
    

    def _count_tokens(self, text: str) -> int:
        token_encoder = tiktoken.encoding_for_model(self.model_name)
        encoded_tokens = token_encoder.encode(text)
        return len(encoded_tokens)
    

    # 1回目のリクエスト時に渡すfunctionsの情報のトークン数を取得する
    # functionsも実は内部でtypescriptのフォーマットに変換されてsystem_messageに埋め込まれるため、結果として入力トークンとして扱われている模様。それを力技で再現して計算する必要がある。
    # 参考: https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/10#:~:text=Python%20function%20if%20anyone%20needs%20it%3A
    def _get_tokens_from_functions(self, functions):
        """Return the number of tokens used by a list of functions."""
        # functionsが空の時は0を返却
        if not functions:
            return 0
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            _logger.debug("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        
        num_tokens = 0
        for function in functions:
            function_tokens = len(encoding.encode(function['name']))
            function_tokens += len(encoding.encode(function['description']))
            
            if 'parameters' in function:
                parameters = function['parameters']
                if 'properties' in parameters:
                    for propertiesKey in parameters['properties']:
                        function_tokens += len(encoding.encode(propertiesKey))
                        v = parameters['properties'][propertiesKey]
                        for field in v:
                            if field == 'type':
                                function_tokens += 2
                                function_tokens += len(encoding.encode(v['type']))
                            elif field == 'description':
                                function_tokens += 2
                                function_tokens += len(encoding.encode(v['description']))
                            elif field == 'enum':
                                function_tokens -= 3
                                for o in v['enum']:
                                    function_tokens += 3
                                    function_tokens += len(encoding.encode(o))
                            else:
                                _logger.debug(f"Warning: not supported field {field}")
                    function_tokens += 11

            num_tokens += function_tokens

        num_tokens += 12 
        return num_tokens
    

    def _get_tokens_from_messages(self, messages: List) -> int:
        texts_of_messages = []
        for message in messages:
            # function_callキーが存在する場合
            if (function_call_value := message.get('function_call')) is not None:
                # 文字列に変換して追加する
                texts_of_messages.append(f'{function_call_value}')

            # それ以外の場合でcontentキーが存在する場合
            elif (content_text := message.get('content')) is not None:
                texts_of_messages.append(content_text)
            
        total_token_count = self._count_tokens(''.join(texts_of_messages))
        return total_token_count


    # 質疑応答が完了した時点での今回のトークン使用量をDBに保存する
    def _add_token_usage(self):
        total_token_count = self.prompt_token_count + self.completion_token_count
        _logger.info(f"usage_token_count\n - prompt: {self.prompt_token_count}\n - completion: {self.completion_token_count}\n - total: {total_token_count}")
        self.usage_token_repo.add_tokens(self.company, self.user_id, total_token_count)