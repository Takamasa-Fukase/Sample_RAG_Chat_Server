from typing import List
from data_models import ActionInfo, StreamAnswerResponseData, AnswerResponseQueue


class CallbackHandler():
    queue: AnswerResponseQueue

    def __init__(self, queue: AnswerResponseQueue):
        self.queue = queue

    def on_function_selected(self, action_prefix: str):
        print(f'on_function_selected\n - action_prefix: {action_prefix}')
        self.queue.send(StreamAnswerResponseData(
            answer_type_id=0,
            action_info=ActionInfo(
                action_prefix=action_prefix,
            ),
        ))

    def on_part_of_function_input_generated(self, text: str):
        print(f'on_part_of_function_input_generated\n - text: {text}')
        # 見た目には表示したくない出力の一覧（json形式の出力がバラバラに返ってくる＆都度表示する必要があるため、待ってからパースとかも不可なのでこの対応をしています）
        # TODO: - queryという引数名をハードコーディングではなく、すべてのfunctionの想定しうる引数名の一覧から撮ってくる様に後で変えた方が良い。
        not_output_token_list = ["}", "\"\n", " \"", "\":", "query", " \"", " ", "{\n", "", "index", "_data", "_search", "_query", "web"]
        if text in not_output_token_list:
            return
        self.queue.send(StreamAnswerResponseData(
            answer_type_id=0,
            action_info=ActionInfo(
                part_of_action_input_text=text,
            ),
        ))

    def on_function_input_generation_completed(self):
        print(f'on_function_input_generation_completed')
        self.queue.send(StreamAnswerResponseData(
            answer_type_id=4, # 4: action_input_generation_completed
        ))

    def on_source_url_list_extracted(self, url_list: List[str]):
        print(f'on_source_url_list_extracted\n - url_list: {url_list}')
        # Serperだとlinkが必ずしもあるわけじゃないので、空文字で入ってきたやつは除外する
        filtered_list = list(filter(lambda x: x != "", url_list))
        self.queue.send(StreamAnswerResponseData(
            answer_type_id=1,
            source_url_list=filtered_list,
        ))

    # 時間のかかるウェブスクレピング＆要約処理の進捗を表す値を0~100でアプリに送信する
    def on_web_contents_scraping_progress_updated(self, progress: int):
        print(f'on_web_contents_scraping_progress_updated - progress: {progress}')
        self.queue.send(StreamAnswerResponseData(
            answer_type_id=5,
            web_contents_scraping_progress=progress,
        ))

    def on_part_of_answer_generated(self, text: str):
        # _logger.debug(f'on_part_of_answer_generated\n - text: {text}')
        self.queue.send(StreamAnswerResponseData(
            answer_type_id=2,
            part_of_final_answer_text=text,
        ))