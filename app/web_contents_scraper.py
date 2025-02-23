import openai
import tiktoken
import requests
import asyncio
import math
from typing import Callable
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from env import Env
from callback_handler import CallbackHandler


# pythonのOpenAIラッパーライブラリに環境変数からAPIキーをセットする
openai.api_key = Env.OPENAI_API_KEY


class WebContentsScraper():
    # 0~100で全体の進捗を表す値（ここに加算していく）
    progress = 0

    # 1つの処理の完了に対して加算する進捗の値
    each_process_value: int

    links: [str]
    query: str
    callback_handler: CallbackHandler

    def __init__(
        self,
        links: [str],
        query: str,
        callback_handler: CallbackHandler,
    ):
        # 計算式：(100 ÷ (_create_summary()内の主な処理の数「3」✖️ linkの数)）を少数切り捨てした整数（linkが3件なら11）
        # 表示を簡素化する為に整数に丸めている関係でそれぞれの処理が全て終わっても100にはならないが、
        # asyncio.gather()のawaitが終わった時点で明示的に100でイベントを流すのでそこで整合性が取れる
        self.each_process_value = math.floor((100 / (3 * len(links))))
        self.links = links
        self.query = query
        self.callback_handler = callback_handler


    # 外部データ検索で取得した各リンク（上位3件）に対して行いたい処理を並列実行させる為の関数
    async def create_summary_from_links(self) -> str:
        # 各処理の完了時に行いたい処理
        def on_update_progress():
            # クラスの初期化時に計算した、各処理ごとに割り当てられた進捗の値を加算する
            self.progress += self.each_process_value
            # 加算された値（更新後の値）でアプリに進捗を通知するために、コールバックを呼ぶ
            self.callback_handler.on_web_contents_scraping_progress_updated(progress=self.progress)

        # リンクの数だけ非同期処理のタスクを生成する
        tasks = [
            self._create_summary(
                link, 
                self.query, 
                on_update_progress, # 上記で定義した「各処理の完了時に行いたい処理」を注入する
            ) for link in self.links
        ]
        # 非同期処理を開始するので、progress=0としてアプリに通知し、進捗表示用の吹き出しを表示させる
        self.callback_handler.on_web_contents_scraping_progress_updated(progress=0)
        # 非同期処理を並列実行する
        # （return_exceptions=Trueについて：一部の処理で例外が発生した場合でも他の処理を続行させ、最終的なすべての結果の中で一緒に例外も受け取れる様にしている）
        summaries = await asyncio.gather(*tasks, return_exceptions=True)
        # 上記の非同期処理が完了したので、progress=100で明示的にアプリに完了を通知する
        self.callback_handler.on_web_contents_scraping_progress_updated(progress=100)

        # 例外が含まれている可能性があるので、strだけにフィルターする
        # MEMO: - サイトによってはアクセスやスクレイピングに失敗する場合があったので
        filtered_summaries = list(filter(lambda x: isinstance(x, str), summaries))

        # 各サイトの結果の文字列を結合して1つの文字列にする
        result = '\n'.join(filtered_summaries)
        print(f'⭐️⭐️⭐️create_summary_from_linksの最終結果:\n{result}')
        return result


    # 1つのリンクに対して行わせたい処理をまとめた関数
    async def _create_summary(
        self,
        link: str, 
        query: str,
        on_update_progress: Callable[..., None],
    ):
        print(f'⭐️{link}に対する_create_summary()処理を開始')

        content = await self._get_content_from_link(link)
        print(f' - {link}のコンテンツ抽出完了')
        on_update_progress()

        cleaned_content = self._clean_content(content)
        print(f' - {link}から抽出したコンテンツのクリーン完了')
        on_update_progress()

        summary = await self._summarize_content(cleaned_content, query)
        print(f' - {link}のクリーン済みコンテンツの要約完了')
        on_update_progress()

        return f'## ({link})から抽出したコンテンツの要約文章: {summary}'


    # リンク先のHTMLコンテンツを全て抽出
    async def _get_content_from_link(
        self, 
        link: str
    ) -> str:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, requests.get, link)
        # エラーレスポンスの場合は例外を発生させる
        response.raise_for_status()
        return response.text


    # 抽出したHTMLコンテンツの中から欲しい情報だけにフィルタリングする
    def _clean_content(
        self,
        content: str
    ) -> str:
        bs_transformer = BeautifulSoupTransformer()
        transformed_docs = bs_transformer.transform_documents(
            documents=[Document(page_content=content)],
            # 除外したいHTMLタグを指定
            unwanted_tags=["nav", "header", "footer", "script", "style"],
            # 抽出したいHTMLタグを指定（汎用的に指定する必要がある。暫定でこの設定値にしている。改善の余地あり）
            tags_to_extract=["div", "span"]
        )
        # 最初の10000トークン分だけを取り出す（トークン制限の問題もあって無限にコンテンツを取得しても結局使えないので）
        # かといって、ここで十分な量を確保しないと深い回答に繋がる参考情報は取れないので、暫定で10000にしている。ここはあまりケチるべきでは無いと思っています。
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=10000,
                                                                        chunk_overlap=0)
        splits = splitter.split_documents(documents=transformed_docs)
        return splits[0].page_content   


    # クリーニングしたコンテンツの中から元の質問分に関連する部分を抽出させつつ、500文字以内に要約させる
    async def _summarize_content(
        self,
        content: str,
        query: str,
    ) -> str:
        token_count = len(tiktoken.encoding_for_model('gpt-3.5-turbo-16k').encode(content))
        # 元から500token以下の場合は要約せずにそのまま返す
        if token_count <= 500:
            return content
        else:
            # 非同期処理を行える様にacreate()（async createのこと）の方のメソッドを使用している
            response = await openai.ChatCompletion.acreate(
                model='gpt-3.5-turbo-16k', # 莫大なサイズの参考情報を一度に処理するために16kモデルを使用する
                temperature=0, # 情報の抽出にランダム性は不要なので固定で0にしている
                max_tokens=500, # 出力サイズを小さくすることで処理時間の短縮を図っている
                messages=[{
                    "role": "user",
                    "content": f"## 命令文:対象の文章に関して、{query}という質問に関連する文章を抽出してください。\n\n## 対象の文章:{content}"
                }],
            )
            return response["choices"][0]["message"]["content"]