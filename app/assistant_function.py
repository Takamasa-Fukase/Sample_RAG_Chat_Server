from enum import Enum
from typing import List, Optional, Union
from langchain.vectorstores import VectorStore
from callback_handler import CallbackHandler
from google_serper import CustomGoogleSerper
from web_contents_scraper import WebContentsScraper

class Dnum(Enum):
    """
    Dispatching Enum。これを継承する
    """
    @classmethod
    def register(cls, key):
        if not hasattr(cls, "table"):
            cls.table = {}
        def registration(func):
            cls.table[key] = func
        return registration

    def __call__(self, *args, **kwargs):
        return self.__class__.table[self](*args, **kwargs)


def register(enum):
    def ret(func):
        enum.__class__.register(enum)(func)
    return ret


# TODO: 後でEnumをやめてAssistantFunctionというクラスなどにしたい
class AssistantFunctionType(Dnum):
    Search_On_Web = 'search_on_web'
    Search_On_Index_Data = 'search_on_index_data'
    Search_On_Web_And_Index_Data = 'search_on_web_and_index_data'

    def get_function_info(self):
        if self == AssistantFunctionType.Search_On_Web:
            return {
                "name": "search_on_web",
                "description": "It can be used to answer about the latest topics. You can also check today's date, today's temperature, weather, exchange rates, and other current conditions. The input is the search content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "検索したい内容を入力。",
                        },
                    },
                    "required": ["query"],
                },
            }
        elif self == AssistantFunctionType.Search_On_Index_Data:
            return {
                "name": "search_on_index_data",
                "description": "ウルトラ深瀬に関する質問の場合に役に立ちます。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "検索したい内容を入力。",
                        },
                    },
                    "required": ["query"],
                },
            }
        elif self == AssistantFunctionType.Search_On_Web_And_Index_Data:
            return {
                "name": "search_on_web_and_index_data",
                "description": "ウルトラ深瀬に関する様々な情報と、それ以外の外部の情報やインターネット上の最新情報を比較したり、ウルトラ深瀬の情報と外部情報を統合して新たな情報として分析したい場合などに役に立ちます。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_data_search_query": {
                            "type": "string",
                            "description": "ウルトラ深瀬に関する様々な情報から検索したい内容を入力。"
                        },
                        "web_search_query": {
                            "type": "string",
                            "description": "外部の情報やインターネット上の最新情報から検索したい内容を入力。"
                        }
                    },
                    "required": ["index_data_search_query", "web_search_query"],
                },
            }
        else:
            return ''
        
    @property
    def action_prefix(self) -> str:
        if self == AssistantFunctionType.Search_On_Web:
            return '外部データから検索しています'
        elif self == AssistantFunctionType.Search_On_Index_Data:
            return '組織内データから解析しています'
        elif self == AssistantFunctionType.Search_On_Web_And_Index_Data:
            return '組織内外のデータを統合検索しています'
        else:
            return ''


@register(AssistantFunctionType.Search_On_Web)
async def _Search_On_Web(
    query: str,
    callback_handler: CallbackHandler,
    is_enabled_deep_search_mode: bool,
) -> (List[str], str): # 戻り値のタプル　1つ目: リンクの配列、2つ目： 参考情報の文字列
    print(f'_Search_On_Web query: {query}')
    result = await search_on_google_serper(
        query=query,
        callback_handler=callback_handler,
        is_enabled_deep_search_mode=is_enabled_deep_search_mode,
    )
    print(f'result: {result}')
    return result

@register(AssistantFunctionType.Search_On_Index_Data)
def _Search_On_Index_Data(
    query: str,
    vector_store: VectorStore,
) -> str:
    print(f'Search_On_Index_Data query: {query}')
    documents = vector_store.similarity_search(
        query=query,
        # 取り出すドキュメントの上位⚪︎件の値。関係ない情報が回答に紛れ込まない様に上位1件だけに設定。
        k=1
    )
    # 今はk=1にしていて結果は1つなので連結する必要はないが今後kの値を複数にする可能性もありえるのでループで連結させている
    documents_text = ''.join([doc.page_content for doc in documents])
    print(f'documents_text: {documents_text}')
    return documents_text

# 組織内外データ統合検索でも基本的には外部データ検索と同じ型の戻り値、流れで処理を行う。
@register(AssistantFunctionType.Search_On_Web_And_Index_Data)
async def _Search_On_Web_And_Index_Data(
    index_data_search_query: str,
    web_search_query: str,
    vector_store: VectorStore,
    callback_handler: CallbackHandler,
    is_enabled_deep_search_mode: bool,
) -> (List[str], str): # 戻り値のタプル　1つ目: リンクの配列、2つ目： 参考情報の文字列
    print(f'Search_On_Web_And_Index_Data index_data_search_query: {index_data_search_query}, web_search_query: {web_search_query}')

    # 組織内データ検索を行う（現状ベクトル検索は時間のかかる処理では無いので一旦同期処理でOK）
    index_data_search_result = vector_store.similarity_search(
        query=index_data_search_query,
        # 取り出すドキュメントの上位⚪︎件の値。関係ない情報が回答に紛れ込まない様に上位1件だけに設定。
        k=1
    )
    # 外部データ検索を行う
    web_search_result = await search_on_google_serper(
        query=web_search_query,
        callback_handler=callback_handler,
        is_enabled_deep_search_mode=is_enabled_deep_search_mode,
    )

    # 組織内データ検索結果のドキュメント配列を結合して文字列にする（今はk=1にしていて結果は1つなので連結する必要はないが今後kの値を複数にする可能性もありえるのでループで連結させている）
    documents_text = ''.join([doc.page_content for doc in index_data_search_result])
    # インデックスデータ検索結果の文字列を、外部データ検索結果の文字列と結合する。
    # また、両者を言い感じに比較してる風の回答をさせるために、ここで回答指示を追加して挙動をコントロールしている。
    web_and_index_data_integrated_result_text = f'''
    #命令:「組織内データから取得した情報」と、「外部データから取得した情報」の両方を相互に比較し、その比較結果をまず出力するとともにそれを踏まえた上でユーザーの元の質問に答えて下さい。
    #組織内データから取得した情報:{documents_text}
    #外部データから取得した情報:{web_search_result[1]}
    '''
    print(f'web_and_index_data_integrated_result_text: {web_and_index_data_integrated_result_text}')
    # 両者の文字列を結合した上で、（リンク配列, 結果の文字列）の形式のタプルにして返却
    return (web_search_result[0], web_and_index_data_integrated_result_text)

def parse_function_type_from_string(function_name: str) -> AssistantFunctionType:
    if function_name == AssistantFunctionType.Search_On_Web.value:
        return AssistantFunctionType.Search_On_Web
    elif function_name == AssistantFunctionType.Search_On_Index_Data.value:
        return AssistantFunctionType.Search_On_Index_Data
    elif function_name == AssistantFunctionType.Search_On_Web_And_Index_Data.value:
        return AssistantFunctionType.Search_On_Web_And_Index_Data
    else:
        print(f'想定外のfunction_name: {function_name}')

async def search_on_google_serper(
    query: str,
    callback_handler: CallbackHandler,
    is_enabled_deep_search_mode: bool,
) -> (List[str], str):
    result = CustomGoogleSerper().run(query=query)

    # ディープサーチのON/OFFに関わらず、AnswerBoxかKnowledgeGraphの値が取れている場合はそれだけで十分な情報なのでそのまま参考情報として返す。Linkのスクレイピング＆要約はしない。
    if result.answer_box or result.knowledge_graph:
        print(f"🟨ディープサーチは{is_enabled_deep_search_mode}だが、AnswerBoxかKnowledgeGraphの値が取れている場合 それだけで十分な情報なのでそのまま参考情報として返す。")
        result_text = '\n\n'.join([result.answer_box, result.knowledge_graph])
        # この場合はリンク先の情報は参考にしていないが、UI上で表示した方がリッチな見た目になる為リンクも返却する
        return (result.links, result_text)

    # ディープサーチがONの場合
    elif is_enabled_deep_search_mode:
        print("🟥ディープサーチがONの場合 検索結果上位3件のリンクが渡されるのでスクレイピング&要約して返す。")
        # AnswerBoxもKnowledgeGraphも取れなかった場合は通常の検索結果上位3件のリンクが渡されるのでスクレイピング＆要約して返す。
        if result.links:
            scraper = WebContentsScraper(
                links=result.links,
                query=query,
                callback_handler=callback_handler,
            )
            summary = await scraper.create_summary_from_links()
            # この場合は各リンクの表示とともに、スクレイピングした回答も参考情報として渡す
            return (result.links, summary)

        # 何も取れなかった場合は空で返す。スクレピング＆要約もしない。
        else:
            return ([], "")
    
    # ディープサーチがOFFの場合
    else:
        # この場合は各リンクの表示とともに、検索結果に含まれているsnippet（浅い情報だが）を含めた文字列を返却する
        print(f"🟦ディープサーチがOFFの場合 この場合は各リンクの表示とともに、検索結果に含まれているsnippet(浅い情報だが)を含めた文字列を返却する: {(result.links, result.organic_results_text)}")
        return (result.links, result.organic_results_text)